from __future__ import division

import math
import pickle
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import dataset
from find_contours import findmaxcontours
from fpn import AutoScale
from image import *
from rate_model import RATEnet

warnings.filterwarnings('ignore')
from config import args
import  os
import imageio

torch.cuda.manual_seed(args.seed)


def main():

    if args.test_dataset == 'ShanghaiA':
        test_file = './ShanghaiA_test.npy'
    elif args.test_dataset == 'ShanghaiB':
        test_file = './ShanghaiB_test.npy'
    elif args.test_dataset =='UCF_QNRF':
        test_file = './Qnrf_test.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    model = AutoScale()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    rate_model = RATEnet()
    rate_model = nn.DataParallel(rate_model, device_ids=[0]).cuda()

    pickle.load = partial(pickle.load, encoding="iso-8859-1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="iso-8859-1")

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            # checkpoint = torch.load(args.pre, map_location=lambda storage, loc: storage, pickle_module=pickle)
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            rate_model.load_state_dict(checkpoint['rate_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # torch.save({
    #         'state_dict': model.state_dict(),
    #         'rate_state_dict': rate_model.state_dict()
    #     }, "./model/ShanghaiA/model_best.pth")

    validate(val_list, model, rate_model, args)



def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]

    d1 = d[:, :, abs(int(math.floor((d_h - g_h) / 2.0))):abs(int(math.floor((d_h - g_h) / 2.0))) + g_h,
         abs(int(math.floor((d_w - g_w) / 2.0))):abs(int(math.floor((d_w - g_w) / 2.0))) + g_w]
    return d1


def choose_crop(output, target):
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    return output, target


def distance_generate(img_size, k, lamda, crop_size):
    distance = 1.0
    new_size = [0, 1]

    new_size[0] = img_size[2] * lamda
    new_size[1] = img_size[3] * lamda

    d_map = (np.zeros([int(new_size[0]), int(new_size[1])]) + 255).astype(np.uint8)
    gt = np.nonzero(k)

    if len(gt) == 0:
        distance_map = np.zeros([int(new_size[0]), int(new_size[1])])
        distance_map[:, :] = 10

        x = int(crop_size[0] * lamda)
        y = int(crop_size[1] * lamda)
        w = int(crop_size[2] * lamda)
        h = int(crop_size[3] * lamda)

        distance_map = distance_map[y:(y + h), x:(x + w)]

        return new_size, distance_map

    # print(k,gt_data,gt,lamda)
    for o in range(0, len(gt)):
        x = int(max(1, gt[o][1].numpy() * lamda))
        y = int(max(1, gt[o][2].numpy() * lamda))
        # print(len(gt),x,y)
        if x >= new_size[0] - 1 or y >= new_size[1] - 1:
            # print(o)
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 5)
    #distance_mask = distance_map.copy()

    distance_map[(distance_map >= 0) & (distance_map < 1)] = 0
    distance_map[(distance_map >= 1) & (distance_map < 2)] = 1
    distance_map[(distance_map >= 2) & (distance_map < 3)] = 2
    distance_map[(distance_map >= 3) & (distance_map < 4)] = 3
    distance_map[(distance_map >= 4) & (distance_map < 5 * distance)] = 4
    distance_map[(distance_map >= 5 * distance) & (distance_map < 6 * distance)] = 5
    distance_map[(distance_map >= 6 * distance) & (distance_map < 8 * distance)] = 6
    distance_map[(distance_map >= 8 * distance) & (distance_map < 12 * distance)] = 7
    distance_map[(distance_map >= 12 * distance) & (distance_map < 18 * distance)] = 8
    distance_map[(distance_map >= 18 * distance) & (distance_map < 28 * distance)] = 9
    distance_map[(distance_map >= 28 * distance)] = 10

    #print('time cost', time_end - time_start, 's')
    #mask, distance_map = mask_generate(distance_map,distance_mask)
    x = int(crop_size[0]*lamda)
    y = int(crop_size[1]*lamda)
    w = int(crop_size[2]*lamda)
    h = int(crop_size[3]*lamda)

    distance_map = distance_map[y:(y + h), x:(x + w)]

    # Distance_map = distance_map / np.max(distance_map) * 255
    # cv2.imwrite("1_dis.jpg",Distance_map)

    return new_size, distance_map


def count_distance(input_img):
    input_img = input_img.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    imageio.imsave('./distance_map.pgm', input_img)
    f = os.popen('./count_localminma/extract_local_minimum_return_xy ./distance_map.pgm 256 ./distance_map.pgm distance_map.pp')
    count = f.readlines()

    count = float(count[0].split('=')[1])

    return count


def validate(Pre_data, model, rate_model, args):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args.task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0.0
    mse = 0.0
    original_mae = 0.0
    visi = []


    for i, (fname, img, target, kpoint,sigma)  in enumerate(test_loader):
        img_size = img.size()
        count_crop = 0
        count_other = 0
        img = img.cuda()
        target = target.type(torch.LongTensor)

        d0, d1, d2, d3, d4, d5, scales_feature = model(img, target,refine_flag=True)
        original_distance_map = torch.max(F.softmax(d5), 1, keepdim=True)[1]
        crop_size, crop_size_second, crop_size_third,contours = findmaxcontours(original_distance_map.data.cpu().numpy(),
                                                                       find_max=True, fname=fname)
        original_count = count_distance(original_distance_map)

        scale_crop = scales_feature[:, :, crop_size[1]:(crop_size[1] + crop_size[3]),
                     crop_size[0]:(crop_size[0] + crop_size[2])]

        scale_crop = F.adaptive_avg_pool2d(scale_crop, (14, 14))

        rate_feature = scale_crop
        rate_list = rate_model(rate_feature)
        rate_list.clamp_(0.5, 5)

        rate = torch.sqrt(rate_list)

        distance_map_gt_crop = distance_generate(img_size, kpoint, rate.item(), crop_size)[1]
        distance_map_gt_crop = torch.from_numpy(distance_map_gt_crop).unsqueeze(0).type(torch.LongTensor)

        if (float(crop_size[2] * crop_size[3]) / (img_size[2] * img_size[3])) > args.area_threshold :

            img_crop = img[:, :, crop_size[1]:(crop_size[1] + crop_size[3]),
                       crop_size[0]:(crop_size[0] + crop_size[2])]
            img_crop = F.upsample_bilinear(img_crop,
                                           (int(img_crop.size()[2] * rate),
                                            int(img_crop.size()[3] * rate)))

            dd0, dd1, dd2, dd3, dd4, dd5 = model(img_crop,  distance_map_gt_crop, refine_flag=False)

            dd5 = torch.max(F.softmax(dd5), 1, keepdim=True)[1]
            count_crop = count_distance(dd5)

            original_distance_map[:, :, crop_size[1]:(crop_size[1] + crop_size[3]),
            crop_size[0]:(crop_size[0] + crop_size[2])] = 10
            count_other = count_distance(original_distance_map)

        else:
            count_crop = original_count
            count_other = 0

        count = count_crop + count_other
        Gt_count = torch.sum(kpoint).item()
        mae += abs( count- Gt_count)
        mse += abs(count - Gt_count) * abs(count - Gt_count)
        original_mae += abs(original_count - Gt_count)

        if i % args.print_freq == 0:
            print(fname[0], 'rate {rate:.3f}'.format(rate=rate.item()), 'gt', torch.sum(kpoint).item(),
                  "pred", int(count), "original:", int(original_count))

    mae = mae * 1.0/ len(test_loader)
    mse = math.sqrt(mse/len(test_loader))
    original_mae = original_mae / len(test_loader)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}\n'.format(mse=mse),'* ORI_MAE {ori_mae:.3f}\n'.format(ori_mae=original_mae))

    return mae, original_mae, visi



if __name__ == '__main__':
    main()
