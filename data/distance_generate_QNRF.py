import numpy as np
import cv2
import math
import scipy.io
import os
import h5py
import scipy.misc
import time
import scipy.spatial

root = '/home/dkliang/projects/synchronous/UCF-QNRF_ECCV18'

img_train_path = root + '/Train/'
gt_train_path = root + '/Train/'
img_test_path = root + '/Test/'
gt_test_path = root + '/Test/'

save_train_img_path = root + '/train_data/images/'
save_train_gt_path = root + '/train_data/gt_distance_map/'
save_test_img_path = root + '/test_data/images/'
save_test_gt_path = root + '/test_data/gt_distance_map/'

if not os.path.exists(save_train_img_path):
    os.makedirs(save_train_img_path)
    
if not os.path.exists(save_train_gt_path):
    os.makedirs(save_train_gt_path)
    
if not os.path.exists(save_train_img_path.replace('images','gt_show_distance')):
    os.makedirs(save_train_img_path.replace('images','gt_show_distance'))
    
if not os.path.exists(save_test_img_path):
    os.makedirs(save_test_img_path)
    
if not os.path.exists(save_test_gt_path):
    os.makedirs(save_test_gt_path)
    
if not os.path.exists(save_test_img_path.replace('images','gt_show_distance')):
    os.makedirs(save_test_img_path.replace('images','gt_show_distance'))



def Distance_generate(im_data, gt_data, lamda):
    distance = 1
    new_im_data = im_data
    new_size = im_data.shape

    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda*gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y]-255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 5)

    distance_map[(distance_map >= 0) & (distance_map < 1 * distance)] = 0
    distance_map[(distance_map >= 1 * distance) & (distance_map < 2 * distance)] = 1
    distance_map[(distance_map >= 2 * distance) & (distance_map < 3 * distance)] = 2
    distance_map[(distance_map >= 3 * distance) & (distance_map < 4 * distance)] = 3
    distance_map[(distance_map >= 4 * distance) & (distance_map < 5 * distance)] = 4
    distance_map[(distance_map >= 5 * distance) & (distance_map < 6 * distance)] = 5
    distance_map[(distance_map >= 6 * distance) & (distance_map < 8 * distance)] = 6
    distance_map[(distance_map >= 8 * distance) & (distance_map < 12 * distance)] = 7
    distance_map[(distance_map >= 12 * distance) & (distance_map < 18 * distance)] = 8
    distance_map[(distance_map >= 18 * distance) & (distance_map < 28 * distance)] = 9
    distance_map[(distance_map >= 28 * distance)] = 10

    return new_im_data, distance_map


img_train = []
gt_train = []
img_test = []
gt_test = []

for file_name in os.listdir(img_train_path):
    if file_name.split('.')[1] == 'jpg':
        img_train.append(file_name)

for file_name in os.listdir(gt_train_path):
    if file_name.split('.')[1] == 'mat':
        gt_train.append(file_name)

for file_name in os.listdir(img_test_path):
    if file_name.split('.')[1] == 'jpg':
        img_test.append(file_name)

for file_name in os.listdir(gt_test_path):
    if file_name.split('.')[1] == 'mat':
        gt_test.append(file_name)

img_train.sort()
gt_train.sort()
img_test.sort()
gt_test.sort()

print(len(img_train),len(gt_train), len(img_test),len(gt_test))

start = time.time()
for k in range(len(img_train)):

    Img_data = cv2.imread(img_train_path + img_train[k])
    Gt_data = scipy.io.loadmat(gt_train_path + gt_train[k])
    rate = 1
    rate1 = 1
    rate2 = 1

    flag = 0
    if Img_data.shape[1]>Img_data.shape[0] and Img_data.shape[1] >= 2048:
        rate1 = 2048.0 / Img_data.shape[1]
        flag =1
    if Img_data.shape[0]>Img_data.shape[1] and Img_data.shape[0] >= 2048:
        rate1 = 2048.0 / Img_data.shape[0]
        flag =1
    Img_data = cv2.resize(Img_data,(0,0),fx=rate1,fy=rate1)

    min_shape = 512.0
    if Img_data.shape[1] <= Img_data.shape[0] and Img_data.shape[1] <= min_shape:
        rate2 = min_shape / Img_data.shape[1]
    elif Img_data.shape[0] <= Img_data.shape[1] and Img_data.shape[0] <= min_shape:
        rate2 = min_shape / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0,0), fx=rate2, fy=rate2)

    rate = rate1 * rate2


    Gt_data = Gt_data['annPoints']
    Gt_data = Gt_data * rate
    Gt_data_ori = Gt_data.copy()

    '''gengrate kpoint'''
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1])).astype(np.uint8)
    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    '''generate sigma map'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree

    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=2)
    sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
    for i, pt in enumerate(pts):
        sigma = (distances[i][1]) / 2

        sigma_map[pt[1], pt[0]] = sigma
    '''' end '''

    result = Distance_generate(Img_data, Gt_data, 1)
    Distance_map = result[1].astype(np.uint8)

    new_img_path = (save_train_img_path + img_train[k])

    mat_path = new_img_path.split('.jpg')[0]
    gt_show_distance_path = new_img_path.split('.jpg')[0] + 'gt.jpg'
    h5_path = save_train_gt_path + img_train[k].replace('.jpg','.h5')
    with h5py.File(h5_path, 'w') as hf:
        hf['distance_map'] = Distance_map
        hf['kpoint'] = kpoint
        hf['sigma_map'] =sigma_map


    print(save_train_img_path, len(Gt_data))
    cv2.imwrite(new_img_path, Img_data)
    Distance_map = Distance_map / np.max(Distance_map) * 255

    Distance_map = Distance_map / np.max(Distance_map) * 255
    cv2.imwrite(new_img_path.replace('images','gt_show_distance'), Distance_map)

for k in range(len(img_test)):
    Img_data = cv2.imread(img_test_path + img_test[k])
    Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])
    rate = 1
    rate1 = 1
    rate2 = 1

    flag = 0
    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
        rate1 = 2048.0 / Img_data.shape[1]
        flag = 1
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
        rate1 = 2048.0 / Img_data.shape[0]
        flag = 1
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate1, fy=rate1)

    min_shape = 512.0
    if Img_data.shape[1] <= Img_data.shape[0] and Img_data.shape[1] <= min_shape:
        rate2 = min_shape / Img_data.shape[1]
    elif Img_data.shape[0] <= Img_data.shape[1] and Img_data.shape[0] <= min_shape:
        rate2 = min_shape / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate2, fy=rate2)

    rate = rate1 * rate

    patch_x = Img_data.shape[1]/2
    patch_y = Img_data.shape[0]/2
    Gt_data = Gt_data['annPoints']

    # Gt_data[:, 0] = Gt_data[:, 0] * rate_x
    # Gt_data[:, 1] = Gt_data[:, 1] * rate_y
    Gt_data = Gt_data * rate
    Gt_data_ori = Gt_data.copy()


    '''gengrate kpoint'''
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1])).astype(np.uint8)
    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree

    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=2)
    sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
    for i, pt in enumerate(pts):
        sigma = (distances[i][1]) / 2
        sigma_map[pt[1], pt[0]] = sigma
    '''' end '''

    result = Distance_generate(Img_data, Gt_data, 1)
    Distance_map = result[1].astype(np.uint8)
    new_img_path = (save_test_img_path + img_test[k])


    h5_path = save_test_gt_path + img_test[k].replace('.jpg','.h5')
    with h5py.File(h5_path, 'w') as hf:
        hf['distance_map'] = Distance_map
        hf['kpoint'] = kpoint
        hf['sigma_map'] =sigma_map


    cv2.imwrite(new_img_path, Img_data)
    Distance_map = Distance_map / np.max(Distance_map) * 255
    cv2.imwrite(new_img_path.replace('images','gt_show_distance'), Distance_map)

    print(save_test_img_path, len(Gt_data))