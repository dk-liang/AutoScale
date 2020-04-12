import os
import time

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.misc
from scipy.ndimage.filters import gaussian_filter

root = './UCF-QNRF'
img_train_path = root + '/Train/'
gt_train_path = root + '/Train/'
img_test_path = root + '/Test/'
gt_test_path = root + '/Test/'

save_train_img_path = root + '/train_data_1920/images/'
save_train_gt_path = root + '/train_data_1920/gt_distance_map/'
save_test_img_path = root + '/test_data_1920/images/'
save_test_gt_path = root + '/test_data_1920/gt_distance_map/'

distance = 1


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
# print(img_train)
# print(gt_train)
print(len(img_train),len(gt_train), len(img_test),len(gt_test))


min_x = 640
min_y = 480
x = []
y = []
count_min_x = 0
count_min_y = 0
start = time.time()



for k in range(len(img_train)):

    Img_data = cv2.imread(img_train_path + img_train[k])
    Gt_data = scipy.io.loadmat(gt_train_path + gt_train[k])
    rate = 1
    flag = 0
    if Img_data.shape[1]>Img_data.shape[0] and Img_data.shape[1] >= 1920:
        rate = 1920.0 / Img_data.shape[1]
        flag =1
    if Img_data.shape[0]>Img_data.shape[1] and Img_data.shape[0] >= 1920:
        rate = 1920.0 / Img_data.shape[0]
        flag =1
    Img_data = cv2.resize(Img_data,(0,0),fx=rate,fy=rate)

    if Img_data.shape[0]<min_y:
        min_y = Img_data.shape[0]
        count_min_y = count_min_y+1
        print(min_x, min_y, img_train[k])

    if Img_data.shape[1]<min_x:
        min_x = Img_data.shape[1]
        count_min_x =count_min_x +1
        print(min_x,min_y, img_train[k])

    # if Img_data.shape[1]<300 or Img_data.shape[0]<300 :
    #     print (img_train[k])
    #     continue

    x.append(Img_data.shape[1])
    y.append(Img_data.shape[0])
    #print(img_train[k], min_y, min_x, rate, Img_data.shape)
    patch_x = Img_data.shape[1]/2
    patch_y = Img_data.shape[0]/2
    Gt_data = Gt_data['annPoints']
    # Gt_data[:, 0] = Gt_data[:, 0] * rate_x
    # Gt_data[:, 1] = Gt_data[:, 1] * rate_y
    Gt_data = Gt_data * rate
    Gt_data_ori = Gt_data.copy()


    density_map = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            density_map[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
    density_map = gaussian_filter(density_map, 15)

    #print(density_map.shape, Img_data.shape, img_train[k])
    new_img_path = (save_train_img_path + img_train[k])

    mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.split('.jpg')[0] + 'gt.jpg'
    h5_path = new_img_path.split('.jpg')[0] + '.h5'
    with h5py.File(h5_path, 'w') as hf:
        hf['density'] = density_map

    print(rate, gt_show_path, new_img_path)
    np.save(mat_path, Gt_data)
    cv2.imwrite(new_img_path, Img_data)
    density_map = density_map / np.max(density_map) * 255

    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,2)

    cv2.imwrite(gt_show_path, density_map)


x.sort()
y.sort()
print(x)
print(y)


for k in range(len(img_test)):
    Img_data = cv2.imread(img_test_path + img_test[k])
    Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])



    rate = 1
    flag = 0
    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 1920:
        rate = 1920.0 / Img_data.shape[1]
        flag = 1
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 1920:
        rate = 1920.0 / Img_data.shape[0]
        flag = 1
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate, fy=rate)

    if Img_data.shape[0]<min_y:
        min_y = Img_data.shape[0]
        #print(img_test[k])

    if Img_data.shape[1]<min_x:
        min_x = Img_data.shape[1]
    #print(min_y,min_x,rate)

    patch_x = Img_data.shape[1]/2
    patch_y = Img_data.shape[0]/2
    Gt_data = Gt_data['annPoints']

    # Gt_data[:, 0] = Gt_data[:, 0] * rate_x
    # Gt_data[:, 1] = Gt_data[:, 1] * rate_y
    Gt_data = Gt_data * rate
    Gt_data_ori = Gt_data.copy()


    density_map = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            density_map[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
    density_map = gaussian_filter(density_map, 15)


    new_img_path = (save_test_path + img_test[k])
    mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.split('.jpg')[0] + '_gt.jpg'
    h5_path = new_img_path.split('.jpg')[0] + '.h5'
    with h5py.File(h5_path, 'w') as hf:
        hf['density'] = density_map

    np.save(mat_path, Gt_data)
    cv2.imwrite(new_img_path, Img_data)
    density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,2)
    cv2.imwrite(gt_show_path, density_map)




# #
# #
# print("end")