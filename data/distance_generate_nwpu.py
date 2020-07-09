# coding: utf-8

# In[1]:


import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import scipy
import math
import time
import json
from matplotlib import cm as CM

import torch
import cv2

# get_ipython().magic(u'matplotlib inline')

'''please set your dataset path'''
NWPU_Crowd_path = '/home/dkl/projects/synchronous/NWPU_localization/images_2048/'

path_sets = [NWPU_Crowd_path]

if not os.path.exists(NWPU_Crowd_path.replace('images','gt_distance_map')):
    os.makedirs(NWPU_Crowd_path.replace('images','gt_distance_map'))




img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(img_paths[0])
img_paths.sort()


def Distance_generate(im_data, k, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)
    distance = 1
    new_size = new_im_data.shape
    # print(new_size[0], new_size[1])
    # d_map = np.zeros((new_size[0],new_size[0][1])).astype(np.uint8)
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = np.nonzero(k)
    gt = lamda * gt


    if len(gt[0]) == 0:
        distance_map = np.zeros([int(new_size[0]), int(new_size[1])])
        distance_map[:, :] = 10
        return new_size, distance_map

    for o in range(0, len(gt[0])):
        x = np.max([1, math.floor(gt[0][o])])
        y = np.max([1, math.floor(gt[1][o])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

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

for img_path in img_paths:
    # if img_path!='/data/weixu/NWPU_Crowd/train_data/images/2671.jpg':
    #     continue

    img = cv2.imread(img_path)

    k = np.zeros((img.shape[0], img.shape[1]))
    mat_path = img_path.replace('images', 'gt_npydata').replace('jpg', 'npy')

    with open(mat_path, 'rb') as outfile:
        gt = np.load(outfile).tolist()

    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            # print(gt[i][1],gt[i][0])
            k[int(gt[i][1]), int(gt[i][0])] = 1

    kpoint = k.copy()

    print(img_path, k.sum(), kpoint.shape)

    result = Distance_generate(img, k, 1)
    new_img = result[0]
    Distance_map = result[1]

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree

    if int(kpoint.sum()) > 1:
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=2)
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            sigma = (distances[i][1]) / 2
            sigma_map[pt[1], pt[0]] = sigma
    elif int(kpoint.sum()) == 1:
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=1)
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            sigma = (distances[i]) / 1
            sigma_map[pt[1], pt[0]] = sigma
    else:
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)

    with h5py.File(img_path.replace('images', 'gt_distance_map').replace('jpg', 'h5'), 'w') as hf:
        hf['distance_map'] = Distance_map
        hf['kpoint'] = kpoint
        hf['sigma_map'] = sigma_map

    # density_map = Distance_map
    # density_map = density_map / np.max(density_map) * 255
    # density_map = density_map.astype(np.uint8)
    #
    #
    # gt_show = img_path.replace('images', 'gt_show')
    # cv2.imwrite(gt_show, density_map)

print("end")
