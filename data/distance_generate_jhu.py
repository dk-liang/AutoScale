# coding: utf-8

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
import json
from matplotlib import cm as CM

import torch
import cv2
import math

'''please set your dataset path'''
root = '/home/dkliang/projects/synchronous/jhu_crowd_v2.0'

train = root + '/train/images/'
val = root + '/val/images/'
test = root + '/test/images/'


if not os.path.exists(train.replace('images','images_2048')):
    os.makedirs(train.replace('images','images_2048'))

if not os.path.exists(train.replace('images','gt_distance_map_2048')):
    os.makedirs(train.replace('images','gt_distance_map_2048'))

if not os.path.exists(train.replace('images','gt_show_distance')):
    os.makedirs(train.replace('images','gt_show_distance'))

if not os.path.exists(val.replace('images','images_2048')):
    os.makedirs(val.replace('images','images_2048'))

if not os.path.exists(val.replace('images','gt_distance_map_2048')):
    os.makedirs(val.replace('images','gt_distance_map_2048'))

if not os.path.exists(test.replace('images','images_2048')):
    os.makedirs(test.replace('images','images_2048'))

if not os.path.exists(test.replace('images','gt_distance_map_2048')):
    os.makedirs(test.replace('images','gt_distance_map_2048'))





path_sets = [train, test, val]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()
count = 0


def Distance_generate(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)
    distance = 1
    new_size = new_im_data.shape
    # print(new_size[0], new_size[1])
    # d_map = np.zeros((new_size[0],new_size[0][1])).astype(np.uint8)
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
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

    count = count + 1
    img = cv2.imread(img_path)
    gt_file = np.loadtxt(img_path.replace('images', 'gt').replace('jpg', 'txt'))

    rate1 = 1
    rate2 = 1
    rate = 1

    if img.shape[1] >= img.shape[0] and img.shape[1] >= 2048:
        rate1 = 2048.0 / img.shape[1]
    elif img.shape[0] >= img.shape[1] and img.shape[0] >= 2048:
        rate1 = 2048.0 / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1)

    min_shape = 512.0
    if img.shape[1] <= img.shape[0] and img.shape[1] <= min_shape:
        rate2 = min_shape / img.shape[1]
    elif img.shape[0] <= img.shape[1] and img.shape[0] <= min_shape:
        rate2 = min_shape / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate2, fy=rate2)

    rate = rate1 * rate2



    k = np.zeros((img.shape[0], img.shape[1]))

    fname = img_path.split('/')[6]
    if count % 5 == 0:
        print(count, img_path, img.shape)

    try:
        y = gt_file[:, 0] * rate
        x = gt_file[:, 1] * rate

        for i in range(0, len(x)):
            if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
                # print(gt[i][1],gt[i][0])
                k[int(x[i]), int(y[i])] = 1

        Gt_data = gt_file * rate
        result = Distance_generate(img, Gt_data, 1)
        new_img = result[0]
        Distance_map = result[1]

    except Exception:
        try:
            y = gt_file[0] * rate
            x = gt_file[1] * rate
            for i in range(0, 1):
                if int(x) < img.shape[0] and int(y) < img.shape[1]:
                    # print(gt[i][1],gt[i][0])
                    k[int(x), int(y)] = 1

            result = Distance_generate(img, [[y, x]], 1)
            new_img = result[0]
            Distance_map = result[1]

        except Exception:
            print("the image has zero person")

            Gt_data = gt_file * rate
            result = Distance_generate(img, Gt_data, 1)
            new_img = result[0]
            Distance_map = result[1]

    kpoint = k.copy()

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

    with h5py.File(img_path.replace('images', 'gt_distance_map_2048').replace('jpg', 'h5'), 'w') as hf:
        hf['distance_map'] = Distance_map
        hf['kpoint'] = kpoint
        hf['sigma_map'] = sigma_map

    cv2.imwrite(img_path.replace('images', 'images_2048'), img)

    density_map = Distance_map
    density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)

    gt_show = img_path.replace('images', 'gt_show_distance')
    cv2.imwrite(gt_show, density_map)

print("end")
