import glob
import math
import os

import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial

'''please set your dataset path'''
root = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/'

part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')

path_sets = [part_A_train,part_A_test,part_B_train,part_B_test]

if not os.path.exists(part_A_train.replace('images','gt_distance_map') ):
	os.makedirs(part_A_train.replace('images','gt_distance_map'))

if not os.path.exists(part_A_test.replace('images','gt_distance_map')):
	os.makedirs(part_A_test.replace('images','gt_distance_map'))

if not os.path.exists(part_A_train.replace('images','gt_show_distance')):
	os.makedirs(part_A_train.replace('images','gt_show_distance'))

if not os.path.exists(part_A_test.replace('images','gt_show_distance')):
	os.makedirs(part_A_test.replace('images','gt_show_distance'))

if not os.path.exists(part_B_train.replace('images','gt_distance_map')):
	os.makedirs(part_B_train.replace('images','gt_distance_map'))

if not os.path.exists(part_B_test.replace('images','gt_distance_map')):
	os.makedirs(part_B_test.replace('images','gt_distance_map'))

if not os.path.exists(part_B_train.replace('images','gt_show_distance')):
	os.makedirs(part_B_train.replace('images','gt_show_distance'))

if not os.path.exists(part_B_test.replace('images','gt_show_distance')):
	os.makedirs(part_B_test.replace('images','gt_show_distance'))




img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

def Distance_generate(im_data, gt_data, lamda):
	size = im_data.shape
	new_im_data = cv2.resize(im_data, (lamda*size[1], lamda*size[0]), 0)
	distance = 1
	new_size = new_im_data.shape
	#print(new_size[0], new_size[1])
	# d_map = np.zeros((new_size[0],new_size[0][1])).astype(np.uint8)
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


for img_path in img_paths:

	Img_data = cv2.imread(img_path)

	mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
	Gt_data = mat["image_info"][0][0][0][0][0]

	result = Distance_generate(Img_data, Gt_data, 1)
	new_img = result[0]
	Distance_map = result[1]

	kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
	for i in range(0, len(Gt_data)):
		if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
			kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

	pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
	leafsize = 2048
	# build kdtree
	tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
	# query kdtree
	distances, locations = tree.query(pts, k=2)
	sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
	# pt2d = np.zeros(k.shape,dtype= np.float32)
	for i, pt in enumerate(pts):
		sigma = (distances[i][1]) / 2
		sigma_map[pt[1], pt[0]] = sigma

	with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt_distance_map'), 'w') as hf:
		hf['distance_map'] = Distance_map
		hf['kpoint'] = kpoint
		hf['sigma_map'] = sigma_map


	Distance_map = Distance_map/np.max(Distance_map)*255
	cv2.imwrite(img_path.replace('images','gt_show_distance').replace('.jpg','.bmp'), Distance_map)

	print(img_path)
