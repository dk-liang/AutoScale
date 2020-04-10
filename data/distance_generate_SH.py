import math
import os

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial

'''you can uncomment when you generate ShanghaiA target'''
img_train_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_A_final/train_data/images/'
gt_train_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_A_final/train_data/ground_truth/'
img_test_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_A_final/test_data/images/'
gt_test_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_A_final/test_data/ground_truth/'
save_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_A_final/train_data/gt_distance_map/'

'''you can uncomment when you generate ShanghaiB target'''
# img_train_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_B_final/train_data/images/'
# gt_train_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_B_final/train_data/ground_truth/'
# img_test_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_B_final/test_data/images/'
# gt_test_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_B_final/test_data/ground_truth/'
# save_path = '/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_B_final/train_data/gt_distance_map/'


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


for k in range(len(img_train)):
	print(img_train[k], gt_train[k])
	Img_data = cv2.imread(img_train_path + img_train[k])
	Gt_data = scipy.io.loadmat(gt_train_path + gt_train[k])


	Gt_data = Gt_data['image_info'][0][0][0][0][0]
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


	with h5py.File((save_path+img_train[k]).replace('.jpg', '.h5'), 'w') as hf:
		hf['distance_map'] = Distance_map
		hf['kpoint'] = kpoint
		hf['sigma_map'] = sigma_map
	# Distance_map = Distance_map/np.max(Distance_map)*255
	# cv2.imwrite(save_path+img_train[k], Distance_map)


for j in range(len(img_test)):
	print(img_test[j], gt_test[j])
	Img_data = cv2.imread(img_test_path + img_test[j])
	Gt_data = scipy.io.loadmat(gt_test_path + gt_test[j])
	Gt_data = Gt_data['image_info'][0][0][0][0][0]
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

	with h5py.File((save_path+img_test[j]).replace('train_data', 'test_data').replace('.jpg', '.h5'), 'w') as hf:
		hf['distance_map'] = Distance_map
		hf['kpoint'] = kpoint
		hf['sigma_map'] = sigma_map

	# Distance_map = Distance_map/np.max(Distance_map)*255
	# cv2.imwrite((save_path+img_test[j]).replace('train_data', 'test_data'), Distance_map)


print("end")

