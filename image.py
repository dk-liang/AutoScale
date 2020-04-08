import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import scipy

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','gt_distance_map')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['distance_map'])
    kpoint = np.asarray(gt_file['kpoint'])
    sigma_map = np.asarray(gt_file['sigma_map'])

    # mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    # gt = mat["image_info"][0,0][0,0][0]
    # k = np.zeros((img.size[1],img.size[0]))
    # for i in range(0,len(gt)):
    #     if int(gt[i][1])<img.size[1] and int(gt[i][0])<img.size[0]:
    #         k[int(gt[i][1]),int(gt[i][0])]=1
    #
    #
    # pts = np.array(list(zip(np.nonzero(k)[1], np.nonzero(k)[0])))
    # leafsize = 2048
    # # build kdtree
    # tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # # query kdtree
    # distances, locations = tree.query(pts, k=2)
    # sigma_map = np.zeros(k.shape, dtype=np.float32)
    # # pt2d = np.zeros(k.shape,dtype= np.float32)
    # for i, pt in enumerate(pts):
    #     sigma = (distances[i][1]) / 2
    #     sigma_map[pt[1], pt[0]] = sigma


    img=img.copy()
    target=target.copy()
    sigma_map = sigma_map.copy()
    kpoint = kpoint.copy()

    return img, target, kpoint, sigma_map