import h5py
import torch
import shutil
import numpy as np
import cv2
import os
def save_results(input_img, gt_data,density_map,output_dir, fname='results.png'):

    gt_data = 255*gt_data/np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)


    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    density_map = density_map.astype(np.uint8)


    result_img = np.hstack((gt_data,density_map))
    cv2.imwrite(os.path.join('.',output_dir,fname),result_img)


            
def save_checkpoint(visi, task_id):


    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img,target,output,str(task_id),fname[0])
