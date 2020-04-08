import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
from torchvision import transforms
import  time
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4,drop_last=False):
        if train:
            # root =4*root
            #random.shuffle(root)
            self.batch_size = batch_size
        else :
            self.batch_size = 1

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.drop_last = drop_last
        self.num_workers = num_workers



    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        #print(index)



        img_path = self.lines[index]

        fname = os.path.basename(img_path)
        img,target,kpoint,sigma_map= load_data(img_path,self.train)


        # img =self.lines[index]['img']
        # target =self.lines[index]['gt']
        # fname =self.lines[index]['fname']
        # sigma_map = self.lines[index]['sigma']
        # k = self.lines[index]['kpoint']

        # loader = transforms.ToTensor()
        # original_img = loader(img.copy())


        if self.transform is not None:
            img = self.transform(img)

        return fname, img, target, kpoint, sigma_map
