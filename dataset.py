import os
import numpy as np
import cv2
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils



class UDCDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.in_root = opt.in_root
        self.out_root = opt.out_root
        self.file_list = utils.get_files(self.in_root)
        

    def img_aug(self, input_img, gt_img):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            if input_img.shape[0] != input_img.shape[1]:
                rotate = random.choice([0, 2])
            else:
                rotate = random.randint(0, 3)

            if rotate != 0:
                input_img = np.rot90(input_img, rotate)
                gt_img = np.rot90(gt_img, rotate)

            # horizontal flip
            if np.random.random() >= 0.5:
                input_img = cv2.flip(input_img, flipCode = 1)
                gt_img = cv2.flip(gt_img, flipCode = 1)
                
        return input_img, gt_img

    def __getitem__(self, index):

        # Define path
        in_path = os.path.join(self.in_root, self.file_list[index])
        out_path = os.path.join(self.out_root, self.file_list[index])

        
        # Read images
        # input
        input_img = Image.open(in_path)
        width, height = input_img.size
        input_img = input_img.resize((width//2,height//2),Image.BICUBIC)
        input_img = np.array(input_img).astype(np.float64)
        
        # output
        gt_img = Image.open(out_path)
        gt_img = np.array(gt_img).astype(np.float64)

        
        input_img, gt_img = self.img_aug(input_img, gt_img)
        
        #normalize
        input_img = (input_img - 128) / 128
        gt_img = (gt_img - 128) / 128

        input_img = torch.from_numpy(input_img.transpose(2, 0, 1).astype(np.float32)).contiguous()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1).astype(np.float32)).contiguous()
        

        return input_img, gt_img
    
    def __len__(self):
        return len(self.file_list)


class UDCValidDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_files(opt.valid_root)
        self.in_root = opt.valid_root
        

    def __getitem__(self, index):
        # Define path
        input_img_name = str(index) + '.png'                                   # png: input RGBA
        in_path = os.path.join(self.in_root, input_img_name)
        
        # Read images
        # input
        input_img = Image.open(in_path)
        width, height = input_img.size
        input_img = input_img.resize((width//2,height//2),Image.BICUBIC)
        input_img = np.array(input_img).astype(np.float64)
        
        input_img = (input_img - 128) / 128
        
        input_img = torch.from_numpy(input_img.transpose(2, 0, 1).astype(np.float32)).contiguous()
        
        return input_img

    def __len__(self):
        return len(utils.get_files(self.in_root))