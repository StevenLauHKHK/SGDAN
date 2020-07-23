import os
import numpy as np
import glob
from PIL import Image
from torchvision.transforms import *
import random
import cv2
import utils
import guided_filter

def load_img(filepath, inYCbCr=False):
    if inYCbCr:
        img = Image.open(filepath).convert('YCbCr')
    else:
        img = Image.open(filepath).convert('RGB')
    return img


gt_folder = '../ECCV_UDC/Poled/HQ'
gt_lf_folder = '../ECCV_UDC/Poled/HQ_LowFreq'
input_folder = '../ECCV_UDC/Poled/LQ'

output_gt_folder = '../ECCV_UDC/Poled/HQ_Patch_1024'
output_gt_lf_folder = '../ECCV_UDC/Poled/HQ_LowFreq_Patch_1024'
output_input_folder = '../ECCV_UDC/Poled/LQ_Patch_1024'

output_LQ_base_folder = '../ECCV_UDC/Poled/LQ_base'
output_LQ_detail_folder = '../ECCV_UDC/Poled/LQ_detail'


patch_size = 512
stride = 256

file_list = utils.get_files(input_folder)
img_num = 0

for image_file in sorted(file_list):

    print(image_file)
    input_img_file = os.path.join(input_folder,image_file)
    gt_img_file = os.path.join(gt_folder,image_file)

    input_img = load_img(input_img_file)
    gt_img = load_img(gt_img_file)
    
    input_img = np.array(input_img)
    gt_img = np.array(gt_img)
    
    for i in range(0, input_img.shape[0] - patch_size - 2, stride):
        for j in range(0, input_img.shape[1] - patch_size - 2, stride):

            input_img_patch = input_img[i:i+patch_size,j:j+patch_size,:]
            
            input_img_patch = Image.fromarray(input_img_patch,mode='RGB')
            input_img_patch.save(os.path.join(output_input_folder,str(img_num) + '.png'))

            gt_img_patch = gt_img[i:i+patch_size,j:j+patch_size,:]
            gt_img_patch = Image.fromarray(gt_img_patch,mode='RGB')
            gt_img_patch.save(os.path.join(output_gt_folder,str(img_num) + '.png'))

            img_num += 1
