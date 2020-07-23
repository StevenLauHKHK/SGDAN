import numpy as np
import os
from scipy.io.matlab.mio import savemat, loadmat
import cv2
from torchvision import transforms
import torch
import torch.nn as nn
import network
import argparse
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
import time
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def restoration(opt, model, udc, batch_idx, run_time):
    # TODO: plug in your method here

    udc = Image.fromarray(udc, 'RGB')
    width, height = udc.size
    udc = udc.resize((width//2,height//2),Image.BICUBIC)

    with torch.no_grad():
        udc = np.array(udc).astype(np.float64)
        udc = (udc - 128) / 128
        udc = torch.from_numpy(udc.transpose(2, 0, 1).astype(np.float32)).contiguous()
        udc = udc.unsqueeze(0)
        udc = udc.cuda()
        t = time.time()
        udc = model(udc)
        run_time = run_time + time.time() - t

        _,_,height,width = udc.shape

        udc = udc.detach().cpu().numpy().reshape(3, height, width).transpose(1, 2, 0)
        udc = (udc * 0.5 + 0.5) * 255.0
        udc = udc.astype(np.uint8)

        r, g, b = cv2.split(udc)
        show_img = cv2.merge([b, g, r])
        valid_img_save_path = os.path.join(opt.saving_img_root, '%d.png' % (batch_idx))
        cv2.imwrite(valid_img_save_path, show_img)

    return udc, run_time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_file', type = str, default = "../ECCV_UDC/poled_test_display.mat", help = 'the testing mat file')
    parser.add_argument('--saving_img_root', type = str, default = "./test_final", help = 'saving the png img root')
    parser.add_argument('--saving_mat_dir', type = str, default = "./", help = 'saving the mat root')
    parser.add_argument('--load_name', type = str, default = './pretrain_model/epoch74_batchsize2.pth', help = 'test model name')
    opt = parser.parse_args()
    print(opt)

    work_dir = opt.saving_mat_dir
    generator = torch.load(opt.load_name)
    generator = generator.cuda()
    generator.eval()

    # load noisy images
    udc_fn = opt.valid_file
    udc_key = 'test_display'
    udc_mat = loadmat(os.path.join(udc_fn))[udc_key]

    run_time = 0
    # restoration
    n_im, h, w, c = udc_mat.shape
    results = udc_mat.copy()
    for i in range(n_im):
        udc = np.reshape(udc_mat[i, :, :, :], (h, w, c))
        restored, run_time = restoration(opt,generator,udc, i, run_time)
        results[i, :, :, :] = restored
    print(run_time)

    # create results directory
    res_dir = 'res_dir'
    os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

    # save images in a .mat file with dictionary key "results"
    res_fn = os.path.join(work_dir, res_dir, 'results.mat')
    res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
    savemat(res_fn, {res_key: results})

    # submission indormation
    # TODO: update the values below; the evaluation code will parse them
    runtime = run_time/30  # seconds / image
    cpu_or_gpu = 0  # 0: GPU, 1: CPU
    method = 1  # 0: traditional methods, 1: deep learning method
    other = '(optional) any additional description or information'

    # prepare and save readme file
    readme_fn = os.path.join(work_dir, res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
    with open(readme_fn, 'w') as readme_file:
        readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
        readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
        readme_file.write('Method: %s\n' % str(method))
        readme_file.write('Other description: %s\n' % str(other))

    # compress results directory
    res_zip_fn = 'results_dir'
    shutil.make_archive(os.path.join(work_dir, res_zip_fn), 'zip', os.path.join(work_dir, res_dir))
