import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import cv2
import os

import dataset
import utils
import ssim_loss

def train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    ssimLoss = ssim_loss.SSIM().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()

    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if epoch % opt.save_by_epoch == 0:
                torch.save(net.module, './model/epoch%d_batchsize%d.pth' % (epoch, opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.save_by_epoch == 0:
                torch.save(net, './model/epoch%d_batchsize%d.pth' % (epoch, opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.UDCDataset(opt)
    #testset = dataset.UDCValidDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    #test_dataloader = DataLoader(testset, batch_size = 1, pin_memory = True)
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        avg_l1_loss = 0
        avg_ssim_loss = 0
        avg_cs_ColorLoss = 0
        avg_l1_loss_lf = 0
        avg_ssim_loss_lf = 0
        generator.train()
        for i, (true_input, true_target) in enumerate(dataloader):

            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input)

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)

            fake_target = fake_target * 0.5 + 0.5
            true_target = true_target * 0.5 + 0.5
            ssim_PixelLoss = 1 - ssimLoss(fake_target, true_target)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + ssim_PixelLoss
            
            avg_l1_loss += Pixellevel_L1_Loss
            avg_ssim_loss += ssim_PixelLoss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [Pixellevel L1 Loss LowFreq: %.4f] [ssim Loss: %.4f] [ssim Loss LowFreq: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(), 0, ssim_PixelLoss.item(), 0, time_left))

            # Save model at certain epochs or iterations
        save_model(generator, (epoch + 1), opt)
        #valid(generator,test_dataloader,(epoch + 1),opt)
        # Learning rate decrease at certain epochs
        adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
        avg_l1_loss = avg_l1_loss / (i + 1)
        avg_ssim_loss = avg_ssim_loss / (i + 1)

        f = open("log.txt", "a")
        f.write('epoch: ' + str(epoch) + ' avg l1 =' + str(avg_l1_loss.item()) + ' avg l1 LowFreq =' + str(0) + ' avg ssim = ' + str(avg_ssim_loss.item()) + ' avg ssim LowFreq = ' + str(0) + '\n')
        f.close()