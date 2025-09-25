import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as uData
import torchvision.utils
from tqdm import tqdm

from math import ceil
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import warnings
from netP.fr_model import FRnet  # Pre-trained quality assessment model
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import mydataset  # Custom dataset module
import numpy as np
import math
from torchvision.transforms.functional import to_tensor
from PIL import Image
from networks import UNetD, UNetG, sample_generator
import cv2
from scipy.stats import entropy
import pandas as pd
from vgg19loss import VGG19PerceptualLoss

# Suppress numpy warnings
np.seterr(all=None, divide='ignore', over=None, under=None, invalid='ignore')

def NonOverlappingCropPatches(img, re, patch_size=32):
    """Split images into non-overlapping patches for perceptually lossy/lossless predictor"""
    batch_size = re.shape[0]
    n = int(img.shape[-1]/patch_size)*int(img.shape[-2]/patch_size)
    a = torch.ones(batch_size, 2, int(n), 1, 32, 32)
    for id in range(batch_size):
        im = img[id]
        im = transforms.ToPILImage(mode='L')(im)
        ref = re[id]
        ref = transforms.ToPILImage(mode='L')(ref)
        w, h = im.size

        patches = ()
        ref_patches = ()
        stride = patch_size
        for i in range(0, h - stride + 1, stride):
            for j in range(0, w - stride + 1, stride):
                patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))-0.5
                patches = patches + (patch,)
                ref_patch = to_tensor(ref.crop((j, i, j + patch_size, i + patch_size)))-0.5
                ref_patches = ref_patches + (ref_patch,)
        img_p = torch.unsqueeze(torch.stack(patches), 0)
        ref_p = torch.unsqueeze(torch.stack(ref_patches),0)
        b = torch.cat((img_p,ref_p), 0)
        a[id] = b
    return a

def L2Loss(model, alpha):
    """Calculate L2 regularization loss"""
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma,2)))
    return l2_loss

# Set device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr_G = 5e-4  # Generator learning rate
batch_size = 32
num_epochs = 300

# Load pre-trained quality assessment model
pre_model = FRnet().to(device)
pre_model.load_state_dict(torch.load('./netP/model_99.pth'))
pre_model.eval()  # Set to evaluation mode

# Define image transformations
transform = transforms.Compose([transforms.ToTensor()])

# Create datasets and data loaders
train_dataset = mydataset.Mydataset(transform, img_path="../JND_train_dataset/sonar_train_320/")
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = mydataset.Mydataset(transform, img_path="../JND_train_dataset/sonar_test_320/")
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

# Define loss functions
criterion1 = nn.BCELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.L1Loss()
criterion4 = VGG19PerceptualLoss()  # Perceptual loss
sigmoid = nn.Sigmoid()

# Initialize generator model and optimizer
netG = UNetG(3, 32, 4).to(device)
optimizer = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.9, 0.99), eps=1e-8)

# Main training loop
for epoch in range(num_epochs):
    # Initialize tracking variables
    e_b_id = []
    t_loss = []
    t_pre = []
    t_ploss = []
    t_psnr = []

    train_loop = tqdm(train_loader, desc='Train')
    for id, data in enumerate(train_loop):
        data = data.to(device)

        # Generate CDI images
        fake, vis = sample_generator(netG, data)
        optimizer.zero_grad()

        # Process real images
        for i in range(int(data.shape[0])):
            data_ = torch.squeeze(data[i], dim=0)
            real_ = data_[0]
            real_ = torch.unsqueeze(real_, dim=0)
            real_ = torch.unsqueeze(real_, dim=0)
            if i == 0:
                real = real_
            else:
                real = torch.cat((real, real_), dim=0)

        # Calculate and normalize difference map
        x = abs(fake - real)
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

        # Save sample images periodically
        if id % 100 == 0:
            torchvision.utils.save_image(fake, 'photo2/train/fake/'+str(epoch)+ '_' + str(id)+'.png')
            torchvision.utils.save_image(real, 'photo2/train/real/'+str(epoch)+ '_' + str(id)+'.png')
            torchvision.utils.save_image(x, 'photo2/train/jnd/'+str(epoch)+ '_' + str(id)+'.png')

        # Initialize metrics
        psnrloss = torch.zeros(int(fake.shape[0]))
        P = torch.zeros(int(fake.shape[0]))  # Quality prediction

        # Calculate metrics per image in batch
        for i in range(int(fake.shape[0])):
            # Calculate PSNR
            msei = criterion2(fake[i], real[i])
            psnri = (20 * math.log10(1 / math.sqrt(msei))) / 100
            psnrloss[i] = psnri

            # Evaluate quality using pre-trained model
            patches = NonOverlappingCropPatches(fake[i], real[i])
            Img = patches.to(device)
            score = pre_model(Img)
            score = sigmoid(score)
            P[i] = 1 if score > 0.5 else 0

        # Aggregate metrics
        ploss = torch.mean(P)
        psnrloss = torch.mean(psnrloss)
        vggloss = criterion4(fake, real)  # Perceptual loss

        # Combined loss function
        loss_all = (1 - ploss) * psnrloss + ploss * (vggloss/3) + ploss + L2Loss(netG, 0.0001)

        # Backpropagation and optimization
        loss_all.backward()
        optimizer.step()

        # Update tracking variables
        e_b_id.append(str(epoch)+'_'+str(id))
        t_loss.append(loss_all.item())
        t_pre.append(pre)
        t_ploss.append(ploss.item())
        t_psnr.append(psnrloss.item())
        
        # Calculate running averages
        tloss_avg = np.mean(t_loss)
        ploss_avg = np.mean(t_ploss)
        psnr_avg = np.mean(t_psnr)

        # Update progress bar
        train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        train_loop.set_postfix(loss=tloss_avg, pre=ploss_avg, psnr=psnr_avg)

    # Save training logs
    dfData = {'epoch_batchsize': e_b_id, 'loss': t_loss, 'pre': t_pre}
    df = pd.DataFrame(dfData)
    df.to_excel('photo2/train/train_log'+str(epoch)+'.xlsx', index=True)

    # Validation phase
    with torch.no_grad():
        test_e_b_id = []
        test_t_loss = []
        test_t_pre = []
        test_ploss = []
        test_psnr = []
        test_loop = tqdm(test_loader, desc='Test')
        
        for id, data in enumerate(test_loop):
            data = data.to(device)

            # Generate CDI images
            fake, vis = sample_generator(netG, data)

            # Process real images
            for i in range(int(data.shape[0])):
                data_ = torch.squeeze(data[i], dim=0)
                real_ = data_[0]
                real_ = torch.unsqueeze(real_, dim=0)
                real_ = torch.unsqueeze(real_, dim=0)
                if i == 0:
                    real = real_
                else:
                    real = torch.cat((real, real_), dim=0)

            # Calculate and normalize difference map
            x = abs(fake - real)
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

            # Save sample images periodically
            if id % 100 == 0:
                torchvision.utils.save_image(fake, 'photo2/test/fake/' + str(epoch) + '_' + str(id) + '.png')
                torchvision.utils.save_image(real, 'photo2/test/real/' + str(epoch) + '_' + str(id) + '.png')
                torchvision.utils.save_image(x, 'photo2/test/jnd/' + str(epoch) + '_' + str(id) + '.png')

            # Initialize metrics
            P = torch.zeros(int(fake.shape[0]))
            psnrloss = torch.zeros(int(fake.shape[0]))

            # Calculate metrics per image
            for i in range(int(fake.shape[0])):
                # Calculate PSNR
                msei = criterion2(fake[i], real[i])
                psnri = (20 * math.log10(1 / math.sqrt(msei))) / 100
                psnrloss[i] = psnri

                # Evaluate quality using pre-trained model
                patches = NonOverlappingCropPatches(fake[i], real[i])
                Img = patches.to(device)
                score = pre_model(Img)
                score = sigmoid(score)
                P[i] = 1 if score > 0.5 else 0

            # Aggregate metrics
            ploss = torch.mean(P)
            psnrloss = torch.mean(psnrloss)
            vggloss = criterion4(fake, real)

            # Combined loss function
            loss_all = (1 - ploss) * psnrloss + ploss * (vggloss/3) + ploss + L2Loss(netG, 0.0001)

            # Update tracking variables
            test_e_b_id.append(str(epoch) + '_' + str(id))
            test_t_loss.append(loss_all.item())
            test_t_pre.append(pre)
            test_ploss.append(ploss.item())
            test_psnr.append(psnrloss.item())
            
            # Calculate running averages
            tloss_avg = np.mean(test_t_loss)
            ploss_avg = np.mean(test_ploss)
            psnr_avg = np.mean(test_psnr)

            # Update progress bar
            test_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            test_loop.set_postfix(loss=tloss_avg, pre=ploss_avg, psnr=psnr_avg)

    # Save validation logs
    test_dfData = {'epoch_batchsize': test_e_b_id, 'loss': test_t_loss, 'pre': test_t_pre}
    df = pd.DataFrame(test_dfData)
    df.to_excel('photo2/test/test_log'+str(epoch)+'.xlsx', index=True)

    # Save model checkpoint
    torch.save(netG.state_dict(), 'models/model_' + str(epoch) + '.pth')