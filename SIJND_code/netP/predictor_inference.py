import numpy as np
import c_patches
from fr_model import FRnet
import torch
from torch.utils.data import Dataset,DataLoader,random_split

from torch import nn,optim
import torchvision.transforms as transforms
from tqdm import tqdm
import copy
import pandas as pd
from PIL import Image
import SI_dataset
from torchvision.transforms.functional import to_tensor


def NonOverlappingCropPatches(im, ref, patch_size=32):
    """
    Function: Crop input image and reference image into non-overlapping patches
    Parameters:
    - im: distorted image to be processed
    - ref: reference image
    - patch_size: patch size, default is 32
    Returns:
    - torch stacked image patches and reference image patches
    """
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
    return torch.stack(patches), torch.stack(ref_patches)


def pre(read_path, des_path, model_path):
    """
    Function: Batch prediction of image perceptual quality and calculate accuracy
    Parameters:
    - read_path: Input Excel file path containing image names and JND values
    - des_path: Output Excel file save path
    - model_path: Model weight file path
    Functionality:
    - Read reference and distorted images
    - Use FRnet model for quality prediction
    - Calculate prediction accuracy and compare with true labels
    """
    m = nn.Sigmoid()
    test_correct = 0
    test_total = 408

    path = 'Dataset/image/'
    batch_size = 1
    img_name = []
    pre = []
    model = FRnet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = pd.read_excel(read_path)  # read excel file
    ref_name = data["img_name"]
    jnd_num = data['JND']
    dc = dict(zip(ref_name, jnd_num))

    for i in range(len(ref_name)):
        jnd = dc.get(ref_name[i])

        ref_path = path + 'reference_img/' + ref_name[i]
        ref_data = Image.open(ref_path).convert('L')
        for j in range(1, 52):
            if j < int(jnd):
                tag = 0
            else:
                tag = 1
            img_path = path + 'distortion_img/' + ref_name[i][:2:] + '_' + str(j) + '.png'
            img_data = Image.open(img_path).convert('L')
            img_patch, ref_patch = NonOverlappingCropPatches(img_data, ref_data)
            Img = [[img_patch, ref_patch]]
            Img = np.array(Img)
            score = model(Img)
            y = m(score)
            if y > 0.5:
                label = 1
            else:
                label = 0

            if tag == label:
                test_correct+=1

            if len(str(j)) == 1:
                qp = '0' + str(j)
            else:
                qp = str(j)
            img_name.append(ref_name[i][:2:] + '_' + qp + '.png')
            pre.append(label)
            print(ref_name[i][:2:] + '_' + qp + '.png', label)

    acc = test_correct/test_total
    print('acc: ', acc*100, '%')
    dfData = {
        'img_name': img_name,
        'pre_label': pre,
        'acc':acc
    }
    df = pd.DataFrame(dfData)
    df.to_excel(des_path, index=False)


def pre_single(ref_path, img_path, model_path):
    """
    Function: Single image quality prediction
    Parameters:
    - ref_path: Reference image path
    - img_path: Image path to be evaluated
    - model_path: Model weight file path
    Returns:
    - Predicted label (0 or 1)
    """
    m = nn.Sigmoid()
    model = FRnet()
    model.load_state_dict(torch.load(model_path))
    # torch.save(model, 'Kfold/model_3_99_pth.pth')
    model.eval()

    ref_data = Image.open(ref_path).convert('L')
    img_data = Image.open(img_path).convert('L')
    img_patch, ref_patch = NonOverlappingCropPatches(img_data, ref_data)
    Img = torch.stack([img_patch, ref_patch], 0)
    Img = Img.unsqueeze(0)
    score = model(Img)
    y = m(score)
    if y > 0.5:
        label = 1
    else:
        label = 0
    print(img_path,label)
    return label

# Batch process images in specified directory, calculate perceptual quality and count number of labels with value 0
import os
img_path= "your/reference_image/path/"
name = list(os.listdir(img_path))
count = 0
for i in name:
    ref_path = "your/reference_image/path/" + i
    dis_path = "your/distorted_image/path/" + i
    model_path = './model_99.pth'
    label = pre_single(ref_path, dis_path, model_path)
    if label == 0:
        count+=1
print('count:', count)