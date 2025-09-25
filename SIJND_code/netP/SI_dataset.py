import numpy as np
import torch.utils.data as data
from PIL import Image
import pandas as pd
import os

import c_patches


# Target shape: (B, 2, 320, 320, 3)

class Mydataset(data.Dataset):

    def __init__(self, excel_file="Dataset/JND.xlsx", img_path="Dataset/image/", mode='train', transform=None):
        super(Mydataset, self).__init__()

        self.excel_file = excel_file  # JND table path
        self.img_path = img_path      # Image path
        self.mode = mode              # Mode: 'train' or others
        self.Data = []                # Store processed data
        self.transform = transform    # Data augmentation
        self.read_data()              # Read and process data

    def read_data(self):
        Data_ref = []  # Store reference image patches
        Data_img = []  # Store distorted image patches
        Labels = []    # Store labels
        
        # Read excel file containing image names and JND values
        data = pd.read_excel(self.excel_file)  
        ref_name = data["img_name"]  # Get image name column
        jnd_num = data["JND"]        # Get JND value column
        JND = dict(zip(ref_name, jnd_num))  # Create dictionary mapping image name to JND value

        # Set paths for distorted and reference images
        img_path = self.img_path + 'distortion_img/'
        ref_path = self.img_path + 'reference_img/'

        length = len(ref_name)

        # Process each reference image
        for i in range(length):
            ref = ref_name[i][:2:]  # Get first 2 characters as base name
            path_ref = ref_path + ref_name[i]  # Build reference image path
            ref_data = Image.open(path_ref).convert('L')  # Open as grayscale
            ref_data = np.array(ref_data)
            
            # Generate image pairs for each QP value (1-51)
            for j in range(1, 52):
                qp = j
                path_img = img_path + ref + '_' + str(j) + '.png'  # Build distorted image path
                img_data = Image.open(path_img).convert('L')  # Open as grayscale
                img_data = np.array(img_data)
                
                # Choose different patch cropping methods based on mode
                if self.mode == 'train':
                    img_patches, ref_patches = c_patches.RandomCropPatches1(img_data, ref_data)  # Random crop for training
                else:
                    img_patches, ref_patches = c_patches.NonOverlappingCropPatches(img_data, ref_data)  # Non-overlapping crop for testing

                img_patches = np.array(img_patches)
                ref_patches = np.array(ref_patches)
                Data_img.append(img_patches)
                Data_ref.append(ref_patches)

                jnd1 = JND[ref_name[i]]  # Get JND threshold for current image

                # Set label based on QP value vs JND threshold
                if qp < jnd1:
                    label = 0  # Label 0 if QP < JND threshold
                else:
                    label = 1  # Label 1 if QP >= JND threshold
                Labels.append(label)

        # Convert to numpy arrays
        Labels = np.array(Labels)
        Data_img = np.array(Data_img)
        Data_ref = np.array(Data_ref)
        
        # Expand dimensions for concatenation
        Data_img = np.expand_dims(Data_img, axis=0)  # (1, N, H, W, C)
        Data_ref = np.expand_dims(Data_ref, axis=0)  # (1, N, H, W, C)
        
        # Concatenate distorted and reference images along axis 0
        Img = np.concatenate((Data_img, Data_ref), axis=0)  # (2, N, H, W, C)
        
        # Transpose to get shape (N, 2, H, W, C)
        Img = Img.transpose(1, 0, 2, 3, 4, 5)
        
        # Store processed data
        self.Data = [Img, Labels]

    def __getitem__(self, index):
        img, label = self.Data
        # img = self.transform(img)  # Optional data augmentation
        return img[index], label[index]  # Return image and label at specified index

    def __len__(self):
        return self.Data[0].shape[0]  # Return dataset length


if __name__ == '__main__':
    # Test code: create dataset instance
    data = Mydataset(excel_file='Dataset/data1/JND_train1.xlsx')
    print(data)