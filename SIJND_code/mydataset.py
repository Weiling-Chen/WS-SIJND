import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms

class Mydataset(data.Dataset):

    def __init__(self, transform, img_path='../JND_train_dataset/sonar_train_320/', ic_path='../JND_train_dataset/IC/', sa_path='../JND_train_dataset/IC/'):
        super(Mydataset, self).__init__()

        self.img_path = img_path    # original image path
        self.ic_path = ic_path      # image complexity heatmap path
        self.sa_path = sa_path      # edge map path
        self.Data = []              # store data
        self.transform = transform  # data preprocessing transform
        self.read_data()

    def read_data(self):
        # read image file name list
        img_name = os.listdir(self.img_path)
        for i in img_name[::]:
            # read original image and convert to grayscale
            path = self.img_path + i
            img_data = np.array(Image.open(path).convert('L'))
            img_data = self.transform(img_data)
            
            # read IC image and convert to grayscale
            ic_data = np.array(Image.open(self.ic_path + i[:-4:]+'.png').convert('L'))
            ic_data = self.transform(ic_data)
            
            # read SA image and convert to grayscale
            sa_data = np.array(Image.open(self.sa_path + i[:-4:] + '.bmp').convert('L'))
            sa_data = self.transform(sa_data)
            
            # concatenate three channels of data
            t_data = torch.cat((img_data,ic_data,sa_data),dim=0)
            self.Data.append(t_data)

    def __getitem__(self, index):
        # get data at specified index
        img = self.Data
        return img[index]

    def __len__(self):
        # return dataset length
        return len(self.Data)


if __name__ == '__main__':
    # define data preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor()
         ]
    )
    # create dataset instance
    data = Mydataset(transform)
    print(data)