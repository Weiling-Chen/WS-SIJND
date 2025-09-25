import torch.nn as nn
import torch
# from c_patches import RandomCropPatches
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor


class FRnet(nn.Module):
    """
    (Wa)DIQaM-NR Model
    """

    def __init__(self):
        super(FRnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)

        self.fc1 = nn.Linear(512 * 3, 512)
        self.fc2 = nn.Linear(512, 1)

        self.fc3 = nn.Linear(512 * 3, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        # self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.flatten = nn.Flatten(1)

    def extract_feature(self, x):
        """
        Feature extraction
        :param x: Input image tensor
        :return: Output feature tensor
        """
        # x shape: (32, 3, 32, 32)
        h = self.relu(self.conv1(x))  # (32, 32, 32, 32)
        h = self.relu(self.conv2(h))  # (32, 32, 32, 32)
        h = self.maxpool(h)  # (32, 32, 16, 16)

        h = self.relu(self.conv3(h))  # (32, 64, 16, 16)
        h = self.relu(self.conv4(h))  # (32, 64, 16, 16)
        h = self.maxpool(h)  # (32, 64, 8, 8)

        h = self.relu(self.conv5(h))  # (32, 128, 8, 8)
        h = self.relu(self.conv6(h))  # (32, 128, 8, 8)
        h = self.maxpool(h)  # (32, 128, 4, 4)
        vs_f = h  # Visual features

        h = self.relu(self.conv7(h))  # (32, 256, 4, 4)
        h = self.relu(self.conv8(h))  # (32, 256, 4, 4)
        h = self.maxpool(h)  # (32, 256, 2, 2)

        h = self.relu(self.conv9(h))  # (32, 512, 2, 2)
        h = self.relu(self.conv10(h))  # (32, 512, 2, 2)
        h = self.maxpool(h)  # (32, 512, 1, 1)

        return h

    def forward(self, data):
        """
        Forward pass
        :param data: Distorted and reference image patches
        :return: Predicted quality scores
        """
        # data shape: (B, 2, 32, 1, 32, 32)
        data = data.transpose(1, 0)
        # Transposed data shape: (2, B, 32, 1, 32, 32)

        # Handle different input dimensions
        if type(data.size) is int or data.dim() == 0:
            batch_size = 1
            x, x_ref = data  # (B, 32, 1, 32, 32)
            q = torch.ones(batch_size)
        else:
            batch_size = data.size(1)  # Number of images in batch
            x, x_ref = data  # (B, 32, 1, 32, 32)
            q = torch.ones((batch_size, 1), device=x.device)  # Quality tensor

        for i in range(batch_size):
            # Extract features for distorted and reference images
            h = self.extract_feature(x[i])  # (32, 512, 1, 1)
            h_ref = self.extract_feature(x_ref[i])  # (32, 512, 1, 1)

            # Feature fusion: (P, 3C, H, W) -> (32, 1536, 1, 1)
            h = torch.cat((h - h_ref, h, h_ref), 1)
            h = self.flatten(h)  # Flatten to (32, 1536)

            # Save fused features for weight calculation
            h_ = h

            # Quality regression branch
            h_reg = self.fc1(h)  # (32, 1536) -> (32, 512)
            h_reg = self.relu(h_reg)
            h_reg = self.dropout(h_reg)
            h_reg = self.fc2(h_reg)  # (32, 512) -> (32, 1)

            # Weight calculation branch
            w = self.fc3(h_)  # (32, 1536) -> (32, 512)
            w = self.relu(w)
            w = self.dropout(w)
            w = self.fc4(w)  # (32, 512) -> (32, 1)
            w = self.relu(w) + 1e-6  # Ensure non-zero weights

            # Compute weighted quality score
            q_img = torch.sum(h_reg * w) / torch.sum(w)
            q[i] = q_img

        return q


def loss_func(scores, labels):
    """Calculate Binary Cross Entropy loss"""
    bce_c = nn.BCEWithLogitsLoss()
    loss = bce_c(scores, labels)
    return loss


if __name__ == "__main__":
    # Test with random data
    data = np.random.rand(1, 2, 320, 320, 3)
    model = FRnet()
    y = model(data)
    print(y)