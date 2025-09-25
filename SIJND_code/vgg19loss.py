import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGG19PerceptualLoss(nn.Module):
    def __init__(self):
        super(VGG19PerceptualLoss, self).__init__()
        # Extract the first 35 layers of VGG19 model
        self.vgg19 = models.vgg19(pretrained=True).features[:35].eval()  

        # Normalization required when using pre-trained VGG19 model
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])

    def forward(self, input_image, target_image):
        # Convert single-channel images to 3-channel by replication
        input_image = torch.cat([input_image] * 3, dim=1)
        target_image = torch.cat([target_image] * 3, dim=1)

        # Preprocess input images with normalization
        input_image = self.normalize(input_image)
        target_image = self.normalize(target_image)

        vgg19 = self.vgg19.to('cuda')

        # Compute features using VGG19 model
        input_features = vgg19(input_image)
        target_features = vgg19(target_image)

        # Calculate perceptual loss
        perceptual_loss = nn.MSELoss()(input_features, target_features)

        return perceptual_loss