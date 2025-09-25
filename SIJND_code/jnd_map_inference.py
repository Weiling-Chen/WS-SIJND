import torch
import torch.nn as nn
from networks import UNetD, UNetG, sample_generator
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import torchvision.utils




def pre(ref_path, ic_path, sa_path, out_path, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    netG = UNetD(3, 32, 4).to(device)
    netG.load_state_dict(torch.load(model_path))
    netG.eval()

    ref_data = to_tensor(Image.open(ref_path).convert('L'))
    ic_data = to_tensor(Image.open(ic_path).convert('L'))
    sa_data = to_tensor(Image.open(sa_path).convert('L'))
    data = torch.cat((ref_data,ic_data,sa_data),dim=0)
    data = torch.unsqueeze(torch.tensor(data), dim=0)
    data = data.to(device)

    fake, vis = sample_generator(netG, data)
    torchvision.utils.save_image(fake, out_path)
#
for i in range(1, 31):
    ref_path = 'your/pristine_image/path/' + str(i).zfill(2) + '.bmp'
    ic_path = 'your/image_complexity_heatmap/path/' + str(i).zfill(2) + '.png'
    sa_path = 'your/edge_map/path/' + str(i).zfill(2) + '.bmp'
    out_path = 'your/output/path/' + str(i).zfill(2) + '.png'
    model_path = '../epoch124_fal/model_124.pth'
    pre(ref_path, ic_path, sa_path, out_path, model_path)