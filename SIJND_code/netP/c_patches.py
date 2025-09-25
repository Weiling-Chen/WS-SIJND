from torchvision.transforms.functional import to_tensor
import torch
from PIL import Image
import numpy as np

def NonOverlappingCropPatches(im, ref, patch_size=32):
    """
    NonOverlapping Crop Patches
    :param im: the distorted image
    :param ref: the reference image
    :param patch_size: patch size (default: 32)
    :return: patches
    """


    im = Image.fromarray(im, mode='L')
    ref = Image.fromarray(ref, mode='L')

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

def RandomCropPatches1(im, ref, patch_size=32, n_patches=32):
    """
    Random Crop Patches
    :param im: the distorted image
    :param ref: the reference image
    :param patch_size: patch size (default: 32)
    :param n_patches: numbers of patches (default: 32)
    :return: patches
    """
    im = Image.fromarray(im, mode='L')
    ref = Image.fromarray(ref, mode='L')

    w, h = im.size

    patches = ()
    ref_patches = ()
    for i in range(n_patches):
        w1 = np.random.randint(low=0, high=w-patch_size+1)
        h1 = np.random.randint(low=0, high=h-patch_size+1)
        patch = to_tensor(im.crop((w1, h1, w1 + patch_size, h1 + patch_size)))-0.5
        patches = patches + (patch,)
        ref_patch = to_tensor(ref.crop((w1, h1, w1 + patch_size, h1 + patch_size)))-0.5
        ref_patches = ref_patches + (ref_patch,)
    return torch.stack(patches), torch.stack(ref_patches)

def RandomCropPatches(data, patch_size=32, n_patches=32):
    """
    Random Crop Patches
    :param im: the distorted image
    :param ref: the reference image
    :param patch_size: patch size (default: 32)
    :param n_patches: numbers of patches (default: 32)
    :return: patches
    """
    # data:(B, 2, 3, 320, 320)
    batch_size = data.shape[0]

    a = torch.ones(batch_size, 2, 32, 1, 32, 32)
    for j in range (batch_size):
        im, ref = data[j]
        im = Image.fromarray(im, mode='L')
        ref = Image.fromarray(ref, mode='L')
        w, h = im.size

        patches = ()
        ref_patches = ()
        for i in range(n_patches):
            w1 = np.random.randint(low=0, high=w-patch_size+1)
            h1 = np.random.randint(low=0, high=h-patch_size+1)
            patch = to_tensor(im.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
            patches = patches + (patch,)
            ref_patch = to_tensor(ref.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
            ref_patches = ref_patches + (ref_patch,)
        img_p = torch.stack(patches)
        img_ref_p = torch.stack(ref_patches)
        b = torch.stack((img_p, img_ref_p))
        a[j]= b

    return a

if __name__ == "__main__":
    img = np.array(Image.open("01_1.png"))
    img_ref = np.array(Image.open("01.png"))
    # data = [np.array(img), np.array(img_ref)]
    # data = np.array(data)
    # data = np.expand_dims(data, axis=0)

    # data = np.random.rand(4, 2, 320, 320, 3)

    a = NonOverlappingCropPatches(img, img_ref)
    print(a)