import torch
import glob
from os.path import join
from PIL import Image
import json
from os import listdir
import os
import numpy as np
import imageio
import torch.nn.functional as  F
import cv2
from tqdm import tqdm
import string
import pickle
import PIL
import io
from torchvision import transforms



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, transform=None, n_max=-1, composite_on_white=False):
        self.transform = transform
        self.composite_on_white = composite_on_white

        self.images = glob.glob(join(ds_path, '*.jpg')) + glob.glob(join(ds_path, '*.png')) + glob.glob(join(ds_path, '*.npy'))
        self.images.sort()
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.images[idx][-3:] == 'npy':
            img = Image.fromarray(np.load(self.images[idx])[0].transpose(1, 2, 0))
        else:
            img = Image.open(self.images[idx])
        if self.composite_on_white:
            png = img.convert('RGBA')
            background = Image.new('RGBA', png.size, (255,255,255))
            img = Image.alpha_composite(background, png).convert("RGB")
        else:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return {
            'image': img,
        }
