import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2

class RealsenseData(Dataset):
    def __init__(self, img_root = '../realsense/'):
        allims = os.listdir(img_root)
        self.ims = sorted([k for k in allims if 'color' in k])
        self.gts = sorted([k for k in allims if 'open' in k])
        self.root = img_root

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        im = cv2.imread(os.path.join(self.root, self.ims[i]))
        gt = cv2.imread(os.path.join(self.root, self.gts[i]), 0)
        return {'im': im, 'gt': gt}