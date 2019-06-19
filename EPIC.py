import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os, math, random
from random import randint
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

EPIC_KITCHENS_PATH = '/mnt/nisho_data2/Datasets/EPIC-Dataset/EPIC_KITCHENS_2018/frames_rgb_flow/'
#  contains flow/ and rgb/
# in rgb/ contrains test/ and train/
# in train/ P01/ ....
# min len in train is 30 frames, max is a turn of heater, more than 10k frames

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader_g(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class EPIC_KITCHENS(Dataset):
    def __init__(self, filename, dataroot=EPIC_KITCHENS_PATH, cliplen=32,\
                 transform = None, modality='rgb', split='train'):
        self.dataset = pd.read_csv(filename, index_col=False)
        self.cliplen = cliplen
        self.dataroot = dataroot
        self.modality = modality
        self.split = split
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        if self.split =='train':
            verb = self.dataset.verb_class.values[index]
            noun = self.dataset.noun_class.values[index]
            uid = self.dataset.uid.values[index]
            pid = self.dataset.participant_id.values[index]
            vid = self.dataset.video_id.values[index]
            start_frame = self.dataset.start_frame.values[index]
            stop_frame = self.dataset.stop_frame.values[index]
            if stop_frame - start_frame >= self.cliplen:
                start = randint(start_frame, stop_frame-self.cliplen)
                stop = start + self.cliplen
                ims = self.get_single(start, stop, True, pid, vid)
            else:
                start = start_frame
                stop = stop_frame
                ims = self.get_single(start, stop, False, pid, vid)
            
            return {'ims': ims, 'verb':torch.LongTensor([verb]), 'noun':torch.LongTensor([noun]), 'pid': pid, 'vid':vid, 'uid':uid}
        else:
            uid = self.dataset.uid.values[index]
            pid = self.dataset.participant_id.values[index]
            vid = self.dataset.video_id.values[index]
            start_frame = self.dataset.start_frame.values[index]
            stop_frame = self.dataset.stop_frame.values[index]
            if stop_frame - start_frame >= self.cliplen:
                multi_ims = []
                for s in range(10):
                    start = randint(start_frame, stop_frame-self.cliplen)
                    stop = start + self.cliplen
                    ims = self.get_single(start, stop, True, pid, vid)
                    multi_ims.append(ims)
                multi_ims = torch.stack(multi_ims, 0)
            else:
                start = start_frame
                stop = stop_frame
                ims = self.get_single(start, stop, False, pid, vid)
                ims = ims.unsqueeze(0)
                multi_ims = ims.expand(10, -1, -1, -1, -1)
            return {'ims': multi_ims, 'pid': pid, 'vid':vid, 'uid':uid}

    def get_single(self, start, stop, enough, pid, vid):
        ims = []
        for i in range(start, stop):
            imname = os.path.join(self.dataroot, self.modality, self.split,\
             pid, vid, 'frame_%010d.jpg'%i)
            ims.append(self.transform(pil_loader(imname)))
        ims = torch.stack(ims, 1)
        if not enough:
            while ims.size(1) < 32:
                ims = torch.cat((ims, ims),1)
            ims = ims[:,:32]
        return ims
    
    def get_multiple(self, ):
        pass


if __name__ == '__main__':
    from transform import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
    D = EPIC_KITCHENS('/mnt/nisho_data2/hyf/EPIC-annotations/EPIC_train_action_labels.csv', 
        transform=Compose([Scale([224,224]), ToTensor(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    loader = DataLoader(dataset=D, batch_size=2, shuffle=False, num_workers=8, pin_memory=True,)
    print(len(loader))
    from tqdm import tqdm
    for i, sample in tqdm(enumerate(loader)):
        pass
        # print(sample['ims'].size())  #(b, 3, cliplen, 224, 224)
        # print(sample['vid'])  #['P01_01', 'P01_01']
        # print(sample['verb'].size())  #(b, 1)
        # print(sample['noun'].size())   #(b, 1)



