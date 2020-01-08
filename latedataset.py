import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import math
from tqdm import tqdm

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader_g(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class lateDataset(Dataset):
    def __init__(self, imgPath_s, gtPath, featPath, listFiles, listGtFiles, listFeat):
        self.imgPath_s = imgPath_s
        self.gtPath = gtPath
        self.featPath = featPath
        self.listFeat = listFeat
        self.listFiles = listFiles
        self.listGtFiles = listGtFiles

    def __len__(self):
        return len(self.listFeat)

    def __getitem__(self, index):
        im = pil_loader_g(os.path.join(self.imgPath_s, self.listFiles[index]))
        # gt = pil_loader_g(os.path.join(self.gtPath, self.listGtFiles[index]))
        gtname = self.listFiles[index].split('_')
        gtname = gtname[:2] + ['gt'] + gtname[2:]
        gtname = '_'.join(gtname)
        gt = pil_loader_g(os.path.join(self.gtPath, gtname))
        feat = pil_loader_g(os.path.join(self.featPath, self.listFeat[index]))
        im = torch.from_numpy(np.array(im))
        gt = torch.from_numpy(np.array(gt))
        feat = torch.from_numpy(np.array(feat))
        im = im.float().div(255)
        gt = gt.float().div(255)
        feat = feat.float().div(255)
        im = im.unsqueeze(0)
        gt = gt.unsqueeze(0)
        feat = feat.unsqueeze(0)
        return {'im': im, 'gt': gt, 'feat': feat}

class lateDataset_meta(Dataset):
    def __init__(self, imgPath_s, gtPath, featPath, listFiles, listGtFiles, listFeat, listValFiles=None, listValFeats=None):
        self.imgPath_s = imgPath_s
        self.gtPath = gtPath
        self.featPath = featPath
        self.listFeat = listFeat
        self.listFiles = listFiles
        self.listGtFiles = listGtFiles
        self.listValFiles = listValFiles
        self.listValFeats = listValFeats

        self.val_ims = torch.FloatTensor([1])
        self.val_feats = torch.FloatTensor([1])

        if self.listValFiles is not None:
            val_ims = []
            for n in listValFiles:
                valim = pil_loader_g(os.path.join(self.imgPath_s, n))
                valim = torch.from_numpy(np.array(valim))
                valim = valim.float().div(255.)
                val_ims.append(valim)
            self.val_ims = torch.stack(val_ims, 0)
            val_feats = []
            for n in listValFeats:
                valim = pil_loader_g(os.path.join(self.imgPath_s, n))
                valim = torch.from_numpy(np.array(valim))
                valim = valim.float().div(255.)
                val_feats.append(valim)
            self.val_feats = torch.stack(val_feats, 0)

    def __len__(self):
        return len(self.listFeat)

    def __getitem__(self, index):
        im = pil_loader_g(os.path.join(self.imgPath_s, self.listFiles[index]))
        # gt = pil_loader_g(os.path.join(self.gtPath, self.listGtFiles[index]))
        gtname = self.listFiles[index].split('_')
        gtname = gtname[:2] + ['gt'] + gtname[2:]
        gtname = '_'.join(gtname)
        gt = pil_loader_g(os.path.join(self.gtPath, gtname))
        feat = pil_loader_g(os.path.join(self.featPath, self.listFeat[index]))
        im = torch.from_numpy(np.array(im))
        gt = torch.from_numpy(np.array(gt))
        feat = torch.from_numpy(np.array(feat))
        im = im.float().div(255)
        gt = gt.float().div(255)
        feat = feat.float().div(255)
        im = im.unsqueeze(0)
        gt = gt.unsqueeze(0)
        feat = feat.unsqueeze(0)

        return {'im': im, 'gt': gt, 'feat': feat, 'valim': self.val_ims, 'valfeat': self.val_feats}


if __name__ == '__main__':

    gtPath = '../gtea_gts'
    listGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' not in k]
    listGtFiles.sort()
    listValGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' in k]
    listValGtFiles.sort()
    print('num of training samples: ', len(listGtFiles))


    imgPath_s = '../new_pred'
    listTrainFiles = [k for k in os.listdir(imgPath_s) if 'Alireza' not in k]
    #listGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' not in k]
    listValFiles = [k for k in os.listdir(imgPath_s) if 'Alireza' in k]
    #listValGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' in k]
    listTrainFiles.sort()
    listValFiles.sort()
    print('num of val samples: ', len(listValFiles))

    featPath = '../new_feat'
    listTrainFeats = [k for k in os.listdir(featPath) if 'Alireza' not in k]
    listValFeats = [k for k in os.listdir(featPath) if 'Alireza' in k]
    listTrainFeats.sort()
    listValFeats.sort()
    assert(len(listTrainFeats) == len(listTrainFiles))
    assert(len(listValGtFiles) == len(listValFiles))

    lateDatasetTrain = lateDataset(imgPath_s, gtPath, featPath, listTrainFiles, listGtFiles, listTrainFeats)
    lateDatasetVal = lateDataset(imgPath_s, gtPath, featPath, listValFiles, listValGtFiles, listValFeats)
