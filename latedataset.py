import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import math, random
from tqdm import tqdm

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

class lateDataset_meta(object):
    def __init__(self, imgPath_s, gtPath, featPath, listFiles, listFeat, test_batchsize=32):
        self.imgPath_s = imgPath_s
        self.gtPath = gtPath
        self.featPath = featPath
        self.listFeat = listFeat
        self.listFiles = listFiles
        self.test_batchsize = test_batchsize
        self.test_count = 0

        # self.all_ims = []
        # for n in listFiles:
        #     valim = pil_loader_g(os.path.join(self.imgPath_s, n))
        #     valim = torch.from_numpy(np.array(valim)).squeeze()
        #     valim = valim.float().div(255.)
        #     val_ims.append(valim)
        # self.all_ims = torch.stack(val_ims, 0).unsqueeze(1)
        # self.all_feats = []
        # for n in listFeats:
        #     valim = pil_loader_g(os.path.join(self.imgPath_s, n))
        #     valim = torch.from_numpy(np.array(valim)).squeeze()
        #     valim = valim.float().div(255.)
        #     val_feats.append(valim)
        # self.all_feats = torch.stack(val_feats, 0).unsqueeze(1)
        
        # self.all_gts = []
        # for n in self.listGtFiles:
        #     valim = pil_loader_g(os.path.join(gtPath, n))
        #     valim = torch.from_numpy(np.array(valim)).squeeze()
        #     valim = valim.float().div(255.)
        #     val_ims.append(valim)
        # self.all_gts = torch.stack(val_ims, 0).unsqueeze(1)
        # (len, 1, 224, 224)
        self.all_indices = list(range(len(self.listFeat)))


    def __len__(self):
        return self.all_feats.size(0)

    def sample(self, num_train=4, num_test=100, sample_test=False):
        if not sample_test:
            self.test_count = 0
            self.indices = random.sample(self.all_indices, num_train+num_test)  # sampled indices for meta learning
        train_indices = self.indices[:num_train]
        test_indices = self.indices[-num_test:]
        self.remaining_indices = [k for k in self.all_indices if k not in self.indices]  # for testing
        ims = []
        feats = []
        gts = []
        for i in train_indices:
            im = pil_loader_g(os.path.join(self.imgPath_s, self.listFiles[i]))
            im = torch.from_numpy(np.array(im)).squeeze()
            im = im.float().div(255.)
            ims.append(im)

            im = pil_loader_g(os.path.join(self.featPath, self.listFeat[i]))
            im = torch.from_numpy(np.array(im)).squeeze()
            im = im.float().div(255.)
            feats.append(im)

            gtname = self.listFiles[i].split('_')
            gtname = gtname[:2] + ['gt'] + gtname[2:]
            gtname = '_'.join(gtname)
            im = pil_loader_g(os.path.join(self.gtPath, gtname))
            im = torch.from_numpy(np.array(im)).squeeze()
            im = im.float().div(255.)
            gts.append(im)

        ims = torch.stack(ims, 0)
        feats = torch.stack(feats, 0)
        gts = torch.stack(gts, 0)

        val_ims = []
        val_feats = []
        val_gts = []
        for i in test_indices:
            im = pil_loader_g(os.path.join(self.imgPath_s, self.listFiles[i]))
            im = torch.from_numpy(np.array(im)).squeeze()
            im = im.float().div(255.)
            val_ims.append(im)

            im = pil_loader_g(os.path.join(self.featPath, self.listFeat[i]))
            im = torch.from_numpy(np.array(im)).squeeze()
            im = im.float().div(255.)
            val_feats.append(im)

            gtname = self.listFiles[i].split('_')
            gtname = gtname[:2] + ['gt'] + gtname[2:]
            gtname = '_'.join(gtname)
            im = pil_loader_g(os.path.join(self.gtPath, gtname))
            im = torch.from_numpy(np.array(im)).squeeze()
            im = im.float().div(255.)
            val_gts.append(im)

        val_ims = torch.stack(ims, 0)
        val_feats = torch.stack(feats, 0)
        val_gts = torch.stack(gts, 0)

        if sample_test:
            if (self.test_count+1)*self.test_batchsize >= len(self.remaining_indices):
                return {'im': ims, 'gt': gts, 'feat': feats, 'valim': val_ims, 'valfeat': val_feats, 'valgt': valgts, \
                'testim': None, 'testfeat': None, 'testgt': None}
            indices = self.remaining_indices[self.test_count*self.test_batchsize:(self.test_count+1)*self.test_batchsize]
            self.test_count += 1
            testim = []
            testfeat = []
            testgt = []
            for i in indices:
                im = pil_loader_g(os.path.join(self.imgPath_s, self.listFiles[i]))
                im = torch.from_numpy(np.array(im)).squeeze()
                im = im.float().div(255.)
                testim.append(im)

                im = pil_loader_g(os.path.join(self.featPath, self.listFeat[i]))
                im = torch.from_numpy(np.array(im)).squeeze()
                im = im.float().div(255.)
                testfeat.append(im)

                gtname = self.listFiles[i].split('_')
                gtname = gtname[:2] + ['gt'] + gtname[2:]
                gtname = '_'.join(gtname)
                im = pil_loader_g(os.path.join(self.gtPath, gtname))
                im = torch.from_numpy(np.array(im)).squeeze()
                im = im.float().div(255.)
                testgt.append(im)
            testim = torch.stack(testim, 0)
            testfeat = torch.stack(testfeat, 0)
            testgt = torch.stack(testgt, 0)
        else:
            testim, testfeat, testgt = torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0])

        return {'im': ims, 'gt': gts, 'feat': feats, 'valim': val_ims, 'valfeat': val_feats, 'valgt': valgts, \
                'testim': testim, 'testfeat': testfeat, 'testgt': testgt}


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
