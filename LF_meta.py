import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy import ndimage
import math
from tqdm import tqdm
import os
from random import randint

from floss import floss
from utils import *
from models.late_fusion import late_fusion_meta as late_fusion
from data.lateDataset import lateDataset

class LF():
    def __init__(self, pretrained_model, save_path = 'save', late_save_img = 'loss_late_meta.png',\
            save_name = 'best_late.pth.tar', device = '0', late_pred_path = '../gtea2_pred', num_epoch = 10,\
            late_feat_path = '../gtea2_feat', gt_path = '../gtea_gts', val_name = 'Alireza', batch_size = 32,\
            loss_function = 'f', lr_outer=1e-3, task = None, meta_size=5, meta_val=100, steps_outer=1, steps_inner=1,\
            lr_inner=1e-2, disable_tqdm=False):
        self.model = late_fusion()
        self.meta_model = self.model.clone()
        self.device = torch.device('cuda:'+device)
        self.save_name = save_name
        self.save_path = save_path
        pretrained_dict = torch.load(pretrained_model, map_location='cpu')
        pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('loaded pretrained late fusion model from '+ pretrained_model)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.epochnow = 0
        self.late_save_img = late_save_img
        gtPath = gt_path

        self.disable_tqdm = disable_tqdm
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.steps_inner = steps_inner
        self.steps_outer = steps_outer
        # listGtFiles = [k for k in os.listdir(gtPath) if val_name not in k]
        # listGtFiles.sort()
        # listValGtFiles = [k for k in os.listdir(gtPath) if val_name in k]
        # if task is not None:
        #     listGtFiles = [k for k in listGtFiles if task in k]
        #     listValGtFiles = [k for k in listValGtFiles if task in k]
        # listValGtFiles.sort()
        # print('num of training LF samples: %d'%len(listGtFiles))

        imgPath_s = late_pred_path
        print('Loading SP predictions from /%s'%imgPath_s)
        listFiles = [k for k in os.listdir(imgPath_s) if val_name in k]
        randstart = randint(0, len(listTrainFiles)- meta_size - meta_val)
        listTrainFiles = listFiles[randstart:randstart+meta_size]
        listValFiles = listFiles[randstart+meta_size: randstart+meta_size+meta_val]
        listTestFiles = listFiles[:randstart] + listFiles[randstart+meta_size+meta_val:]
        if task is not None:
            x=input('not implemented yet...')
            listTrainFiles = [k for k in listTrainFiles if task in k]
            listValFiles = [k for k in listValFiles if task in k]
        listTrainFiles.sort()
        listValFiles.sort()
        listTestFiles.sort()
        print('num of LF val samples: ', len(listValFiles))

        featPath = late_feat_path
        listFeats = [k for k in os.listdir(featPath) if val_name in k]
        listTrainFeats = listFeats[randstart:randstart+meta_size]
        listValFeats = listFeats[randstart+meta_size: randstart+meta_size+meta_val]
        listTestFeats = listFeats[:randstart] + listFiles[randstart+meta_size+meta_val:]
        # if task is not None:
        #     listTrainFeats = [k for k in listTrainFeats if task in k]
        #     listValFeats = [k for k in listValFeats if task in k]
        listTrainFeats.sort()
        listValFeats.sort()
        listTestFeats.sort()
        # assert(len(listTrainFeats) == len(listTrainFiles) and len(listGtFiles) > 0)
        # assert(len(listValGtFiles) == len(listValFiles))
        self.loader = DataLoader(dataset=lateDataset(imgPath_s, gtPath, featPath, listFiles, listGtFiles, listFeats, \
            listValFiles, listValFeats), batch_size = batch_size, shuffle=True, num_workers=0, pin_memory=True)
        if loss_function == 'f':
            self.criterion = floss().to(self.device)
        else:
            self.criterion = torch.nn.BCELoss().to(self.device)

    def forward_and_backward(self, data, optimizer=None, create_graph=False, train_data=None):
        self.model.train()
        if optimizer is not None:
            optimizer.zero_grad()
        loss = self.forward(data, train_data=train_data, for_backward=True)
        if optimizer is not None:
            optimizer.step()
        return loss

    def forward(self, data, return_predictions=False, train_data=None, for_backward=False):
        self.model.train()

        feat, im, gt = data
        y_hat = self.model(feat, im)
        loss = self.criterion(y_hat, gt)
        if return_predictions:
            return y_hat
        elif for_backward:
            return loss
        else:
            return loss.data.cpu()

    def train(self):

        optimizer = torch.optim.Adam(self.model.params(), lr=self.lr_outer)
        valid_model = self.model.clone()
        valid_optim = torch.optim.SGD(valid_model.params(), lr = self.lr_inner)
        for i in tqdm(range(steps_outer, disable=self.disable_tqdm)):
            for j in range(steps_inner):
                self.meta_model.copy(self.model, same_var=True)
                datadict = self.loader.sample(num_train=self.meta_size, num_test=self.meta_val, )
                train_data = (datadict['feat'], datadict['im'], datadict['gt'])
                task_loss = self.inner_loop(train_data, self.lr_inner)
                test_data = (datadict['featval'], datadict['imval'], datadict['gtval'])

            new_task_loss = forward_and_backward(self.meta_model, test_data, train_data=train_data)
            optimizer.step()
            optimizer.zero_grad()

        torch.save(self.model.state_dict(), 'save/meta_model.pth.tar')


    def test(self, ):
        model = self.model.clone()
        datadict = self.loader.sample(num_train=self.meta_size, num_test=self.meta_val, sample_test=True)
        test_data = (datadict['testfeat'], datadict['testim'], datadict['testgt'])
        total_len = test_data[0].size(0)
        batch_size = 32
        losses = AverageMeter()
        auc = AverageMeter()
        aae = AverageMeter()
        for i in range(total_len//32):
            im = test_data[0][i*batch_size:(i+1)*batch_size].to(self.device)
            gt = test_data[2][i*batch_size:(i+1)*batch_size].to(self.device)
            feat = test_data[1][i*batch_size:(i+1)*batch_size].to(self.device)
            out = model(feat, im)
            loss = self.criterion(out, gt)
            outim = out.cpu().data.numpy().squeeze()
            targetim = gt.cpu().data.numpy().squeeze()
            aae1, auc1, _ = computeAAEAUC(outim,targetim)
            auc.update(auc1)
            aae.update(aae1)
            losses.update(loss.item())

        print(losses.avg, 'auc: ', auc.avg, 'aae:', aae.avg)
        return losses.avg, auc.avg, aae.avg


    def inner_loop(self, train_data, lr_inner=0.01):
        loss = forward_and_backward(self.meta_model, train_data, create_graph=True)
        for name, param in self.meta_model.named_params():
            self.meta_model.set_param(name, param - lr_inner * param.grad)
        return loss
                
    # def trainLate(self):
    #     losses = AverageMeter()
    #     auc = AverageMeter()
    #     aae = AverageMeter()
    #     for i,sample in tqdm(enumerate(self.train_loader)):
    #         im = sample['im']
    #         gt = sample['gt']
    #         feat = sample['feat']
    #         im = im.float().to(self.device)
    #         gt = gt.float().to(self.device)
    #         feat = feat.float().to(self.device)
    #         out = self.model(feat, im)
    #         loss = self.criterion(out, gt)
    #         outim = out.cpu().data.numpy().squeeze()
    #         targetim = gt.cpu().data.numpy().squeeze()
    #         aae1, auc1, _ = computeAAEAUC(outim,targetim)
    #         auc.update(auc1)
    #         aae.update(aae1)
    #         losses.update(loss.item())
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         if (i+1)%3000 == 0:
    #             print('Epoch: [{0}][{1}/{2}]\t''AUCAAE_late {auc.avg:.3f} ({aae.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
    #                 self.epochnow, i+1, len(self.train_loader)+1, auc = auc, loss= losses, aae=aae,))

    #     return losses.avg, auc.avg, aae.avg

    # def testLate(self):
    #     losses = AverageMeter()
    #     auc = AverageMeter()
    #     aae = AverageMeter()
    #     with torch.no_grad():
    #         for i,sample in tqdm(enumerate(self.val_loader)):
    #             im = sample['im']
    #             gt = sample['gt']
    #             feat = sample['feat']
    #             im = im.float().to(self.device)
    #             gt = gt.float().to(self.device)
    #             feat = feat.float().to(self.device)
    #             out = self.model(feat, im)
    #             loss = self.criterion(out, gt)
    #             outim = out.cpu().data.numpy().squeeze()
    #             targetim = gt.cpu().data.numpy().squeeze()
    #             aae1, auc1, _ = computeAAEAUC(outim,targetim)
    #             auc.update(auc1)
    #             aae.update(aae1)
    #             losses.update(loss.item())
    #             if (i+1) % 1000 == 0:
    #                 print('Epoch: [{0}][{1}/{2}]\t''AUCAAE_late {auc.avg:.3f} ({aae.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
    #                     self.epochnow, i+1, len(self.val_loader)+1, auc = auc, loss= losses, aae=aae,))

    #     return losses.avg, auc.avg, aae.avg

    # def train(self):
    #     print('begin training LF module...')
    #     trainprev = 999
    #     valprev = 999
    #     loss_train = []
    #     loss_val = []
    #     for epoch in range(self.num_epoch):
    #         self.epochnow = epoch
    #         loss, auc, aae = self.trainLate()
    #         loss_train.append(loss)
    #         print('training, auc is %5f, aae is %5f'%(auc, aae))
    #         if loss < trainprev:
    #             torch.save({'state_dict': self.model.state_dict(), 'loss': loss, 'auc': auc, 'aae': aae}, os.path.join(self.save_path, self.save_name))
    #             trainprev = loss

    #         loss, auc, aae = self.testLate()
    #         loss_val.append(loss)
    #         plot_loss(loss_train, loss_val, os.path.join(self.save_path, self.late_save_img))
    #         if loss < valprev:
    #             torch.save({'state_dict': self.model.state_dict(), 'loss': loss, 'auc': auc, 'aae': aae}, os.path.join(self.save_path, 'val'+self.save_name))
    #             valprev = loss
    #         print('testing, auc is %5f, aae is %5f' % (auc, aae))
    #     print('LF module training finished!')

    # def val(self):
    #     print('begin testing LF module...')
    #     loss, auc, aae = self.testLate()
    #     print('AUC is : %04f, AAE is: %04f'%(auc, aae))
    #     print('LF module testing finished!')
