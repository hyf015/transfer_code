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
from data.lateDataset import lateDataset_meta as lateDataset

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
        self.meta_size = meta_size
        self.meta_val = meta_val
        listGtFiles = [k for k in os.listdir(gtPath) if val_name in k]
        listGtFiles.sort()
        # listValGtFiles = [k for k in os.listdir(gtPath) if val_name in k]
        # if task is not None:
        #     listGtFiles = [k for k in listGtFiles if task in k]
        #     listValGtFiles = [k for k in listValGtFiles if task in k]
        # listValGtFiles.sort()
        # print('num of training LF samples: %d'%len(listGtFiles))

        imgPath_s = late_pred_path
        print('Loading SP predictions from /%s'%imgPath_s)
        listFiles = [k for k in os.listdir(imgPath_s) if val_name in k]
        # all_inds = list(range(len(listFiles)))
        # meta_training_inds = random.sample(all_inds, meta_size)
        # all_inds = [k for k in all_inds if k not in meta_training_inds]
        # meta_validation_inds = random.sample(all_inds, meta_val)
        # all_inds = [k for k in all_inds if k not in meta_validation_inds]
        listFiles.sort()
        # listTrainFiles = [listFiles[k] for k in meta_training_inds]
        # listValFiles = [listFiles[k] for k in meta_validation_inds]
        # listTestFiles = [listFiles[k] for k in all_inds]
        print('num of LF samples: ', len(listFiles))

        featPath = late_feat_path
        listFeats = [k for k in os.listdir(featPath) if val_name in k]
        listFeats.sort()
        # listTrainFeats = [listFeats[k] for k in meta_training_inds]
        # listValFeats = [listFeats[k] for k in meta_validation_inds]
        # listTestFeats = [listFeats[k] for k in all_inds]

        self.loader = lateDataset(imgPath_s, gtPath, featPath, listFiles, listFeats,)
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
        print('begin meta training')
        steps_outer = self.steps_outer
        steps_inner = self.steps_inner

        optimizer = torch.optim.Adam(self.model.params(), lr=self.lr_outer)
        valid_model = self.model.clone()
        valid_optim = torch.optim.SGD(valid_model.params(), lr = self.lr_inner)
        for i in tqdm(range(steps_outer), disable=self.disable_tqdm):
            for j in range(steps_inner):
                print(j)
                self.meta_model.copy(self.model, same_var=True)
                datadict = self.loader.sample(num_train=self.meta_size, num_test=self.meta_val, )
                train_data = (datadict['feat'], datadict['im'], datadict['gt'])
                task_loss = self.inner_loop(train_data, self.lr_inner)
                test_data = (datadict['featval'], datadict['imval'], datadict['gtval'])

            new_task_loss = forward_and_backward(self.meta_model, test_data, train_data=train_data)
            optimizer.step()
            optimizer.zero_grad()

            valid_model.cpoy(self.model)
            train_loss = forward_and_backward(valid_model, test_data, valid_optim)

        torch.save(self.model.state_dict(), 'save/meta_model.pth.tar')
        print('training done')


    def test(self, ):
        print('begin test')
        model = self.model.clone()
        losses = AverageMeter()
        auc = AverageMeter()
        aae = AverageMeter()
        while True:
            datadict = self.loader.sample(num_train=self.meta_size, num_test=self.meta_val, sample_test=True)
            test_data = (datadict['testfeat'], datadict['testim'], datadict['testgt'])
            if test_data[0] is None:
                break
            
            with torch.no_grad():
                im = test_data[0].to(self.device)
                gt = test_data[2].to(self.device)
                feat = test_data[1].to(self.device)
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
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_outer', type=float, default=1e-3, required=False, help='lr for LF Adam')
    parser.add_argument('--late_save_img', default='loss_late.png', required=False, help='name of train/val image of LF module')
    parser.add_argument('--pretrained_late', default='../save/late.pth.tar', required=False, help='pretrained LF module')
    parser.add_argument('--lstm_save_img', default='loss_lstm.png', required=False, help='name of train/val loss image of AT module')
    parser.add_argument('--save_late', default='best_late_small.pth.tar', required=False, help='name of saving trained LF module')
    parser.add_argument('--save_path', default='save', required=False)
    parser.add_argument('--gtPath', default='../gtea_gts', required=False, help='directory of all groundtruth gaze maps in grey image format')
    parser.add_argument('--loss_function', default='f', required=False, help= 'if is not set as f, use bce loss')
    parser.add_argument('--num_epoch', type=int, default=10, required=False, help='num of training epoch of LF and SP')
    parser.add_argument('--train_late', action='store_true', help='whether to train LF module')
    parser.add_argument('--extract_late', action='store_true', help='whether to extract training data for LF module')
    parser.add_argument('--extract_late_pred_folder', default='../gtea2_pred/', required=False, help='directory to store the training data for LF')
    parser.add_argument('--extract_late_feat_folder', default='../gtea2_feat/', required=False, help='directory to store the training data for LF')
    parser.add_argument('--device', default='0', help='now only support single GPU')
    parser.add_argument('--val_name', default='Alireza', required=False, help='cross subject validation')
    parser.add_argument('--task', default=None, required=False, help='cross task validation')
    parser.add_argument('--fixsacPath', default='fixsac', required=False, help='directory of fixation prediction txt files')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of LF')
    parser.add_argument('--meta_size', type=int, default=10, help='meta training size')
    parser.add_argument('--meta_val', type=int, default=100, help='meta validation size')
    parser.add_argument('--steps_outer', type=int, default=5)
    parser.add_argument('--steps_inner', type=int, default=10)
    parser.add_argument('--lr_inner', type=int, default=1e-5)
    parser.add_argument('--disable_tqdm', type=bool, default=False)
    args = parser.parse_args()

    device = torch.device('cuda:'+args.device)

    batch_size = args.batch_size
    lf = LF(pretrained_model = args.pretrained_late, save_path = args.save_path, late_save_img = args.late_save_img,\
            save_name = args.save_late, device = args.device, late_pred_path = args.extract_late_pred_folder, num_epoch = args.num_epoch,\
            late_feat_path = args.extract_late_feat_folder, gt_path = args.gtPath, val_name = args.val_name, batch_size = args.batch_size,\
            loss_function = args.loss_function, lr_outer=args.lr_outer, task = args.task, meta_size=args.meta_size, \
            meta_val=args.meta_val, steps_outer=args.steps_outer, steps_inner=args.steps_inner,\
            lr_inner=args.lr_inner, disable_tqdm=args.disable_tqdm)

    for i in range(1000):
        lf.train()
        lf.test()
