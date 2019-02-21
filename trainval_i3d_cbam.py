import torch
import torch.nn as nn
import os, time, sys, cv2
from maskrcnn_benchmark.structures.bounding_box import BoxList
from models.i3d_cbam import InceptionI3D_att
from data.EGTEA import EGTEA, collate, accuracy
from data.transform import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, required=False)
parser.add_argument('--device', default='0', required=False)
parser.add_argument('--save_path', default='save', required=False)
parser.add_argument('--save_name', default='cbam.pth', required=False)
parser.add_argument('--loss_save', default='cbam.png', required=False)
parser.add_argument('--batch_size', default=5, type=int, required=False)
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(os.path.join(args.save_path, cbam)):
    os.makedirs(os.path.join(args.save_path, cbam))
if not os.path.exists(os.path.join(args.save_path, cbam, gt)):
    os.makedirs(os.path.join(args.save_path, cbam, gt))
device = torch.device('cuda:'+args.device)
if not torch.cuda.is_available():
    device = torch.device('cpu')

Dt = EGTEA('/home/hyf/EGTEA/cropped_clips', '/home/hyf/EGTEA/action_annotation/train_split1.txt', max_len = 24, transform=Compose([
        RandomHorizontalFlip(),
     ToTensor(255), 
     Normalize([102.9801/255, 115.9465/255, 122.7717/255],[1,1,1]),
     ]),)

Dv = EGTEA('/home/hyf/EGTEA/cropped_clips', '/home/hyf/EGTEA/action_annotation/test_split1.txt', max_len = 24, transform=Compose([
    RandomHorizontalFlip(),
 ToTensor(255), 
 Normalize([102.9801/255, 115.9465/255, 122.7717/255],[1,1,1]),
 ]))

trainloader = DataLoader(dataset=Dt, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
testloader = DataLoader(dataset=Dv, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
model = InceptionI3D_att(num_classes=400)
model.replace_logits(106, True)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0000001)
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [20, 60, 100])
criterion = torch.nn.CrossEntropyLoss().to(device)

def vis_cbam(ims, gt, cbam, name):
    ims = ims[0,:,0,:,:]
    ims = ims.add(torch.FloatTensor([102.9801/255, 115.9465/255, 122.7717/255]).view(3,1,1))
    ims = ims.cpu().numpy()
    ims = ims.transpose((1,2,0))
    ims = (ims*255).astype(np.uint8)
    gt = gt[0,:,0,:,:].squeeze().cpu().numpy()
    cbam = cbam[0,:,0,:,:].squeeze().cpu().numpy()
    gt = (gt*255).astype(np.uint8)
    cbam = (cbam*255).astype(np.uint8)

    colormap = cv2.applyColorMap(cbam, cv2.COLORMAP_JET)
    res = ims * 0.7 + colormap * 0.3
    cv2.imwrite(name, res)

    colormap = cv2.applyColorMap(gt, cv2.COLORMAP_JET)
    res = ims * 0.7 + colormap * 0.3
    gtname = name.split('/')
    gtname.insert(2, 'gt')
    gtname = '/'.join(gtname)
    cv2.imwrite(gtname, res)



def train(model, optimizer, criterion, dataloader, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    aae = AverageMeter()
    auc = AverageMeter()
    end = time.time()
    loss_mini_batch = 0.0
    top1 = AverageMeter()

    for i,sample in enumerate(dataloader):
        optimizer.zero_grad()
        ims = sample[0].to(device) #(b, t, c, h, w)
        gt_activity = sample[2].to(device)
        gt_gaze = sample[1].to(device)
       
        logits, x, att_map = model(ims)

        loss = criterion(out, gt_activity)
        loss_mini_batch = loss.item()
        loss.backward()

        optimizer.step()
        losses.update(loss_mini_batch, ims.size(0))
        prec1 = accuracy(logits, gt_activity)
        top1.update(prec1.item(), ims.size(0))

        att_map = torch.nn.functional.interpolate(att_map, size=gt_gaze.size(), mode='trilinear')
        aae1, auc1, gp = computeAAEAUC(att_map.squeeze().cpu().numpy(), gt_gaze.squeeze().cpu().numpy(), size=(240,320))
        aae.update(aae1)
        auc.update(auc1)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i) % 100 ==0:
            name = 'save/cbam/cbam%02d_%04d.jpg'%(epoch, i)
            vis_cbam(ims, gt_gaze, boxes, adj_p.detach(), name)
            print('Train Epoch: [{0}][{1}/{2}]\t'
              'accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'AAE {aae.val:.4f} ({aae.avg:.4f})\t'
              'AUC {auc.val:.4f} ({auc.avg:.4f})\t'.format(epoch, i+1, len(dataloader)+1, top1=top1, loss=losses, aae=aae, auc=auc))
    return losses.avg, top1.avg, aae.avg, auc.avg

def validate(model, criterion, dataloader, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    loss_mini_batch = 0.0
    top1 = AverageMeter()
    aae = AverageMeter()
    auc = AverageMeter()
    with torch.no_grad():
        for i,sample in enumerate(dataloader):
            ims = sample[0].to(device) #(b, t, c, h, w)
            gt_activity = sample[2].to(device)
            gt_gaze = sample[1].to(device)
           
            logits, x, att_map = model(ims)

            loss = criterion(out, gt_activity)
            loss_mini_batch = loss.item()

            losses.update(loss_mini_batch, ims.size(0))
            prec1 = accuracy(logits, gt_activity)
            top1.update(prec1.item(), ims.size(0))

            att_map = torch.nn.functional.interpolate(att_map, size=gt_gaze.size(), mode='trilinear')
            aae1, auc1, gp = computeAAEAUC(att_map.squeeze().cpu().numpy(), gt_gaze.squeeze().cpu().numpy(), size=(240,320))
            aae.update(aae1)
            auc.update(auc1)
            batch_time.update(time.time() - end)
            end = time.time()
            if (i) % 200 ==0:
                print('Test Epoch: [{0}][{1}/{2}]\t'
              'accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'AAE {aae.val:.4f} ({aae.avg:.4f})\t'
              'AUC {auc.val:.4f} ({auc.avg:.4f})\t'.format(epoch, i+1, len(dataloader)+1, top1=top1, loss=losses, aae=aae, auc=auc))
    return losses.avg, top1.avg, aae.avg, auc.avg

def plot_loss(losses, legends, save_path):
    for loss in losses:
        plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legends, loc = 'upper right')
    plt.savefig(os.path.join(save_path,args.loss_save))
    plt.close()


def trainval_epoch(model, optimizer, criterion, trainloader, testloader, num_epoch=100):
    best_acc = 0
    save_path = args.save_path
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(num_epoch):
        lr_sched.step()
        tl,ta,taaep,taucp= train(model, optimizer, criterion, trainloader, epoch)
        train_loss.append(tl)
        train_acc.append(ta)
        print('train epoch %03d, loss: %04f, acc: %04f, aae: %04f, auc: %04f'
            %(epoch, tl, ta, taaep, taucp))
        print('========+========+========+========+========+========+========+')

        loss1, acc1, aaep, aucp = validate(model, criterion, testloader, epoch)
        test_loss.append(loss1)
        test_acc.append(acc1)
        print('epoch%03d, val loss is: %04f, val acc is : %04f, val aae is : %04f, val auc is : %04f'
         % (epoch, loss1, acc1, aaep, aucp))
        
        checkpoint_name = args.save_name
        plot_loss([train_loss, test_loss], ['train', 'test',], args.save_path)
        plot_loss([train_acc, test_acc], ['train', 'test',], args.save_path)
        if acc1 > best_acc:
            best_acc = acc1
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'loss': loss1},
                            checkpoint_name, args.save_path)
        print('best acc is: %f'%best_acc)

if __name__ == '__main__':
    trainval_epoch(model, optimizer, criterion, trainloader, testloader, 150)
