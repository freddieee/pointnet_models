from __future__ import print_function
import argparse
import os,sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
import torch.nn.functional as F
from pointnet6DV1Multiple import PointNetDenseCls

def func_miou(num_classes,target,pred_choice):
    part_ious = list()
    segl=target.detach().cpu().numpy()
    segp=pred_choice.detach().cpu().numpy()
    for l in range(num_classes):
        if l not in segl:
            continue
        else:
            if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                part_ious.append(1.0)
            else:
                part_ious.append( np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l))))
    return part_ious

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='./seg/V1M',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--n_views', type=int, default = 13,  help='view numbers')
parser.add_argument('--lr', type=float, default = 0.001,  help='learning rate')
parser.add_argument('--momentum', type=float, default = 0.9,  help='momentum')
parser.add_argument('--classType', type=str, default = 'Bag',  help='class')
parser.add_argument('--devices',type=list,default=[0],help='multiple devices supported')
opt = parser.parse_args()
opt.devices=[int(i) for i in opt.devices]
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

classes = ['Bag','Chair','Car','Mug','Table','Airplane','Cap','Earphone','Guitar','Knife','Lamp','Laptop','Motorbike','Pistol','Rocket','Skateboard']
dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False,class_choice = classes, npoints = opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, train = False, class_choice = classes,npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k = num_classes,views=opt.n_views)

if opt.model != '':
    print("Finish Loading")
    classifier.load_state_dict(torch.load(opt.model))

classifier=torch.nn.DataParallel(classifier, device_ids=opt.devices)
cudnn.benchmark=True

if opt.model != '':
    print("Finish Loading")
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=opt.momentum)
classifier.cuda()

num_batch = len(dataset)/opt.batchSize
miou_list=list()
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1) 
        points, target = points.cuda(), target.cuda()   
        optimizer.zero_grad()
        classifier = classifier.train()
        pred= classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize*opt.num_points)))
        
        if i % 100 == 0:

            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1) 
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred= classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] - 1

            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            ioumax=func_miou(num_classes,target,pred_choice)
            iou=sum(ioumax)/len(ioumax)
            miou_list.append(iou)
            miou=np.mean(miou_list)
            a=('V1_multiple [%d: %d/%d] %s loss: %f accuracy: %f IOU: %f mIOU %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize*opt.num_points),iou,miou))
            f = open(opt.outf+"/log.txt", "a")
            f.write(a)
            f.close
            print(a)
    
    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))