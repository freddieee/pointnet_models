from __future__ import print_function
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import socket,os,sys,importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import part_dataset_all_normal
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from datasets import PartDataset
from pointnetMV1 import PointNetDenseCls

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=16881*20, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--n_views', type=int, default = 13,  help='view numbers')
FLAGS = parser.parse_args()

EPOCH_CNT = 0


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.lr
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

NUM_CLASSES = 50

# Shapenet official train/test split
DATA_PATH = os.path.join( 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TRAIN_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='trainval', return_cls_label=True)
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test', return_cls_label=True)
dataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,shuffle=True, num_workers=int(FLAGS.workers),pin_memory=True)
testdataloader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,shuffle=True, num_workers=int(FLAGS.workers),pin_memory=True)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_cls_label = np.zeros((bsize,), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg,cls = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
        batch_cls_label[i] = cls
    return batch_data, batch_label, batch_cls_label

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

net = PointNetDenseCls(k = NUM_CLASSES,views=FLAGS.n_views)
classifier=net

# classifier = torch.nn.DataParallel(net)
# cudnn.benchmark = True

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_cls_label = np.zeros((bsize,), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg,cls = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
        batch_cls_label[i] = cls
    return batch_data, batch_label, batch_cls_label

if FLAGS.model != '':
    print("Finish Loading")
    # classifier.load_state_dict(torch.load(opt.model))
    classifier.load_weights(FLAGS.model)
optimizer = optim.SGD(classifier.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)
classifier.cuda()

    
for epoch in range(FLAGS.max_epoch):
    for i, data in enumerate(dataloader, 0):
        ps,normal,seg,cls  = data
        ps, batch_label = Variable(batch_data), Variable(batch_label)
        batch_label=batch_label.type(torch.LongTensor)
        batch_data = batch_data.transpose(2,1) 
        print(batch_data.shape)
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()   
        optimizer.zero_grad()
        classifier = classifier.train()
        pred_val= classifier(batch_data)
        pred_val = pred_val.view(-1, NUM_CLASSES)
        batch_label = batch_label.view(-1,1)[:,0] - 1
        #print(pred.size(), target.size())
        print(pred_val)
        print(batch_label)
        print(pred_val.type())
        print(batch_label.type())
#         sys.exit(0)
        loss = F.nll_loss(pred_val, batch_label)
        
        
        loss.backward()
        print(pred_val.shape)
        sys.exit(0)
        optimizer.step()
        pred_choice = pred_val.data.max(1)[1]
        correct = pred_choice.eq(batch_label.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(), correct.item()/float(BATCH_SIZE*NUM_POINT)))
        
    for i, data in enumerate(dataloader, 0):
        log_string(str(datetime.now()))
        log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
        
        j, data = next(enumerate(testdataloader, 0))
        points, _,target,_ = data
        target=target.type(torch.LongTensor)
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1) 
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred_val= classifier(points)
        pred_val = pred_val.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        print(pred_val.shape)
        sys.exit(0)
        loss_val = F.nll_loss(pred_val, trget)
        pred_choice = pred_val.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        cur_pred_val = pred_val[0:cur_batch_size]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i,0]]
            logits = cur_pred_val_logits[i,:,:]
            cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
        correct = np.sum(cur_pred_val == cur_batch_label)
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        if cur_batch_size==BATCH_SIZE:
            loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label==l)
            total_correct_class[l] += (np.sum((cur_pred_val==l) & (cur_batch_label==l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i,:]
            segl = cur_batch_label[i,:] 
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                    part_ious[l-seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l-seg_classes[cat][0]] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_shape_ious = np.mean(shape_ious.values())
    log_string('eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))
         
    EPOCH_CNT += 1
    
    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))