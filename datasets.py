
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import sys


num_seg_classes = {'Earphone': 3, 'Motorbike': 6, 'Rocket': 3, 'Car': 4, 'Laptop': 2, 'Cap': 2, 'Skateboard': 3, 'Mug': 2, 'Guitar': 3, 'Bag': 2, 'Lamp': 3, 'Table': 3, 'Airplane': 4, 'Pistol': 3, 'Chair': 4, 'Knife': 2}

class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True,reference = [[0,0,0],[1,1,1],[1,1,-1],[-1,1,1],[-1,-1,1],[1,-1,1],[1,-1,-1],[-1,-1,1],[-1,-1,-1],[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]]):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.ref = reference
        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}


        self.meta = {}
        for item in self.cat:

            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')

            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))

            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])

                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

        self.datapath = []

        self.num_seg_classes = 0
        for item in self.cat:
            for fn in self.meta[item]:       
                self.datapath.append((item, fn[0], fn[1]))

            self.num_seg_classes+= num_seg_classes[item]
        
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))



        # if not self.classification:
        #     for i in range(len(self.datapath)//50):
        #         l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
        #         # if l > self.num_seg_classes:
        #         self.num_seg_classes += l




        #print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        
        #resample
        point_set = point_set[choice, :]
        
        seg = seg[choice]

        ref = np.array(self.ref)
        point_set = self.expand_ref(point_set,ref)

        #add the reference 


        # print point_set.shape

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        point_set=point_set.float()
        if self.classification: 
            return point_set, cls
        else:
            return point_set, seg


    def __len__(self):
        return len(self.datapath)
        """
    params:
        pointset_dim:N*3
        ref_dim:K*3
    return:
        expanded_pointset:N*K*6 
    """
    def expand_ref(self,pointset,ref):
        K,_ = ref.shape
        N,_ = pointset.shape
        pointset = np.expand_dims(pointset,1)
        pointset = np.repeat(pointset,K,1)
        ref = np.expand_dims(ref,0)
        ref = np.repeat(ref,N,0)

        expanded_pointset = np.concatenate((pointset,ref),2)
    
        return expanded_pointset


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())