from __future__ import print_function
import argparse
import os,sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
# from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# import pdb
import torch.nn.functional as F


# class STN3d(nn.Module):
#     def __init__(self):
#         super(STN3d, self).__init__()
#         self.conv1 = torch.nn.Conv1d(6, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)


#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x

# class N_Views_MLP(nn.Module):
#     def __init__(self,views=12):
#         super(N_Views_MLP,self).__init__()
#         self.views = views
#         self.conv1 = torch.nn.Conv1d(6, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#     def forward(self,x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = (self.bn3(self.conv3(x)))
#         return x

# class To64(nn.Module):
#     def __init__(self,views=12):
#         self.conv1=torch.nn.Conv1d(1024*views,64,1)
#         self.bn1=nn.BatchNorm1d(64)
#         self.views=views
#         self.r_mlps=list()
#         for r in range(self.views):
#             self.r_mlps.append(N_Views_MLP(self.views).cuda())
#     def forward(self,x):
#         r_mlp_result_glo=list()
#         r_vews= x.size()[1]
#         # x to R*B*6*N
#         x=x.transpose(0,1).transpose(2,3)
#         # send R B*6*N tensors to R different mlps
#         for r in range(r_vews):
#             r_mlp_result_glo.append(self.r_mlps[r](x[r]))
#         # concate n_views B*64*N to B*(64*view)*N
#         x=torch.cat(tuple(r_mlp_result_glo),1)
#         # B*(64*view)*N =>B*64*N
#         x=F.relu(self.bn1(self.conv1(x)))
#         return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True):
        super(PointNetfeat, self).__init__()
        # self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x,pointfeat
        # if self.global_feat:
        #     return x, trans
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        #     return torch.cat([x, pointfeat], 1), trans

class PointNetCls(nn.Module):
    def __init__(self, k = 2,views=12):
        super(PointNetCls, self).__init__()
        self.views=views
        self.feat = PointNetMultiView(global_feat=True,views=self.views)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        _,x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=0)

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2,views=12):
        super(PointNetDenseCls, self).__init__()
        self.views=views
        self.k = k
        self.feat = PointNetMultiView(global_feat=False,views=self.views)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x,_ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x

class PointNetMultiView(nn.Module):
    def __init__(self,k=2,views=12,global_feat = True):
        super(PointNetMultiView,self).__init__()
        self.global_feat=global_feat
        self.views=views
        self.k=k
        self.r_pointnets=list()
        self.conv1 = torch.nn.Conv1d(64*views, 64, 1)
        self.conv2=torch.nn.Conv1d(1024*views,1024,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(1024)
        for v in range(self.views):
            self.r_pointnets.append(PointNetfeat(global_feat=self.global_feat).cuda())
        self.r_mlps=nn.ModuleList(self.r_pointnets)
    def forward(self,x):
        n_pts = x.size()[2]
        r_pointnets_rst_glo=list()
        r_pointnets_rst_loc=list()
        r_vews= x.size()[1]
        # x to R*B*6*N
        x=x.transpose(1,3).transpose(2,3)
        # send R B*6*N tensors to R different mlps
        for r in range(r_vews):
            g,l=self.r_pointnets[r](x[:,:,r,:].squeeze(2))
            r_pointnets_rst_loc.append(l)
            r_pointnets_rst_glo.append(g)

        glo=torch.cat(tuple(r_pointnets_rst_glo),1)
        loc=torch.cat(tuple(r_pointnets_rst_loc),1)
        loc=F.relu(self.bn1(self.conv1(loc)))
        glo=glo.unsqueeze(-1)
        glo=F.relu(self.bn2(self.conv2(glo)))
        glo = glo.view(-1, 1024, 1).repeat(1, 1, n_pts)
        glo=torch.cat([glo, loc], 1)

        return glo,loc


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())
