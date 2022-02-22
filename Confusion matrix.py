#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import torch

data_transform = transforms.Compose([transforms.ToTensor()])

img_H = 224
img_W = 224

class DS(data.Dataset):
    def __init__(self,mode,dir):
        self.mode = mode
        self.list_file = []
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = data_transform
        
        if self.mode == 'train':
            for file in os.listdir(dir):
                self.list_file.append(file)
                self.list_img.append(dir + file)
                self.data_size += 1
                name = file.split(sep='.')
                if name[0] == 'crack':
                    self.list_label.append(0)
                else:
                    self.list_label.append(1)
        elif self.mode == 'test':
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.data_size += 1
                name = file.split(sep='.')
                if name[0] == 'crack':
                    self.list_label.append(0)
                else:
                    self.list_label.append(1)
        else:
            print('Undefined')
            
    def __getitem__(self,item):
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            img = img.resize((img_H,img_W))
            img = np.array(img)[:,:,:3]
            label = self.list_label[item]
            return self.transform(img),torch.LongTensor([label])
        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            img = img.resize((img_H,img_W))
            img = np.array(img)[:,:,:3]
            label = self.list_label[item]
            return self.transform(img),torch.LongTensor([label])
        else:
            print('None')
            
    def __len__(self):
        return self.data_size


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Alex_Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,4,0)
        self.conv2 = nn.Conv2d(96,256,5,1,2)
        self.conv3 = nn.Conv2d(256,384,3,1,1)
        self.conv4 = nn.Conv2d(384,384,3,1,1)
        self.conv5 = nn.Conv2d(384,256,3,1,1)
        self.fc1 = nn.Linear(9216,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,2)
        
    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out,kernel_size=3, stride=2)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,kernel_size=3, stride=2)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = self.conv5(out)
        out = F.relu(out)
        out = F.max_pool2d(out,kernel_size=3,stride=2)
        
        out = out.view(out.size()[0],-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        
        return F.softmax(out,dim=1)

#VGG16 architecture
class VGG16_Net(nn.Module):
    def __init__(self):
        super(VGG16_Net,self)
        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
        
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
        
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 1)
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 1)
        
        self.fc1 = nn.Linear(512*7*7,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,2)
        
    def forward(self,x):
        out = self.conv1_1(x)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = self.conv5_1(out)
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        out = self.conv5_3(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = out.view(out.size()[0],-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.dropout(out,p=0.5)
        out = self.fc3(out)
        
        return out


# In[ ]:


from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import os

dir_crack = '../input/crack-nocrack-img/crack/crack/'
dir_nocrack = '../input/crack-nocrack-img/nocrack/nocrack/'

model = Alex_Net()
model.cuda()
model = torch.load('../input/weight-alexnet/AlexNet_60.pkl')
model.eval()

datafile_crack = DS('test',dir_crack)
datafile_nocrack = DS('test',dir_nocrack)
print(len(datafile_crack))
print(len(datafile_nocrack))

TP=0
FN = 0
for index in range(len(datafile_crack)):
    img = datafile_crack.__getitem__(index)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()
    out = model(img)
    if out[0,0]>out[0,1]:
        TP += 1
    else:
        FN += 1

FP = 0
TN = 0
for index in range(len(datafile_nocrack)):
    img = datafile_nocrack.__getitem__(index)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()
    out = model(img)
    if out[0,0]>out[0,1]:
        FP += 1
    else:
        TN += 1
        
print('TP:{0}, FN:{1}, FP:{2}, TN:{3}'.format(TP,FN,FP,TN))

