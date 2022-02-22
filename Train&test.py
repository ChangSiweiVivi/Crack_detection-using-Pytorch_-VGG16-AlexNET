#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dataset preparation

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


#AlexNet architecture

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


# In[ ]:


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


#Train&test

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

dir_train = '../input/crack-detection-book/train/train/'
dir_test = '../input/crack-detection-book/test/test/'
model_cp = './'
workers = 2
lr = 0.0001
batch_size = 64

def train_test():
    datafile_train = DS('train',dir_train)
    dataloader_train = DataLoader(datafile_train,batch_size = batch_size, shuffle = True, num_workers = workers, drop_last = True)
    datafile_test = DS('test',dir_test)
    dataloader_test = DataLoader(datafile_test,batch_size = batch_size, shuffle = False, num_workers = workers, drop_last = True)

    print(len(datafile_train))
    print(len(datafile_test))

    
    model = Net()
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
   
    Loss_epoch = []
    acc_epoch = []
    epoch = []
    for i in range(30):
        Loss = []
        acc = []
        for img, label in dataloader_train:
            img, label = Variable(img).cuda(), Variable(label).cuda()
            label_list = label.data.cpu().numpy().tolist()
            #print(label_list)
            out = model(img)
            loss = criterion(out,label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss.append(loss.data.cpu().numpy())
            #print('Frame {0}, train_loss {1}'.format(i*batch_size, loss)) 
            _, predicted = torch.max(out.data,1)
            predicted_list = predicted.data.cpu().numpy().tolist()
            predicted_list_split = [predicted_list[i:i+1] for i in range(0,len(predicted),1)]
            #print(predicted_list_split)
            right = []
            for j in range(len(predicted_list)):
                if label_list[j] == predicted_list_split[j]:
                    right.append('y')
            acc.append(len(right) / len(predicted_list))
            #print(acc)
                
        Loss_epoch.append(mean(Loss))
        acc_epoch.append(mean(acc))
        print('epoch{0}, train_loss{1}, train_accuray{2}'.format(i+1, Loss_epoch,acc_epoch))
        
        model.eval()
        acc_test_epoch = []
        loss_test_epoch = []
        with torch.no_grad():
            acc_test = []
            loss_test = []
            for img_t,label_t in dataloader_test:
                img_t, label_t = Variable(img).cuda(), Variable(label).cuda()
                label_t_list = label_t.data.cpu().numpy().tolist()
                out_t = model(img_t)
                loss = criterion(out_t,label_t.squeeze())
                loss_test.append(loss.data.cpu().numpy())
                _, predicted_t = torch.max(out_t.data, 1)
                predicted_t_list = predicted_t.data.cpu().numpy().tolist()
                predicted_t_split_list = [predicted_t_list[i:i+1] for i in range(0,len(predicted_t_list),1)]
                right_t = []
                for j in range(len(predicted_t_list)):
                    if label_t_list[j] == predicted_t_split_list[j]:
                        right_t.append('y')
                acc_test.append(len(right_t) / len(predicted_t_list))
                
        acc_test_epoch.append(mean(acc_test))
        loss_test_epoch.append(mean(loss_test))
        print('epoch{0}, test_loss{1}, test_accuracy{2}'.format(i+1, loss_test_epoch,acc_test_epoch))
        
        epoch.append(i+1)
            
    torch.save(model, model_cp + 'AlexNet.pkl')
    plt.figure('efigure')
    plt.plot(epoch,Loss_epoch)
    plt.plot(epoch,acc_epoch)
    plt.plot(epoch,acc_test_epoch)
    plt.show()

if __name__ == '__main__':
    train_test() 

