# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 11:38:42 2022

@author: q56104034
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import os

from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import glob
import random
#from age_predictor_model import Bottleneck, AgePredictor



import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.chdir('/home/kneehit/Data Science/Bone Age Kaggle/PyTorch')



#%%
# Image and CSV paths
dataset_path = '../../data/'
label_path = '../../label_data/'
csv_path = '../../boneage_file.csv'



torch.backends.cudnn.deterministic=True

#%% 

img_size = 500
image_filenames = glob.glob(dataset_path+'*.bmp')
label_image_filenames = glob.glob(label_path+'*.bmp')

means1 = []
stds1 = []
means2 = []
stds2 = []
means3 = []
stds3 = []


len(image_filenames)
        
for filename in image_filenames : #filename = image_filenames[0]


    image = cv2.imread(filename,0)
    image = cv2.resize(image,(img_size,img_size))

    
    # print( filename )
    label_name = 'label_'+filename
    label_img = cv2.imread(label_name,0)
    label_img = cv2.resize(label_img,(img_size,img_size))    
    label_img[label_img<255] = 0
    label_img[label_img==255] = 1
    label_img_float64 = label_img.astype(np.float64)         
    

    image = image*label_img
        
    seg_image = label_img*image

    temp = np.zeros((img_size, img_size))
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
    clahe_img = clahe.apply(seg_image)
    clahe_img = label_img*clahe_img
    temp = clahe_img
    temp = temp.astype(np.float64)
    temp/=255              
    seg_clahe_img = temp
    
    seg_clahe_img[seg_clahe_img<0.7]=0.0           

    seg_image = seg_image.astype(np.float64)
    seg_image/=255              
    clahe_img = clahe_img.astype(np.float64)
    clahe_img/=255  
        

    # cv2.imshow('My Image',seg_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    

    # merge = cv2.merge([seg_clahe_img, clahe_img])
    # merge = cv2.merge([merge, seg_image])
        
    mean1,std1 = cv2.meanStdDev(seg_clahe_img)
    mean2,std2 = cv2.meanStdDev(clahe_img)
    mean3,std3 = cv2.meanStdDev(seg_image)
    #    mean /= 255
    #    std /= 255
    
    means1.append(mean1)
    stds1.append(std1)
    means2.append(mean2)
    stds2.append(std2)
    means3.append(mean3)
    stds3.append(std3)
    
    
    
avg_mean1 = np.mean(means1) 
avg_std1 = np.mean(stds1)
avg_mean2 = np.mean(means2) 
avg_std2 = np.mean(stds2)
avg_mean3 = np.mean(means3) 
avg_std3 = np.mean(stds3)

#%%






age_df = pd.read_csv('../../boneage_file.csv')
age_df['path'] = age_df['id'].map(lambda x: os.path.join('../../data/', 
                                                         '{}.bmp'.format(x)))
age_df['exists'] = age_df['path'].map(os.path.exists)

age_df['exists'] = age_df['exists'].map(lambda x: True if x else np.nan )

# age_df['gender'] = age_df['sex_male(1)'].map(lambda x: 'male' if x else np.nan)

age_df = age_df.dropna(axis= 'index', how='any')


dataset_size = len(age_df)



age_df['boneage_category'] = pd.cut(age_df['bone_age'], 10)

print('Total:', age_df['exists'].sum(), 'images')




# bones_df = pd.read_csv(csv_path)
age_df.iloc[:,4:6] = age_df.iloc[:,4:6].astype(np.float64)
age_df.iloc[:,1:2] = age_df.iloc[:,1:2].astype(np.float64)

#%%

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(age_df, 
                                   test_size = 0.2, 
                                   stratify = age_df['boneage_category'])


test_df, val_df = train_test_split(test_df, 
                                   test_size = 0.5)



print('train data : ', str(len(train_df)), '\nvalidation data : ', str(len(val_df)), '\ntest data : ' , str(len(test_df)) )


age_max = np.max(age_df['bone_age'])
age_min = np.min(age_df['bone_age'])


#%%

# BonesDataset(dataframe = train_df,image_dir=dataset_path,transform = data_transform)

class BonesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):

        self.dataframe = dataframe

        
        self.image_dir = '../../data/'
        self.label_image_dir = '../../label_data/'

        self.transform = transform
        

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx): #idx = 0
    
        # img_name = image_dir + '13486809.2020.04.15.bmp' 
        # label_name = label_image_dir +'13486809.2020.04.15.bmp' 
        # hessian_name = hessian_dir + '13486809.2020.04.15.bmp' 
        
        # img_name = image_dir + '18123754.2020.02.05.bmp' 
        # label_name = label_image_dir +'18123754.2020.02.05.bmp' 
        # hessian_name = hessian_dir + '18123754.2020.02.05.bmp'       
    
        img_name = self.image_dir + str(self.dataframe.iloc[idx,0]) + '.bmp'
        label_name = self.label_image_dir + str(self.dataframe.iloc[idx,0]) + '.bmp'

        
        image = cv2.imread(img_name,0)
        image = cv2.resize(image, (img_size, img_size))
        # image = image.astype(np.float64)
        # image/=255 
        
        
        label_img = cv2.imread(label_name,0)
        label_img = cv2.resize(label_img, (img_size, img_size))
        label_img[label_img<255] = 0
        label_img[label_img==255] = 1
        label_img_float64 = label_img.astype(np.float64)




        # hessian_img = cv2.imread(hessian_name,0)
        # hessian_img = cv2.resize(hessian_img, (img_size, img_size))
        # hessian_img = hessian_img.astype(np.float64)
        # hessian_img/=255
        
        
        
        #%%
        #gray
        
        
        seg_image = label_img*image
        # hessian_img = label_img_float64*hessian_img

        # temp = np.zeros((img_size, img_size))
        # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
        # clahe_img = clahe.apply(seg_image)
        # clahe_img = label_img*clahe_img
        # clahe_img = clahe_img.astype(np.float64)
        # clahe_img/=255              
        # temp = clahe_img
        
        # clahe_img[clahe_img<0.7]=0.0 
        
        
        
        temp = np.zeros((img_size, img_size))
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
        clahe_img = clahe.apply(seg_image)
        clahe_img = label_img*clahe_img
        temp = clahe_img
        temp = temp.astype(np.float64)
        temp/=255              
        seg_clahe_img = temp
        
        seg_clahe_img[seg_clahe_img<0.7]=0.0           

        seg_image = seg_image.astype(np.float64)
        seg_image/=255              
        clahe_img = clahe_img.astype(np.float64)
        clahe_img/=255  

        
        
        # merge = cv2.merge([seg_image, hessian_img])
        merge = cv2.merge([seg_clahe_img, clahe_img])
        
        
        merge = cv2.merge([merge, seg_image])
        
        # cv2.imshow('My Image',merge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

     
        gender = np.atleast_1d(self.dataframe.iloc[idx,1])
        bone_age = np.atleast_1d(self.dataframe.iloc[idx,5])
        
        

        sample = {'image': merge, 'gender': gender, 'bone_age':bone_age}

        if self.transform:
            sample = self.transform(sample)

        return sample

#%% 
# Custom Transforms for Image and numerical data
        
# Resize and Convert numpy array to tensor
class ToTensor(object):
    

    def __call__(self, sample):
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']


        image = cv2.resize(image,(img_size,img_size))
        
        temp = np.zeros((3,img_size, img_size))
        temp[0, :, :] = image[:, :, 0]
        temp[1, :, :] = image[:, :, 1]
        temp[2, :, :] = image[:, :, 2]
        # temp[3, :, :] = image[:, :, 3]
        # temp[4, :, :] = image[:, :, 4]
        # temp[5, :, :] = image[:, :, 5]        
        image = temp
        # image = np.expand_dims(image,axis = 0)
        
#       we need to convert  cuda.longtensors to cuda.floatTensor data type
        return {'image': torch.from_numpy(image).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age':torch.from_numpy(bone_age).float()}        

# Normalize images and bone age

from sklearn import preprocessing
from matplotlib import pyplot as plt
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# d = scaler.fit_transform(image[:,:,0])


class Normalize(object):
    
    def __init__(self,img_mean,img_std,age_min,age_max):
        self.mean = img_mean
        self.std = img_std
        
        self.age_min = age_min
        self.age_max = age_max
        
    
    
    def __call__(self,sample): #image = merge
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']
        
        # image -= self.mean
        # image /= self.std
        
        # image[:,:,0] -= avg_mean
        # image /= avg_std
        
        
        for i in range(image.shape[2]): #i=0
            if i == 0:
                image[:,:,0] -= avg_mean1
                image[:,:,0] /= avg_std1
            elif i == 1:
                image[:,:,1] -= avg_mean2
                image[:,:,1] /= avg_std2
            elif i == 2:
                image[:,:,2] -= avg_mean3
                image[:,:,2] /= avg_std3
                
                
                
        image[:,:,0] = scaler.fit_transform(image[:,:,0])
        image[:,:,1] = scaler.fit_transform(image[:,:,1])
        image[:,:,2] = scaler.fit_transform(image[:,:,2])

        return {'image': image,
                'gender': gender,
                'bone_age':bone_age} 
        

data_transform = transforms.Compose([
   Normalize(avg_mean1,avg_std1,age_min,age_max),
   ToTensor()
   
   ])     
    


#%%


train_dataset = BonesDataset(dataframe = train_df,image_dir=dataset_path,transform = data_transform)
val_dataset = BonesDataset(dataframe = val_df,image_dir = dataset_path,transform = data_transform)
test_dataset = BonesDataset(dataframe = test_df,image_dir=dataset_path,transform = data_transform)


print(train_dataset[1])

    
train_data_loader = DataLoader(train_dataset,batch_size=4,shuffle=False)
val_data_loader = DataLoader(val_dataset,batch_size=4,shuffle=False)
test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False)

#%%
   

# Sanity Check 2
sample_batch =  next(iter(test_data_loader))
print(sample_batch)

#%%

import torch.nn.functional as F


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q, k.permute(0, 1, 3, 2))

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(q, content_position)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(attention, v)
        out = out.view(n_batch, C, width, height)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# reference

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, resolution=(224, 224), heads=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # for ImageNet
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out) # for ImageNet

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet50(num_classes=1000, resolution=(224, 224), heads=4):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads)


def main():
    x = torch.randn([2, 3, 224, 224])
    model = ResNet50(resolution=tuple(x.shape[2:]), heads=8)
    print(model(x).size())
    print(get_n_params(model))









#%% inceptionv3

age_predictor = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)

age_predictor.Conv2d_1a_3x3.conv=nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

age_predictor.fc=nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.Linear(in_features=1024, out_features=1, bias=True))

print(age_predictor)

#%%

age_predictor = age_predictor.to(device)
criterion = nn.L1Loss()
# optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.RMSprop(age_predictor.parameters(), lr=0.01)
optimizer = optim.Adam(age_predictor.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=25, verbose=1, eps=0.0001, cooldown=5, min_lr=0.0001)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=25, verbose=1, eps=1, cooldown=5, min_lr=0.0001)




# Training Loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=2000): # model = age_predictor
    
    best_valLoss = 9999
    
    for epoch in range(num_epochs):
   
        model.train()
        running_loss = 0.0
        val_running_loss = 0.0
        
        for batch_no,batch in enumerate(train_data_loader):
            # Load batch

            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)

            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # Forward Pass
                # model(image)
                outputs = model(image)
                loss = criterion(outputs.logits, age)
                # loss = criterion(outputs, age)
                # Backprop
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
                optimizer.step()
                
            # Calculate Loss
            running_loss += loss.item() * image.size(0)
            
            # if (batch_no + 1) % 25 == 0: print('Epoch {} Batch {}/2000 '.format(epoch+1,batch_no+1))
            
        total_loss = running_loss / len(train_df)
        scheduler.step(loss)
        print('\n Epoch {}'.format(epoch+1))
        print('-' * 10)    
        print('Loss: {:.6f}'.format(total_loss))
        
        
    # Eval on validation set
        model.eval()
        for val_batch_no, val_batch in enumerate(val_data_loader):
            image = val_batch['image'].to(device)
            age = val_batch['bone_age'].to(device)
    
            
            optimizer.zero_grad()  
            # only forward pass, dont update gradients
            with torch.set_grad_enabled(False):
                outputs = model(image)
                loss = criterion(outputs, age)
                
            val_running_loss += loss.item() * image.size(0)
        
        val_loss = val_running_loss / len(val_df)
        
        print('Validation Loss {:.6f}'.format(val_loss))
            
            
        if best_valLoss > val_loss :
            best_valLoss = val_loss
            best_model = model
            total_epochs = scheduler.state_dict()['last_epoch'] + 1 
            print('Save model : epoch-{}-loss-{:.6f}-val_loss-{:.6f} '.format(total_epochs,total_loss,val_loss))
            if (val_loss < 12.0) :
                torch.save(best_model,'loss-{:.6f}-val_loss-{:.6f}-best_model.pth'.format(total_loss,best_valLoss))


    # save_checkpoint(states,filename = 'epoch-{}-loss-{:.6f}-val_loss-{:.6f}.pth.tar'.format(total_epochs,total_loss,val_loss))

    return model
#%%

model = train_model(age_predictor,criterion,optimizer,scheduler,num_epochs=2000)



#%%
# model = torch.load('./model/inceptionv3/3channel/male/loss-13.970724-val_loss-8.197616-best_model.pth')

def mae_months(model, val_data_loader):
    
    
    error=0.0
    for batch_no, val_batch in enumerate(val_data_loader):
        image = val_batch['image'].to(device)
        age = val_batch['bone_age'].to(device)
        
        outputs = model(image)
        for pred, gt in zip(outputs,age):
            error += abs(pred-gt)
        
        
    
    ave_error = error/float(len(val_df))
    
    

    return ave_error

