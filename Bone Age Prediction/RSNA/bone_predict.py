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
from tqdm import tqdm



import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.chdir('/home/kneehit/Data Science/Bone Age Kaggle/PyTorch')


#%%
# Image and CSV paths
dataset_path = 'data/'
label_path = 'label_data/'
# test_dataset_path = 'bone_data/test/'
# val_dataset_path = 'bone_data/validation/'
csv_path = 'boneage_file.csv'



# For reproducibility use the seeds below 
# torch.manual_seed(1498920)
# torch.cuda.manual_seed(1498920)
# np.random.seed(1498920)
# random.seed(1498920)
torch.backends.cudnn.deterministic=True

#%% 
# Sample random images from dataset (since loading all images will take time) 
# and calculate mean and standard deviation for normalization 
# k = 100
img_size = 512
image_filenames = glob.glob(dataset_path+'*.bmp')
label_image_filenames = glob.glob(label_path+'*.bmp')
#random_images = random.sample(population = image_filenames,k = k)

# for i in range(296): #i=0
#     if image_filenames[i][5:] == label_image_filenames[i][11:]:
#         print('true')
#     else: 
#         print('error')
#         print(image_filenames[i][5:])
#         print(label_image_filenames[i][11:])
#         print('````````````````````````````')


#len(random_images)

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
    # image = image.astype(np.float64)
    # image/=255                
    
    
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






age_df = pd.read_csv('boneage_file.csv')
age_df['path'] = age_df['id'].map(lambda x: os.path.join('./data/', 
                                                         '{}.bmp'.format(x)))
age_df['exists'] = age_df['path'].map(os.path.exists)

age_df['exists'] = age_df['exists'].map(lambda x: True if x else np.nan )

# age_df['gender'] = age_df['sex_male(1)'].map(lambda x: 'male' if x else np.nan)

age_df = age_df.dropna(axis= 'index', how='any')


dataset_size = len(age_df)

# for i in range (age_df.shape[0]): #i=12
#     if age_df['exists'][i] == '':
#         print(age_df['path'][i])
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


# random_seed=200                 


# train_df = age_df.sample(frac=0.8, random_state=random_seed)

# val_df = age_df[~age_df.index.isin(train_df.index)]

 
# test_df = val_df.sample(frac=0.5, random_state=random_seed)

# val_df = val_df[~val_df.index.isin(test_df.index)]





print('train data : ', str(len(train_df)), '\nvalidation data : ', str(len(val_df)), '\ntest data : ' , str(len(test_df)) )


age_max = np.max(age_df['bone_age'])
age_min = np.min(age_df['bone_age'])

# *(age_max-age_min)+age_min


# (bone_age - self.age_min)/ (self.age_max - self.age_min)

#%%

# BonesDataset(dataframe = train_df,image_dir=dataset_path,transform = data_transform)

class BonesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):

        self.dataframe = dataframe

        
        self.image_dir = 'data/'
        self.label_image_dir = 'label_data/'
        # self.hessian_dir = 'Hessian/'
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
        # hessian_name = self.hessian_dir + str(self.dataframe.iloc[idx,0]) + '.bmp'
        
        
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
        seg_clahe_img[seg_clahe_img>=0.7]=1.0  
        seg_image = seg_image.astype(np.float64)
        seg_image/=255              
        clahe_img = clahe_img.astype(np.float64)
        clahe_img/=255  

    
        
        
        # merge = cv2.merge([seg_image, hessian_img])
        merge = cv2.merge([seg_clahe_img, clahe_img])
        
        
        merge = cv2.merge([merge, seg_image])
        
        cv2.imshow('My Image',merge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

     
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
        # arr = []
        # for i in range(image[:,:,0].shape[0]): #i=0
        #     for j in range(image[:,:,0].shape[1]): #j=0
        #         if not image[:,:,0][i][j]  == 0.0:
        #             arr.append(image[:,:,0][i][j])
        

        
        # cv2.imshow('My Image',image[:,:,])
        # from matplotlib import pyplot as plt
        # plt.hist(arr)
        # plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        
        # bone_age /=12
        # bone_age = (bone_age - self.age_min)/ (self.age_max - self.age_min)
        # img_tensor=transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Resize((img_size, img_size)),
        #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #                                 transforms.Grayscale(num_output_channels=1)
        #                                 ])            
        # img=img_tensor(image)
        
        return {'image': image,
                'gender': gender,
                'bone_age':bone_age} 
        

data_transform = transforms.Compose([
   # Normalize(avg_mean1,avg_std1,age_min,age_max),
   ToTensor()
   
   ])     
    
# data_transform = transforms.Compose([
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensor()
#     ])     

#%%

# dataset = BonesDataset(dataframe = bones_df,image_dir=dataset_path,transform = data_transform)

train_dataset = BonesDataset(dataframe = train_df,image_dir=dataset_path,transform = data_transform)
val_dataset = BonesDataset(dataframe = val_df,image_dir = dataset_path,transform = data_transform)
test_dataset = BonesDataset(dataframe = test_df,image_dir=dataset_path,transform = data_transform)


    

# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [236, 59])
# val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [29, 30])


# train_set, val_set = torch.utils.data.random_split(dataset, [236, 10000])

# Sanity Check
print(train_dataset[1])

    
train_data_loader = DataLoader(train_dataset,batch_size=16,shuffle=False)
val_data_loader = DataLoader(val_dataset,batch_size=16,shuffle=False)
test_data_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)

#%%
   

# Sanity Check 2
sample_batch =  next(iter(test_data_loader))
print(sample_batch)

#%%


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)



#%%
# Bottleneck block for ResNet to reduce dimensions
# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out
    
    
    
#%%
# Custom Convolution Neural Network architecture based on ResNet



# class AgePredictor(nn.Module):
    
#     # Define and Initialize Layers
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(AgePredictor, self).__init__()
#         # ResNet Architecture
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,  # <-
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)                          # <-       
#         self.fc = nn.Linear(512 * block.expansion, 400)
#         self.res_relu = nn.ReLU()
        
        
#         # Fully Connected layer for gender
#         self.gen_fc_1 = nn.Linear(1,16)
#         self.gen_relu = nn.ReLU()
        
#         # Feature Concatenation Layer
#         self.cat_fc = nn.Linear(16+400,200)
#         self.cat_relu = nn.ReLU()
        
#         # Final Fully Connected Layer
#         self.final_fc = nn.Linear(200,num_classes)
        
#         # Simply using linear layer (w/o sigmoid) led to network predicting negative values for age
#         # Therefore input was scaled to range from 0 and 1
#         # and sigmoid is used as final layer to predict values which when 
#         # denormalized led to positive values
#         self.sigmoid = nn.Sigmoid()
        
#         # Weight Initialization 
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     # Forward Pass. x = Image tensor, y = gender tensor
#     def forward(self, x,y):
# # =============================================================================
# #       ResNet Layers        
# # =============================================================================
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = self.res_relu(x)
#         x = x.view(x.size(0), -1)
        

# # =============================================================================
# #       Gender Fully Connected Layer
# # =============================================================================
#         y = self.gen_fc_1(y)
#         y = self.gen_relu(y)
#         y = y.view(y.size(0), -1)

        
# # =============================================================================
# #       Feature Concatenation Layer
# # =============================================================================
      
#         z = torch.cat((x,y),dim = 1)
#         z = self.cat_fc(z)
#         z = self.cat_relu(z)

# # =============================================================================
# #       Final FC Layer
# # =============================================================================
        
#         z = self.final_fc(z)
#         z = self.sigmoid(z)

#         return z

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



# def Conv1(in_planes, places, stride=2):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
#         nn.BatchNorm2d(places),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        # self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.layer4_MHSA1 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(512),
            MHSA(n_dims=512, width=32, height=32, heads=4),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048)
        )
        
        self.MHSA_relu = nn.ReLU(inplace=True)
        
        self.layer4_MHSA2 = nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(512),
            MHSA(n_dims=512, width=16, height=16, heads=4),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        
        self.layer4_MHSA3 = nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(512),
            MHSA(n_dims=512, width=16, height=16, heads=4),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )    



        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048,968)

        self.gen_fc_1 = nn.Linear(1,32)
        self.gen_relu = nn.ReLU(inplace=True)
        
        # Feature Concatenation Layer
        self.cat_fc = nn.Linear(32+968,1000)
        self.cat_relu = nn.ReLU(inplace=True)
        
        
        self.fc2 = nn.Linear(1000, 1000)
        # Final Fully Connected Layer
        self.final_fc = nn.Linear(1000,1)
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x, y):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        residual = x
    
        out = self.layer4_MHSA1(x)
        residual = self.downsample(x)
        
        out += residual
        out = self.MHSA_relu(out)
        
        
        # x = self.layer4(x)
        out = self.layer4_MHSA2(out)
        
        
        out = self.layer4_MHSA3(out)
        
        
        

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        
        
    
        y = self.gen_fc_1(y)
        y = self.gen_relu(y)
        y = y.view(y.size(0), -1)        
        
        z = torch.cat((out,y),dim = 1)
        z = self.cat_fc(z)
        z = self.cat_relu(z)      
        
        z = self.fc2(z)
        z = self.final_fc(z)
 
        return z

def ResNet50():
    return ResNet([3, 4, 6, 3])

#%%
age_predictor = ResNet50()
print(age_predictor)

#%%
# Initialize the model
# age_predictor = AgePredictor(block = Bottleneck,layers = [3, 4, 23, 3],num_classes =1)

#%% inceptionv3

# age_predictor = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)

# age_predictor.Conv2d_1a_3x3.conv=nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

# age_predictor.fc=nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
#                                 nn.Linear(in_features=1024, out_features=1, bias=True))

# print(age_predictor)


#%% resnet34
# age_predictor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# age_predictor.conv1=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# age_predictor.layer4[0].conv2=nn.Sequential( MHSA(n_dims=512, width=16, height=16, heads=4),
#                                             nn.AvgPool2d(2, 2))

# age_predictor.layer4[1].conv2=MHSA(n_dims=512, width=8, height=8, heads=4)

# age_predictor.layer4[2].conv2=MHSA(n_dims=512, width=8, height=8, heads=4)

# age_predictor.fc=nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
#                                 nn.Linear(in_features=1024, out_features=1, bias=True))

# print(age_predictor)


#%%
# age_predictor = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# age_predictor.features[0]=nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# age_predictor.classifier[6]=nn.Sequential(nn.Linear(in_features=4096, out_features=1, bias=True))

# print(age_predictor)

#%%
# Set loss as mean squared error (for continuous output)
# Initialize Stochastic Gradient Descent optimizer and learning rate scheduler



age_predictor = age_predictor.to(device)
criterion = nn.L1Loss()
# optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.RMSprop(age_predictor.parameters(), lr=0.01)
optimizer = optim.Adam(age_predictor.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=25, verbose=1, eps=0.0001, cooldown=5, min_lr=0.0001)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=25, verbose=1, eps=1, cooldown=5, min_lr=0.0001)




# Training Loop
# def train_model(model, criterion, optimizer, scheduler, num_epochs=2000): # model = age_predictor

best_valLoss = 9999
num_epochs = 30000
record=[]
for epoch in range(num_epochs):
   
    age_predictor.train()
    running_loss = 0.0
    val_running_loss = 0.0
    
    for batch in tqdm(train_data_loader):
        # Load batch

        image = batch['image'].to(device)
        gender = batch['gender'].to(device)
        age = batch['bone_age'].to(device)

        
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            # Forward Pass
            # model(image)
            outputs = age_predictor(image, gender)
            # loss = criterion(outputs.logits, age)
            loss = criterion(outputs, age)
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
    age_predictor.eval()
    for val_batch in tqdm(val_data_loader):
        image = val_batch['image'].to(device)
        gender = val_batch['gender'].to(device)
        age = val_batch['bone_age'].to(device)

        
        optimizer.zero_grad()  
        # only forward pass, dont update gradients
        with torch.set_grad_enabled(False):
            outputs = age_predictor(image, gender)
            loss = criterion(outputs, age)
            
        val_running_loss += loss.item() * image.size(0)
    
    val_loss = val_running_loss / len(val_df)
    
    print('Validation Loss {:.6f}'.format(val_loss))
        
        
    if best_valLoss > val_loss :
        best_valLoss = val_loss
        best_model = age_predictor
        total_epochs = scheduler.state_dict()['last_epoch'] + 1 
        print('Save model : epoch-{}-loss-{:.6f}-val_loss-{:.6f} '.format(total_epochs,total_loss,val_loss))
        record.append('Train epoch_loss='+str(total_loss)+'Train epoch_loss=' + str(best_valLoss))
      


        

    # return model
#%%
torch.save(best_model, f"./pretrain_botnet_model_{epoch}.pth")
# vgg16_model = train_model(age_predictor,criterion,optimizer,scheduler,num_epochs=2000)
#%%    

# def denormalize(inputs,age_min,age_max):
#     return inputs * (age_max - age_min) + age_min
        



# result_array = eval_model(age_predictor,test_data_loader)





# test_df['output'] = result_array
# test_df['output'] = np.round(test_df['output'], decimals=2)
# test_df['output'] = test_df['output'].astype(np.float64)
# test_df = test_df.reset_index()

# #%%
# rmse = np.sqrt(mean_squared_error(test_df['bone_age'], test_df['output']))
# print(rmse)
# # 25.259
# #%%

# def display_preds(num):
#     idx = random.sample(range(0, test_df.shape[0]), num)
#     for i in idx:
#         image = cv2.imread(dataset_path + str(test_df['id'][i]) + '.bmp')
#         image = cv2.resize(image,(img_size,img_size))
#         cv2.putText(image,'Actual:' + str(test_df['bone_age'][i]/12.0),(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
#         cv2.putText(image,'Predicted:' + str(test_df['output'][i]/12.0),(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
#         cv2.imshow('Bone Age Prediction',image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        


#%%
# model = torch.load('./model/inceptionv3/3channel/male/loss-13.970724-val_loss-8.197616-best_model.pth')

def mae_months(model, val_data_loader):
    
    
    error=0.0
    for batch_no, val_batch in enumerate(val_data_loader):
        torch.cuda.empty_cache()
        image = val_batch['image'].to(device)
        age = val_batch['bone_age'].to(device)
        
        outputs = model(image)
        for pred, gt in zip(outputs,age):
            error += abs(pred-gt)
        
        
    
    ave_error = error/float(len(val_df))
    
    

    return ave_error

# display_preds(4)
