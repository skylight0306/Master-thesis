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
import math
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import glob
import random
#from age_predictor_model import Bottleneck, AgePredictor
from tqdm import tqdm
import scipy.io as io
import torch.nn.functional as F
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statistics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.chdir('/home/kneehit/Data Science/Bone Age Kaggle/PyTorch')


#%%
# Image and CSV paths
dataset_path = 'new_data/'
label_path = 'new_label_data/'
# test_dataset_path = 'bone_data/test/' 
# val_dataset_path = 'bone_data/validation/'
csv_path = 'new_bone.csv'



seed = 1999
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


setup_seed(seed)



#%% 
# Sample random images from dataset (since loading all images will take time) 
# and calculate mean and standard deviation for normalization 
# k = 100
img_size = 512


image_filenames = glob.glob(dataset_path+'*.png')
label_image_filenames = glob.glob(label_path+'*.png')
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

# means1 = []
# stds1 = []
# means2 = []
# stds2 = []
# means3 = []
# stds3 = []


print(len(image_filenames))
        
# for filename in image_filenames : #filename = image_filenames[0]


#     image = cv2.imread(filename,0)
#     image = cv2.resize(image,(img_size,img_size))
#     # image = image.astype(np.float64)
#     # image/=255                
    
    
#     # print( filename )
#     label_name = 'label_'+filename
#     label_img = cv2.imread(label_name,0)
#     label_img = cv2.resize(label_img,(img_size,img_size))    
#     label_img[label_img<255] = 0
#     label_img[label_img==255] = 1
#     label_img_float64 = label_img.astype(np.float64)         
    

#     image = image*label_img
        
#     seg_image = label_img*image

#     temp = np.zeros((img_size, img_size))
#     clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
#     clahe_img = clahe.apply(seg_image)
#     clahe_img = label_img*clahe_img
#     temp = clahe_img
#     temp = temp.astype(np.float64)
#     temp/=255              
#     seg_clahe_img = temp
    
#     seg_clahe_img[seg_clahe_img<0.7]=0.0           

#     seg_image = seg_image.astype(np.float64)
#     seg_image/=255              
#     clahe_img = clahe_img.astype(np.float64)
#     clahe_img/=255  
        

#     # cv2.imshow('My Image',seg_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()    

#     merge = cv2.merge([seg_clahe_img, clahe_img])
#     merge = cv2.merge([merge, seg_image])
#     cv2.imwrite("./merge_data/" + filename[10:],merge*255.0)
#     mean1,std1 = cv2.meanStdDev(seg_clahe_img)
#     mean2,std2 = cv2.meanStdDev(clahe_img)
#     mean3,std3 = cv2.meanStdDev(seg_image)
#     #    mean /= 255
#     #    std /= 255
    
#     means1.append(mean1)
#     stds1.append(std1)
#     means2.append(mean2)
#     stds2.append(std2)
#     means3.append(mean3)
#     stds3.append(std3)
    
    
    
# avg_mean1 = np.mean(means1) 
# avg_std1 = np.mean(stds1)
# avg_mean2 = np.mean(means2) 
# avg_std2 = np.mean(stds2)
# avg_mean3 = np.mean(means3) 
# avg_std3 = np.mean(stds3)


#%%



# age_df.to_csv("bone.csv", encoding='utf_8_sig', index = False)


age_df = pd.read_csv('new_bone.csv')
age_df['path'] = age_df['醫令名稱'].map(lambda x: os.path.join('./new_data/', 
                                                          '{}.png'.format(x)))
age_df['exists'] = age_df['path'].map(os.path.exists)

age_df['exists'] = age_df['exists'].map(lambda x: True if x else np.nan )

age_df['gender'] = age_df['性別'].map(lambda x: 1.0 if x=='M' else 0.0)

age_df = age_df.dropna(axis= 'index', how='any')
age_df.reset_index(drop=True, inplace = True)



# age_df.to_csv("new_bone.csv", encoding='utf_8_sig', index = False)

# age_df['img'] = age_df['path'].map(lambda x: cv2.imread(x,0))

# for img in age_df['img']: #i=5
#     cv2.imshow('My Image',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()    
    

# id_arr = np.array(age_df['id'])
# gender_arr = np.array(age_df['male'])
# boneage_arr = np.array(age_df['boneage'])
# path_arr =  np.array(age_df['path'])
# img_arr = np.array(age_df['img'])





# io.savemat('loss-2.886766-val_loss-7.971161-test_model.mat',{'id':id_arr, 'gender':gender_arr, 'boneage':boneage_arr, 'path':path_arr, 'img':img_arr})




# data = io.loadmat('testmat.mat')



# data['img'][0][8]

# cv2.imshow('My Image',data['img'][0][8])
# cv2.waitKey(0)
# cv2.destroyAllWindows()    

dataset_size = len(age_df)

# for i in range (age_df.shape[0]): #i=12
#     if age_df['exists'][i] == '':
#         print(age_df['path'][i])
        
age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)

print('Total:', age_df['exists'].sum(), 'images')

# bones_df = pd.read_csv(csv_path)
age_df.iloc[:,9:10] = age_df.iloc[:,9:10].astype(np.float64)
# age_df.iloc[:,1:2] = age_df.iloc[:,1:2].astype(np.float64)

#%%

from sklearn.model_selection import train_test_split


# 5-fold
A_df, B_df = train_test_split(  age_df, 
                                test_size = 0.2, 
                                random_state=seed,
                                stratify = age_df['boneage_category'])

A_df, C_df = train_test_split(  A_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = A_df['boneage_category'])

A_df, D_df = train_test_split(  A_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = A_df['boneage_category'])

C_df, E_df = train_test_split(  C_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = C_df['boneage_category'])




train_df = pd.concat([A_df,B_df,D_df,E_df])
val_df = C_df

'''
QOCA
CASE1 :
train_df = B_df, C_df, D_df, E_df
val_df = A_df
InceptionV4 5.72 ± 5.04   train 5.58

CASE2 :
train_df = A_df, C_df, D_df, E_df
val_df = B_df
InceptionV4 5.49 5.16 train 4.77

CASE3 :
train_df = A_df, B_df, D_df, E_df
val_df = C_df
InceptionV4 

CASE4 :
train_df = A_df, B_df, C_df, E_df
val_df = D_df
InceptionV4 

CASE5 :
train_df = A_df, B_df, C_df, D_df
val_df = E_df
InceptionV4 
'''

# test_df, val_df = train_test_split(test_df, 
#                                    random_state=seed,
#                                    test_size = 0.5)


# random_seed=200                 


# train_df = age_df.sample(frac=0.8, random_state=random_seed)

# val_df = age_df[~age_df.index.isin(train_df.index)]

 
# test_df = val_df.sample(frac=0.5, random_state=random_seed)

# val_df = val_df[~val_df.index.isin(test_df.index)]





# print('train data : ', str(len(train_df)), '\nvalidation data : ', str(len(val_df)), '\ntest data : ' , str(len(test_df)) )


age_max = np.max(age_df['boneage'])
age_min = np.min(age_df['boneage'])

# *(age_max-age_min)+age_min


# (bone_age - self.age_min)/ (self.age_max - self.age_min)

#%% BoneDataset
# import PIL.Image as Image
# BonesDataset(dataframe = train_df,image_dir=dataset_path,transform = data_transform)

class BonesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, train_mode = False):

        self.dataframe = dataframe
        self.image_dir = 'new_data/'
        self.label_image_dir = 'new_label_data/'
        # self.hessian_dir = 'Hessian/'
        self.transform = transform
        self.train_mode = train_mode

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx): #idx = 0
    
        # img_name = image_dir + '8457.png' 
        # label_name = label_image_dir +'8457.png' 
        

        # img_name = image_dir + '15187.png' 
        # label_name = label_image_dir +'15187.png' 

    
        img_name = self.image_dir + str(self.dataframe.iloc[idx,7]) + '.png'
        label_name = self.label_image_dir + str(self.dataframe.iloc[idx,7]) + '.png'
        # hessian_name = self.hessian_dir + str(self.dataframe.iloc[idx,0]) + '.bmp'
    
        image = cv2.imread(img_name,0)
        image = cv2.resize(image, (img_size, img_size))
        # image = image.astype(np.float64)
        # image/=255 
        

        image = cv2.imread(img_name,0)
        image = cv2.resize(image, (img_size, img_size))
        # image = image.astype(np.float64)
        # image/=255 
        trans_toPIL = transforms.ToPILImage()
        trans_Contrast = transforms.ColorJitter(brightness=(0.6, 1.4),contrast=(0.7,1.3))
        
        if self.train_mode:
            image = np.asarray(image)
            image = trans_toPIL(image)
            image = trans_Contrast(image)
            image = np.asarray(image)
        
        
        label_img = cv2.imread(label_name,0)
        label_img = cv2.resize(label_img, (img_size, img_size))
        label_img[label_img<255] = 0
        label_img[label_img==255] = 1        


 
        #gray
        
        
        seg_image = label_img*image

        
        
        temp = np.zeros((img_size, img_size))
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
        clahe_img = clahe.apply(seg_image)
        clahe_img = label_img*clahe_img
        temp = np.array(clahe_img)
        
        # temp = temp.astype(np.float64)
        # temp/=255      

        
        seg_clahe_img = temp
        
        # seg_clahe_img[seg_clahe_img<0.7]=0.0           
        seg_clahe_img[seg_clahe_img<180]=0
        # seg_clahe_img[seg_clahe_img>=180]=255
        
        
        # seg_image = seg_image.astype(np.float64)
        # seg_image/=255              
        # clahe_img = clahe_img.astype(np.float64)
        # clahe_img/=255  

        
                
        # cv2.imshow('My Image',seg_clahe_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        
        

        merge = cv2.merge([seg_clahe_img, clahe_img])
        
        
        merge = cv2.merge([merge, seg_image])
        
        
        gender = np.atleast_1d(self.dataframe.iloc[idx,12])
        bone_age = np.atleast_1d(self.dataframe.iloc[idx,9])
        
        # merge = merge.astype('float32')

        # sample = {'image': merge, 'gender': gender, 'bone_age':bone_age}

        # trans_toPIL = transforms.ToPILImage()
        # trans_Rotate = transforms.RandomRotation(30)
        trans_horizon = transforms.RandomHorizontalFlip(p=0.3)
        trans_Affine = transforms.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate = (0, 0.2))
        # trans_Crop = transforms.CenterCrop(img_size*0.8)
        # trans_Color = transforms.ColorJitter(brightness=(0.6, 1.4))
   
        if self.train_mode:
            merge = np.asarray(merge)
            merge = trans_toPIL(merge)
            merge = trans_horizon(merge)
            # merge = trans_Rotate(merge)
            merge = trans_Affine(merge)
            # merge = trans_Color(merge)
            merge = np.asarray(merge)
            merge = merge.astype(np.float64)
            
            
        merge = merge.astype(np.float64)
        merge/=255     
        
        
        
        # cv2.imshow('My Image',merge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        
        sample = {'name' : img_name, 'image': merge, 'gender': gender, 'bone_age':bone_age}
        
        sample = self.transform(sample)
        
        
        return sample

#%% 
# Custom Transforms for Image and numerical data
        
# Resize and Convert numpy array to tensor
class ToTensor(object):
    

    def __call__(self, sample):
        img_name, image, gender, bone_age = sample['name'], sample['image'], sample['gender'], sample['bone_age']


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
        return {'name' : img_name,
                'image': torch.from_numpy(image).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age':torch.from_numpy(bone_age).float()}        

# Normalize images and bone age

from sklearn import preprocessing
from matplotlib import pyplot as plt
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# d = scaler.fit_transform(image[:,:,0])


# class Normalize(object):
    
#     def __init__(self,img_mean,img_std,age_min,age_max):
#         self.mean = img_mean
#         self.std = img_std
        
#         self.age_min = age_min
#         self.age_max = age_max
        
    
    
#     def __call__(self,sample): #image = merge
#         image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']
        
#         # image -= self.mean
#         # image /= self.std
        
#         # image[:,:,0] -= avg_mean
#         # image /= avg_std
        
        
#         for i in range(image.shape[2]): #i=0
#             if i == 0:
#                 image[:,:,0] -= avg_mean1
#                 image[:,:,0] /= avg_std1
#             elif i == 1:
#                 image[:,:,1] -= avg_mean2
#                 image[:,:,1] /= avg_std2
#             elif i == 2:
#                 image[:,:,2] -= avg_mean3
#                 image[:,:,2] /= avg_std3
                
                
                
#         image[:,:,0] = scaler.fit_transform(image[:,:,0])
#         image[:,:,1] = scaler.fit_transform(image[:,:,1])
#         image[:,:,2] = scaler.fit_transform(image[:,:,2])
#         # arr = []
#         # for i in range(image[:,:,0].shape[0]): #i=0
#         #     for j in range(image[:,:,0].shape[1]): #j=0
#         #         if not image[:,:,0][i][j]  == 0.0:
#         #             arr.append(image[:,:,0][i][j])
        

        
#         # cv2.imshow('My Image',image[:,:,])
#         # from matplotlib import pyplot as plt
#         # plt.hist(arr)
#         # plt.show()
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
        
        
        
#         # bone_age /=12
#         # bone_age = (bone_age - self.age_min)/ (self.age_max - self.age_min)
#         # img_tensor=transforms.Compose([transforms.ToTensor(),
#         #                                 transforms.Resize((img_size, img_size)),
#         #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         #                                 transforms.Grayscale(num_output_channels=1)
#         #                                 ])            
#         # img=img_tensor(image)
        
#         return {'image': image,
#                 'gender': gender,
#                 'bone_age':bone_age} 
        

data_transform = transforms.Compose([
   # Normalize(avg_mean1,avg_std1,age_min,age_max),
   # transforms.RandomCrop(224,padding=16),
   # transforms.ToPILImage(),
   # ToTensor(),
   # transforms.RandomRotation(30),
   # transforms.CenterCrop(img_size*0.8),
    # transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),
   ToTensor()
   
   ])     
    
# data_transform = transforms.Compose([
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensor()
#     ])     

#%%

# dataset = BonesDataset(dataframe = bones_df,image_dir=dataset_path,transform = data_transform)

train_dataset = BonesDataset(dataframe = train_df, image_dir=dataset_path, transform = data_transform, train_mode = True)
val_dataset = BonesDataset(dataframe = val_df, image_dir = dataset_path, transform = data_transform, train_mode = False)
# aug_val_dataset = BonesDataset(dataframe = val_df, image_dir = dataset_path, transform = data_transform, train_mode = True)

# test_dataset = BonesDataset(dataframe = test_df, image_dir=dataset_path, transform = data_transform, train_mode = False)


    

# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [236, 59])
# val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [29, 30])


# train_set, val_set = torch.utils.data.random_split(dataset, [236, 10000])

# Sanity Check
# print(train_dataset[0])

    
train_data_loader = DataLoader(train_dataset,batch_size=32,shuffle=False)
val_data_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)
# aug_val_data_loader = DataLoader(val_dataset,batch_size=64,shuffle=False)
# test_data_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)





#%% MHSA

   
    
# class MHSA(nn.Module):
#     def __init__(self, n_dims, width=14, height=14, heads=4):
#         super(MHSA, self).__init__()
#         self.heads = heads

#         self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

#         self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
#         self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         n_batch, C, width, height = x.size()
#         q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
#         k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
#         v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

#         content_content = torch.matmul(q, k.permute(0, 1, 3, 2))

#         content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
#         content_position = torch.matmul(q, content_position)

#         energy = content_content + content_position
#         attention = self.softmax(energy)

#         out = torch.matmul(attention, v)
#         out = out.view(n_batch, C, width, height)

#         return out

# class BasicConv2d(nn.Module):

#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return F.relu(x, inplace=True)

# def Conv1(in_planes, places, stride=2):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
#         nn.BatchNorm2d(places),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )


#%% resnet50 + gender


# def Conv1(in_planes, places, stride=2):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
#         nn.BatchNorm2d(places),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )


# class Bottleneck(nn.Module):
#     def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
#         super(Bottleneck,self).__init__()
#         self.expansion = expansion
#         self.downsampling = downsampling

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
#             nn.BatchNorm2d(places),
#             # nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(places),
#             # nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(places*self.expansion),
#             # nn.ReLU(inplace=True),
#         )

#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(places*self.expansion)
#             )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.bottleneck(x)

#         if self.downsampling:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)
#         return out

# class MHSABottleneck(nn.Module):
#     def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
#         super(MHSABottleneck,self).__init__()
#         self.expansion = expansion
#         self.downsampling = downsampling
        
        
#         if self.downsampling:
#             self.bottleneck = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
#                 nn.BatchNorm2d(places),
#                 # nn.ReLU(inplace=True),
#                 # nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
#                 MHSA(n_dims=512, width=MHSA_size1, height=MHSA_size1, heads=4),
#                 nn.AvgPool2d(2, 2),
#                 nn.BatchNorm2d(places),
#                 # nn.ReLU(inplace=True),
#                 nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(places*self.expansion),
#                 # nn.ReLU(inplace=True),
#             )
#         else:
#             self.bottleneck = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
#                 nn.BatchNorm2d(places),
#                 # nn.ReLU(inplace=True),
#                 # nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
#                 MHSA(n_dims=512, width=MHSA_size2, height=MHSA_size2, heads=4),
#                 nn.BatchNorm2d(places),
#                 # nn.ReLU(inplace=True),
#                 nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(places*self.expansion),
#                 # nn.ReLU(inplace=True),
#             )
            
#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(places*self.expansion)
#             )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.bottleneck(x)

#         if self.downsampling:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)
#         return out



# class ResNet(nn.Module):
#     def __init__(self,blocks, num_classes=1000, expansion = 4):
#         super(ResNet,self).__init__()
#         self.expansion = expansion

#         self.conv1 = Conv1(in_planes = 3, places= 64)

#         self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
#         self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
#         self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
#         # self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
#         self.layer4 = self.make_MHSAlayer(in_places=1024,places=512, block=blocks[3], stride=2)


#         # self.layer4_MHSA1 = nn.Sequential(
#         #     nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1, bias=False),
#         #     nn.BatchNorm2d(512),
#         #     MHSA(n_dims=512, width=MHSA_size1, height=MHSA_size1, heads=4),
#         #     nn.AvgPool2d(kernel_size=2, stride=2),
#         #     nn.BatchNorm2d(512),
#         #     nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=1,stride=1, bias=False),
#         #     nn.BatchNorm2d(2048)
#         # )
        
        

        
#         # self.downsample = nn.Sequential(
#         #     nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, bias=False),
#         #     nn.BatchNorm2d(2048)
#         # )
            
            
#         # self.MHSA_relu = nn.ReLU(inplace=True)
        
#         # self.layer4_MHSA2 = nn.Sequential(
#         #     nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=1,stride=1, bias=False),
#         #     nn.BatchNorm2d(512),
#         #     MHSA(n_dims=512, width=MHSA_size2, height=MHSA_size2, heads=4),
#         #     nn.BatchNorm2d(512),
#         #     nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=1,stride=1, bias=False),
#         #     nn.BatchNorm2d(2048),
#         #     nn.ReLU()
#         # )
        
#         # self.MHSA_relu2 = nn.ReLU(inplace=True)
        
#         # self.layer4_MHSA3 = nn.Sequential(
#         #     nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=1,stride=1, bias=False),
#         #     nn.BatchNorm2d(512),
#         #     MHSA(n_dims=512, width=MHSA_size2, height=MHSA_size2, heads=4),
#         #     nn.BatchNorm2d(512),
#         #     nn.Conv2d(in_channels=512,out_channels=2048,kernel_size=1,stride=1, bias=False),
#         #     nn.BatchNorm2d(2048),
#         #     nn.ReLU()
#         # )    

#         # self.MHSA_relu3 = nn.ReLU(inplace=True)

#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.fc = nn.Linear(2048,968)

#         self.gen_fc_1 = nn.Linear(1,32)
#         # self.gen_relu = nn.ReLU(inplace=True)
        
#         # Feature Concatenation Layer
#         self.cat_fc = nn.Linear(32+968,1000)
#         # self.cat_relu = nn.ReLU(inplace=True)
        
        
#         # self.fc2 = nn.Linear(1000, 1000)
#         # Final Fully Connected Layer
#         self.final_fc = nn.Linear(1000,1)
       

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def make_layer(self, in_places, places, block, stride):
#         layers = []
#         layers.append(Bottleneck(in_places, places,stride, downsampling =True))
#         for i in range(1, block):
#             layers.append(Bottleneck(places*self.expansion, places))

#         return nn.Sequential(*layers)

#     def make_MHSAlayer(self, in_places, places, block, stride):
#         layers = []
#         layers.append(MHSABottleneck(in_places, places,stride, downsampling =True))
#         for i in range(1, block):
#             layers.append(MHSABottleneck(places*self.expansion, places))

#         return nn.Sequential(*layers)
    
    
#     def forward(self, x, y):
#         x = self.conv1(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         #layer4
#         # residual = x
#         # out1 = self.layer4_MHSA1(x)
#         # residual = self.downsample(x)
        
#         # out1 += residual
#         # out1 = self.MHSA_relu(out1)

#         # residual1 = out1
       
#         # out2 = self.layer4_MHSA2(out1)
#         # out2 += residual1
#         # out2 = self.MHSA_relu2(out2)
        
        
#         # residual2 = out2
        
#         # out = self.layer4_MHSA3(out2)
#         # out += residual2
#         # out = self.MHSA_relu2(out)
        
        

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = x.view(x.size(0), -1)
        
        
    
#         y = self.gen_fc_1(y)
#         # y = self.gen_relu(y)
#         y = y.view(y.size(0), -1)        
        
#         z = torch.cat((x,y),dim = 1)
#         z = self.cat_fc(z)
#         # z = self.cat_relu(z)      
        
#         # z = self.fc2(z)
#         z = self.final_fc(z)
 
#         return z

# def ResNet50():
#     return ResNet([3, 4, 6, 3])

#%% inceptionv3 + gender
# class BasicConv2d(nn.Module):

#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return F.relu(x, inplace=True)

# def inception_v3(pretrained=False, **kwargs):
#     r"""Inception v3 model architecture from
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         if 'transform_input' not in kwargs:
#             kwargs['transform_input'] = True
#         model = Inception3(**kwargs)
#         model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
#         return model

#     return Inception3(**kwargs)


# class Inception3(nn.Module):

#     def __init__(self, num_classes=1, aux_logits=True, transform_input=False):
#         super(Inception3, self).__init__()
#         self.aux_logits = aux_logits
#         self.transform_input = transform_input
#         self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
#         self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
#         self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
#         self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
#         self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
#         self.Mixed_5b = InceptionA(192, pool_features=32)
#         self.Mixed_5c = InceptionA(256, pool_features=64)
#         self.Mixed_5d = InceptionA(288, pool_features=64)
#         self.Mixed_6a = InceptionB(288)
#         self.Mixed_6b = InceptionC(768, channels_7x7=128)
#         self.Mixed_6c = InceptionC(768, channels_7x7=160)
#         self.Mixed_6d = InceptionC(768, channels_7x7=160)
#         self.Mixed_6e = InceptionC(768, channels_7x7=192)
#         # if aux_logits:
#         self.AuxLogits = InceptionAux(768, num_classes)
#         self.Mixed_7a = InceptionD(768)
#         self.Mixed_7b = InceptionE(1280)
#         self.Mixed_7c = InceptionE(2048)
#         # self.fc = nn.Linear(2048, num_classes)
#         self.fc = nn.Linear(2048, 968)
#         self.gen_fc_1 = nn.Linear(1,32)
#         self.gen_relu = nn.ReLU()


#         # Feature Concatenation Layer
#         self.cat_fc = nn.Linear(32+968,1000)
#         self.cat_relu = nn.ReLU()
        
#         self.fc2 = nn.Linear(1000, 1000)
#         self.final_fc = nn.Linear(1000,num_classes)
        
        
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         #         import scipy.stats as stats
#         #         stddev = m.stddev if hasattr(m, 'stddev') else 0.1
#         #         X = stats.truncnorm(-2, 2, scale=stddev)
#         #         values = torch.Tensor(X.rvs(m.weight.data.numel()))
#         #         m.weight.data.copy_(values)
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         m.weight.data.fill_(1)
#         #         m.bias.data.zero_()

#     def forward(self, x, y): #y = gender tensor
#         if self.transform_input: # 1
#             x = x.clone()
#             x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
#             x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
#             x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
#         # 299 x 299 x 3
#         x = self.Conv2d_1a_3x3(x)
#         # 149 x 149 x 32
#         x = self.Conv2d_2a_3x3(x)
#         # 147 x 147 x 32
#         x = self.Conv2d_2b_3x3(x)
#         # 147 x 147 x 64
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 73 x 73 x 64
#         x = self.Conv2d_3b_1x1(x)
#         # 73 x 73 x 80
#         x = self.Conv2d_4a_3x3(x)
#         # 71 x 71 x 192
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 35 x 35 x 192
#         x = self.Mixed_5b(x)
#         # 35 x 35 x 256
#         x = self.Mixed_5c(x)
#         # 35 x 35 x 288
#         x = self.Mixed_5d(x)
#         # 35 x 35 x 288
#         x = self.Mixed_6a(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6b(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6c(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6d(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6e(x)
#         # 17 x 17 x 768
#         # if self.training and self.aux_logits:
#         aux = self.AuxLogits(x)
#         # 17 x 17 x 768
#         x = self.Mixed_7a(x)
#         # 8 x 8 x 1280
#         x = self.Mixed_7b(x)
#         # 8 x 8 x 2048
#         x = self.Mixed_7c(x)
#         # 8 x 8 x 2048
#         x = F.avg_pool2d(x, kernel_size=8)
#         # 1 x 1 x 2048
#         x = F.dropout(x, training=self.training)
#         # 1 x 1 x 2048
#         x = x.view(x.size(0), -1)
#         # 2048
#         x = self.fc(x)
#         # 1000 (num_classes)
#         x = x.view(x.size(0), -1)
        
#         y = self.gen_fc_1(y)
#         y = self.gen_relu(y)
#         y = y.view(y.size(0), -1)
        
        
#         z = torch.cat((x,y), dim = 1)
#         z = self.cat_fc(z)
#         z = self.cat_relu(z)

# # =============================================================================
# #       Final FC Layer
# # =============================================================================
#         z = self.fc2(z)
#         z = self.final_fc(z)
        
#         return z


# class InceptionA(nn.Module):

#     def __init__(self, in_channels, pool_features):
#         super(InceptionA, self).__init__()
#         self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

#         self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
#         self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

#         self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
#         self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

#         self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch5x5 = self.branch5x5_1(x)
#         branch5x5 = self.branch5x5_2(branch5x5)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionB(nn.Module):

#     def __init__(self, in_channels):
#         super(InceptionB, self).__init__()
#         self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

#         self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
#         self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

#     def forward(self, x):
#         branch3x3 = self.branch3x3(x)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

#         branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

#         outputs = [branch3x3, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionC(nn.Module):

#     def __init__(self, in_channels, channels_7x7):
#         super(InceptionC, self).__init__()
#         self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

#         c7 = channels_7x7
#         self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
#         self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

#         self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
#         self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

#         self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch7x7 = self.branch7x7_1(x)
#         branch7x7 = self.branch7x7_2(branch7x7)
#         branch7x7 = self.branch7x7_3(branch7x7)

#         branch7x7dbl = self.branch7x7dbl_1(x)
#         branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionD(nn.Module):

#     def __init__(self, in_channels):
#         super(InceptionD, self).__init__()
#         self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
#         self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

#         self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
#         self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

#     def forward(self, x):
#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = self.branch3x3_2(branch3x3)

#         branch7x7x3 = self.branch7x7x3_1(x)
#         branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
#         branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
#         branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

#         branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
#         outputs = [branch3x3, branch7x7x3, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionE(nn.Module):

#     def __init__(self, in_channels):
#         super(InceptionE, self).__init__()
#         self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

#         self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
#         self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
#         self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

#         self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
#         self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
#         self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
#         self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
        
#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = [
#             self.branch3x3_2a(branch3x3),
#             self.branch3x3_2b(branch3x3),
#         ]
#         branch3x3 = torch.cat(branch3x3, 1)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = [
#             self.branch3x3dbl_3a(branch3x3dbl),
#             self.branch3x3dbl_3b(branch3x3dbl),
#         ]
#         branch3x3dbl = torch.cat(branch3x3dbl, 1)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionAux(nn.Module):

#     def __init__(self, in_channels, num_classes):
#         super(InceptionAux, self).__init__()
#         self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
#         self.conv1 = BasicConv2d(128, 768, kernel_size=5)
#         self.conv1.stddev = 0.01
#         self.fc = nn.Linear(768, num_classes)
#         self.fc.stddev = 0.001

#     def forward(self, x):
#         # 17 x 17 x 768
#         x = F.avg_pool2d(x, kernel_size=5, stride=3)
#         # 5 x 5 x 768
#         x = self.conv0(x)
#         # 5 x 5 x 128
#         x = self.conv1(x)
#         # 1 x 1 x 768
#         # x = x.view(x.size(0), -1)
#         x = x.view(-1,768)
#         # 768
#         x = self.fc(x)
#         # 1000
#         return x





# age_predictor = inception_v3(pretrained=False)

#%% inceptionV4
class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                  dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
class Stem_v4_Res2(nn.Module):
    """
    stem block for Inception-v4 and Inception-RestNet-v2
    """

    def __init__(self):
        super(Stem_v4_Res2, self).__init__()
        self.step1 = nn.Sequential(
            BN_Conv2d(3, 32, 3, 2, 0, bias=False),
            BN_Conv2d(32, 32, 3, 1, 0, bias=False),
            BN_Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = BN_Conv2d(64, 96, 3, 2, 0, bias=False)
        self.step3_1 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step3_2 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = BN_Conv2d(192, 192, 3, 2, 0, bias=False)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        # print(tmp1.shape)
        # print(tmp2.shape)
        out = torch.cat((tmp1, tmp2), 1)
        return out

class Inception_A(nn.Module):
    """
    Inception-A block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n3, b4_n1, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3, 3, 1, 1, bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n3, 3, 1, 1, bias=False),
            BN_Conv2d(b4_n3, b4_n3, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)

class Reduction_A(nn.Module):
    """
    Reduction-A block for Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2 nets
    """

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch2 = BN_Conv2d(in_channels, n, 3, 2, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, k, 1, 1, 0, bias=False),
            BN_Conv2d(k, l, 3, 1, 1, bias=False),
            BN_Conv2d(l, m, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)
class Inception_B(nn.Module):
    """
    Inception-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x7, b3_n7x1, b4_n1, b4_n1x7_1,
                  b4_n7x1_1, b4_n1x7_2, b4_n7x1_2):
        super(Inception_B, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x7_1, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_1, b4_n7x1_1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b4_n7x1_1, b4_n1x7_2, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_2, b4_n7x1_2, (7, 1), (1, 1), (3, 0), bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)

class Reduction_B_v4(nn.Module):
    """
    Reduction-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False)
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)

class Inception_C(nn.Module):
    """
    Inception-C block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x3_3x1, b4_n1,
                  b4_n1x3, b4_n3x1, b4_n1x3_3x1):
        super(Inception_C, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3_1 = BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False)
        self.branch3_1x3 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch3_3x1 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)
        self.branch4_1 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            BN_Conv2d(b4_n1x3, b4_n3x1, (3, 1), (1, 1), (1, 0), bias=False)
        )
        self.branch4_1x3 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch4_3x1 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        tmp = self.branch3_1(x)
        out3_1 = self.branch3_1x3(tmp)
        out3_2 = self.branch3_3x1(tmp)
        tmp = self.branch4_1(x)
        out4_1 = self.branch4_1x3(tmp)
        out4_2 = self.branch4_3x1(tmp)
        return torch.cat((out1, out2, out3_1, out3_2, out4_1, out4_2), 1)

class Inception(nn.Module):
    """
    implementation of Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2
    """

    def __init__(self, version, num_classes):
        super(Inception, self).__init__()
        self.version = version
        self.stem = Stem_v4_Res2()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()

        # self.fc = nn.Linear(1536, num_classes)

            
        self.fc = nn.Linear(1536, 968)
        self.gen_fc_1 = nn.Linear(1,32)
        self.gen_relu = nn.ReLU()


        # Feature Concatenation Layer
        self.cat_fc = nn.Linear(32+968,1000)
        self.cat_relu = nn.ReLU()
        
        self.fc2 = nn.Linear(1000, 1000)
        self.final_fc = nn.Linear(1000,num_classes)
        
                    
        

    def __make_inception_A(self):
        layers = []
        if self.version == "v4":
            for _ in range(4):
                layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))

        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == "v4":
            return Reduction_A(384, 192, 224, 256, 384) # 1024
        elif self.version == "res1":
            return Reduction_A(256, 192, 192, 256, 384) # 896
        else:
            return Reduction_A(384, 256, 256, 384, 384) # 1152

    def __make_inception_B(self):
        layers = []
        if self.version == "v4":
            for _ in range(7):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                          192, 192, 224, 224, 256))   # 1024

        return nn.Sequential(*layers)

    def __make_reduction_B(self):
        if self.version == "v4":
            return Reduction_B_v4(1024, 192, 192, 256, 256, 320, 320)  # 1536


    def __make_inception_C(self):
        layers = []
        if self.version == "v4":
            for _ in range(3):
                layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)
        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        # out = self.fc(out)
        
        out = out.view(out.size(0), -1)
        # 2048
        # print(out.shape)
        out = self.fc(out)
        # 1000 (num_classes)
        out = out.view(out.size(0), -1)
        
        y = self.gen_fc_1(y)
        y = self.gen_relu(y)
        y = y.view(y.size(0), -1)
        
        
        z = torch.cat((out,y), dim = 1)
        z = self.cat_fc(z)
        z = self.cat_relu(z)

# =============================================================================
#       Final FC Layer
# =============================================================================
        z = self.fc2(z)
        z = self.final_fc(z)
        
        return z 
        
        
        
        
        # return F.softmax(out)



def inception_v4(classes=1):
    return Inception("v4", classes)



age_predictor = inception_v4()

#%% resnet50 + gender + MHSA

# age_predictor = ResNet50()
# print(age_predictor)


#%%
# Initialize the model
# age_predictor = AgePredictor(block = Bottleneck,layers = [3, 4, 23, 3],num_classes =1)

#%% inceptionv3

# age_predictor = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)

# age_predictor.Conv2d_1a_3x3.conv=nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

# age_predictor.fc=nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
#                                 nn.Linear(in_features=1024, out_features=1, bias=True))	

# print(age_predictor)


#%% resnet50
# age_predictor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# age_predictor.conv1=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# age_predictor.layer4[0].conv2=nn.Sequential( MHSA(n_dims=512, width=32, height=32, heads=4),
#                                             nn.AvgPool2d(2, 2))

# age_predictor.layer4[1].conv2=MHSA(n_dims=512, width=16, height=16, heads=4)

# age_predictor.layer4[2].conv2=MHSA(n_dims=512, width=16, height=16, heads=4)

# age_predictor.fc=nn.Sequential(nn.Linear(in_features=2048, out_features=1000, bias=True),
#                                 nn.Linear(in_features=1000, out_features=1000, bias=True),
#                                 nn.Linear(in_features=1000, out_features=1, bias=True))

# print(age_predictor)


#%% vgg16
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
# optimizer = optim.RMSprop(age_predictor.parameters(), lr=0.0001)
# lambda1 = lambda epoch: (epoch / 4000) if epoch < 4000 else 0.5 * (math.cos((epoch-4000)/(100*1000-4000)*math.pi) + 1)
optimizer = optim.Adam(age_predictor.parameters(), lr=0.0005)

# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=25, verbose=1, eps=0.0001, cooldown=5, min_lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8,  mode='min',patience=15, verbose=1, cooldown=5, min_lr=0.000001)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min' , factor=0.8, patience=25, verbose=1, eps=1e-08, cooldown=5, min_lr=0.00001)



# Training Loop
# def train_model(model, criterion, optimizer, scheduler, num_epochs=2000): # model = age_predictor

best_epoch = -1
best_valLoss = 9999
best_std = 9999
num_epochs = 30000
index = 0
x=[]
y=[]
err=[]
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
            # outputs = age_predictor(image)
            outputs = age_predictor(image, gender)
            loss = criterion(outputs, age)
            # loss = criterion(outputs.logits, age)
            # Backprop
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            # nn.utils.clip_grad_norm_(age_predictor.parameters(),  max_norm=2.0)
            optimizer.step()
            
        # Calculate Loss
        running_loss += loss.item() * image.size(0)
        
        #lr
        x.append(index)
        y.append(optimizer.param_groups[0]['lr'])
        index += 1
        # scheduler.step()
        
        # if (batch_no + 1) % 25 == 0: print('Epoch {} Batch {}/2000 '.format(epoch+1,batch_no+1))
        
    total_loss = running_loss / len(train_df)
    
    print('\n Epoch {}'.format(epoch+1))
    print('-' * 10)    
    print('Loss: {:.6f}'.format(total_loss))
    
    
# Eval on validation set
    age_predictor.eval()
    std = torch.zeros(0).cuda()
    err = []
    for val_batch in tqdm(val_data_loader):
        image = val_batch['image'].to(device)
        gender = val_batch['gender'].to(device)
        age = val_batch['bone_age'].to(device)
        

        optimizer.zero_grad()  
        # only forward pass, dont update gradients
        with torch.set_grad_enabled(False):
            # outputs = age_predictor(image)
            outputs = age_predictor(image, gender)
            loss = criterion(outputs, age)
            
            err = abs(outputs-age)
            std = torch.cat((std, err), 0)

            
            
        val_running_loss += loss.item() * image.size(0)
        # print( 'image size: ' + str(image.size(0)) + '\nloss : ' + str(loss.item()) + '\ntotal loss : ' + str(val_running_loss) )
    val_loss = val_running_loss / len(val_df)
    
    npstd = std.cpu().detach().numpy()
    total_std = np.std(npstd)
    
    
    print('Validation Loss {:.6f}'.format(val_loss))
    print('Validation std  {:.6f} '.format(total_std))
    print('best Epoch '+ str(best_epoch))
    print('Best val_loss {:.6f} '.format(best_valLoss))
    print('Best std  {:.6f} '.format(best_std))
        
    if best_valLoss > val_loss :
        best_valLoss = val_loss
        best_std = total_std
        best_epoch = epoch+1
        best_model = age_predictor
        total_epochs = scheduler.state_dict()['last_epoch'] + 1 
        print('Save model : \nepoch-{}-loss-{:.6f}-val_loss-{:.6f} '.format(total_epochs,total_loss,val_loss))
        # print('Best val_loss {:.6f} '.format(best_valLoss))
        record.append('Train epoch_loss='+str(total_loss)+'Test epoch_loss=' + str(best_valLoss)+'Test std=' + str(best_std))
      
    scheduler.step(val_loss)
    # scheduler.step()
        

    # return model
#%%
torch.save(best_model, f"./5-fold/new/5-fold_B_inceptionv4_loss_5.49_size500.pth")
# torch.save(age_predictor, f"./5-fold_B_inceptionv4_loss_6.67_size500.pth")

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
model = torch.load('./5-fold_B_inceptionv4_loss_5.49_size500.pth')

# 20142, 14965
def mae_months(model, val_data_loader): #model = best_model   model = age_predictor 
    
    arr=[]
    error=0.0


    df = pd.DataFrame(columns=['name','boneage'])
    
    # df.append(data, ignore_index=True) 

    for val_batch in tqdm(train_data_loader):
       
        torch.cuda.empty_cache()
        img_name = val_batch['name']
        image = val_batch['image'].to(device)
        gender = val_batch['gender'].to(device)
        age = val_batch['bone_age'].to(device)

        # cv2.imshow('My Image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()    
        # outputs = model(image, gender)

        
        # only forward pass, dont update gradients
        with torch.no_grad():
            # outputs = age_predictor(image)
            outputs = model(image, gender)
            err = abs(outputs-age)

            
            # a = err.cpu().detach().numpy()
            # np.std(a)
            # loss = criterion(outputs, age)
            for i in range(len(err)):

                df = df.append({'name':img_name[i],
                            'boneage':str(float(outputs[i])) }, ignore_index=True) 

                    # arr.append(img_name[i]+ ", mae = " + str(float(err[i])))
                
                
                
            for pred, gt in zip(outputs,age):
                # print(abs(pred-gt))
                error += abs(pred-gt)
            # print(error)
    
    # arr[251:]
    
    ave_error = error/float(len(train_df))
    
    df.to_csv('listC.csv',encoding='utf_8_sig', index = False)


    return float(ave_error)

# display_preds(4)

# for i in range(len(arr)): #i=0
#     os.remove("new_"+arr[i])



# df = df.sort_values(by='name')
# df.to_csv('sort_pred_boneage.csv',encoding='utf_8_sig', index = False)


'''
df = pd.DataFrame(columns=['name','boneage_err','count'])

lista = pd.read_csv('listA.csv')
listb = pd.read_csv('listB.csv')
listc = pd.read_csv('listC.csv')

total = pd.concat([lista,listb,listc])


total = total.dropna(axis= 'index', how='any')
total.reset_index(drop=True, inplace = True)




for i in range(len(total)): #i=0
    exist = False
    num = -1
    for j in range(len(df)): #j=0
        if str(total['name'][i]) == str(df['name'][j]):
            exist = True
            num = j
            break
    
    if exist:
        df['boneage_err'][num] = str((float(df['boneage_err'][num]) + float(total['boneage_err'][i]) ) / 2.0)
        df['count'][num] +=1
        
        
    else:
        df = df.append({'name':total['name'][i],
                    'boneage_err':total['boneage_err'][i],
                    'count':float(1.0)}, ignore_index=True)
        print(total['name'][i])


df = df.sort_values(['count','boneage_err'],ascending=False)

df.to_csv('sort.csv',encoding='utf_8_sig', index = False)

'''
    
