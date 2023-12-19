# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:17:07 2023

@author: doral
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%


seed = 1999
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


setup_seed(seed)


age_df = pd.read_csv('height.csv')

# type(age_df['height'][3])
# age_df.fillna(0)
age_df = age_df.dropna(axis= 'index', how='any')
# len(dir(age_df['height']))

# age_df['mature'] = [216-int(i) for i in age_df['true_age']]
# age_df['height'] = [216-float(i) for i in age_df['height']]
# age_df['state'] = [i-j for i, j in zip(age_df['true_age'], age_df['bone_age'])]
# dataset_size = len(age_df)


print('Total:', len(age_df), 'data')






df_data = pd.DataFrame(data= age_df,
                     columns= ['gender','age','身高', '體重','growth_hormone','父身高','母身高'])


for i in range(len(age_df['身高'])):
    if age_df['身高'][i] < 50:
        print(age_df['身高'][i])

age_df['height_category'] = pd.cut(age_df['身高'], 10)
# iris = load_iris()
# df_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                       columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
# df_data



# X = df_data.drop(labels=['Species'],axis=1).values # 移除Species並取得剩下欄位資料
# y = df_data['Species'].values

data = df_data.drop(labels=['身高'],axis=1).values
label = df_data['身高'].values



from sklearn.model_selection import train_test_split


# 5-fold
A_df, B_df = train_test_split(  age_df, 
                                test_size = 0.2, 
                                random_state=seed,
                                stratify = age_df['height_category'])

A_df, C_df = train_test_split(  A_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = A_df['height_category'])

A_df, D_df = train_test_split(  A_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = A_df['height_category'])

C_df, E_df = train_test_split(  C_df, 
                                test_size = 0.5, 
                                random_state=seed,
                                stratify = C_df['height_category'])


train_df = pd.concat([B_df,C_df,D_df,E_df])
val_df = A_df

'''
QOCA
CASE1 :
train_df = B_df, C_df, D_df, E_df
val_df = A_df
NN

CASE2 :
train_df = A_df, C_df, D_df, E_df
val_df = B_df
NN

CASE3 :
train_df = A_df, B_df, D_df, E_df
val_df = C_df
NN

CASE4 :
train_df = A_df, B_df, C_df, E_df
val_df = D_df
NN

CASE5 :
train_df = A_df, B_df, C_df, D_df
val_df = E_df
NN
'''


#%%


class HeightDataset(Dataset):
    def __init__(self, dataframe, transform=None, train_mode = False):

        self.dataframe = dataframe
        self.transform = transform
        self.train_mode = train_mode

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx): #idx = 0
    

        
        
        gender = np.atleast_1d(self.dataframe.iloc[idx,10])
        age = np.atleast_1d(self.dataframe.iloc[idx,2])
        height = np.atleast_1d(self.dataframe.iloc[idx,5])
        weight = np.atleast_1d(self.dataframe.iloc[idx,6])
        grow_hormone = np.atleast_1d(self.dataframe.iloc[idx,11])
        father_h = np.atleast_1d(self.dataframe.iloc[idx,8])
        mother_h = np.atleast_1d(self.dataframe.iloc[idx,9])
        


        sample = {'gender' : gender, 'age' : age, '身高' : height, '體重' : weight, 'growth_hormone' : grow_hormone, '父身高' : father_h, '母身高' : mother_h}
        
        sample = self.transform(sample)
        
        
        return sample


#%%

class ToTensor(object):
    

    def __call__(self, sample):
        gender, age, height, weight, growth_hormone, father_h, mother_h  = sample['gender'], sample['age'], sample['身高'], sample['體重'], sample['growth_hormone'], sample['父身高'], sample['母身高']

        param = np.zeros(6, dtype=int)
     
        param[0] = gender
        param[1] = age
        param[2] = weight
        param[3] = growth_hormone
        param[4] = father_h
        param[5] = mother_h

        
        return {'param' : torch.from_numpy(param).float(),
                '身高': torch.from_numpy(height).float()}        









data_transform = transforms.Compose([

   ToTensor()
   
   ])     
    
#%%

train_dataset = HeightDataset(dataframe = train_df, transform = data_transform, train_mode = True)
val_dataset = HeightDataset(dataframe = val_df, transform = data_transform, train_mode = False)




    
train_data_loader = DataLoader(train_dataset,batch_size=128,shuffle=False)
val_data_loader = DataLoader(val_dataset,batch_size=128,shuffle=False)


class ANNModel(nn.Module):
    
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(128, 1)
 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        
        x = self.fc4(x)

        return x


#%%

height_predictor = ANNModel()

height_predictor = height_predictor.to(device)
criterion = nn.L1Loss()
# optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.RMSprop(age_predictor.parameters(), lr=0.0001)
# lambda1 = lambda epoch: (epoch / 4000) if epoch < 4000 else 0.5 * (math.cos((epoch-4000)/(100*1000-4000)*math.pi) + 1)
optimizer = optim.Adam(height_predictor.parameters(), lr=0.02)

# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=25, verbose=1, eps=0.0001, cooldown=5, min_lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8,  mode='min',patience=15, verbose=1, min_lr=0.00001)


best_valLoss = 9999
num_epochs = 30000
index = 0
x=[]
y=[]

record=[]
for epoch in range(num_epochs):
   
    height_predictor.train()
    running_loss = 0.0
    val_running_loss = 0.0
    
    for batch in tqdm(train_data_loader):
 
        # Load batch
        param = batch['param'].to(device)
        height = batch['身高'].to(device)
    
        
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            # Forward Pass
            # model(image)
            # outputs = age_predictor(image)
            outputs = height_predictor(param)
            loss = criterion(outputs, height)
            # loss = criterion(outputs.logits, age)
            # Backprop
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            # nn.utils.clip_grad_norm_(age_predictor.parameters(),  max_norm=2.0)
            optimizer.step()
            
        # Calculate Loss
        running_loss += loss.item() * height.size(0)
        
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
    height_predictor.eval()
    for val_batch in tqdm(val_data_loader):

        param = val_batch['param'].to(device)
        height = val_batch['身高'].to(device)
        
        
        optimizer.zero_grad()  
        # only forward pass, dont update gradients
        with torch.set_grad_enabled(False):
            # outputs = age_predictor(image)
            outputs = height_predictor(param)
            loss = criterion(outputs, height)
            
        
        val_running_loss += loss.item() * height.size(0)
        # print( 'image size: ' + str(image.size(0)) + '\nloss : ' + str(loss.item()) + '\ntotal loss : ' + str(val_running_loss) )
    val_loss = val_running_loss / len(val_df)
    
    print('Validation Loss {:.6f}'.format(val_loss))
    print('Best val_loss {:.6f} '.format(best_valLoss))
        
    if best_valLoss > val_loss :
        best_valLoss = val_loss
        best_model = height_predictor
        total_epochs = scheduler.state_dict()['last_epoch'] + 1 
        print('Save model : \nepoch-{}-loss-{:.6f}-val_loss-{:.6f} '.format(total_epochs,total_loss,val_loss))
        # print('Best val_loss {:.6f} '.format(best_valLoss))
        record.append('Train epoch_loss='+str(total_loss)+'Test epoch_loss=' + str(best_valLoss))
      
    scheduler.step(val_loss)
#%%
torch.save(best_model, "./5-fold_A_loss_6.06_size500.pth")


# model = torch.load('./5-fold/5-fold_A_inceptionv3_loss_6._size500.pth')

def mae_months(model, val_data_loader): #model = best_model   model = age_predictor
    
    arr=[]
    error=0.0
    count = 0
    for val_batch in tqdm(val_data_loader):
       
        torch.cuda.empty_cache()
        img_name = val_batch['name']
        image = val_batch['image'].to(device)
        gender = val_batch['gender'].to(device)
        age = val_batch['bone_age'].to(device)
        count += age.shape[0]
        # cv2.imshow('My Image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()    
        # outputs = model(image, gender)

        
        # only forward pass, dont update gradients
        with torch.no_grad():
            # outputs = age_predictor(image)
            outputs = model(image, gender)
            err = abs(outputs-age)
            for i in range(len(err)):
                if err[i] > 20:
                    print(img_name[i]+ " mae = " + str(float(err[i])))
                    arr.append(img_name[i]+ ", mae = " + str(float(err[i])))
                
                
                
            for pred, gt in zip(outputs,age):
                # print(abs(pred-gt))
                error += abs(pred-gt)
            # print(error)
    
    
    
    ave_error = error/float(len(val_df))
    
    

    return float(ave_error)
