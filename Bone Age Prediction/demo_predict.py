# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:17:15 2023

@author: sky
"""

import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms
import torch.nn.functional as F

import cv2
import numpy as np

#%%
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


#%%



class ToTensor(object):
    

    def __call__(self, image):


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
        
        return torch.from_numpy(image).float()    



def pred_boneage( net, image_dir, label_image_dir, male):
      

    male = int(male)
    
    image = cv2.imread(img_name,0)
    image = cv2.resize(image, (img_size, img_size))
    
    
    
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
    
    seg_clahe_img = temp
    
    seg_clahe_img[seg_clahe_img<150]=0
    
    
    merge = cv2.merge([seg_clahe_img, clahe_img])
    merge = cv2.merge([merge, seg_image])
    merge = merge.astype(np.float64)
    merge/=255     
    
    cv2.imshow('Original Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    cv2.imshow('CLAHE Image',clahe_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

       
    cv2.imshow('Seg_CLAHE Image',seg_clahe_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    cv2.imshow('Final Image',merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
 
    data_transform = transforms.Compose([
       ToTensor()
       ])     
    
    male = torch.from_numpy(np.atleast_1d(male)).float()
    
    # [C, H, W]
    img_tensor = data_transform(merge)
    
    input_tensor = torch.unsqueeze(img_tensor, dim=0) #增加一个batch维度
    male = torch.unsqueeze(male, dim=0)

    input_tensor=input_tensor.to(device)
    male = male.to(device)
    
    
    pred_boneage = net(input_tensor,male)
    
    
    return pred_boneage
    
    
    
    
net = torch.load('./5-fold_D_inceptionv4_loss_5.36_size500.pth')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


img_size = 512    
image_dir = './data/'
label_image_dir = './label_data/'

img_name = image_dir + 'BONEAGE_00051.png' 
label_name = label_image_dir +'BONEAGE_00051.png'    
male = 1

age = pred_boneage(net, image_dir, label_image_dir, male)

print("predicted bone age : " + str(age))
    
    
    # cv2.imshow('My Image',visualization)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()








'5-fold_E_inceptionv4_loss_4.33_size500'