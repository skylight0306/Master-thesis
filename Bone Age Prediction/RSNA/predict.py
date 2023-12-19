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
import glob
import cv2
import numpy as np

#%%
dataset_path = '../boneage-training-dataset/'

image_filenames = glob.glob(dataset_path+'*.png')

img_size = 512    

for img_name in image_filenames:
   
    label_name = '../label_rsna/' + img_name
    
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
    
    
    seg_clahe_img = label_img*np.array(clahe_img)
    temp = np.array(seg_clahe_img)
    
    thresold_seg_clahe_img = temp
    
    thresold_seg_clahe_img[thresold_seg_clahe_img<165]=0
    
    
    merge = cv2.merge([ thresold_seg_clahe_img, seg_clahe_img])
    merge = cv2.merge([merge, seg_image])
    merge = merge.astype(np.float64)
    merge/=255     

    cv2.imshow('Original Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    cv2.imwrite("D:/boneage/all_rsna/image preprocessing/origin/" + img_name, seg_image)
    
    
    cv2.imshow('CLAHE Image',clahe_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("D:/boneage/all_rsna/image preprocessing/clahe/" + img_name, clahe_img)
       
    cv2.imshow('Seg_CLAHE Image',seg_clahe_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("D:/boneage/all_rsna/image preprocessing/clahe+seg/" + img_name, seg_clahe_img)
    
    
    cv2.imshow('Seg_CLAHE Image',thresold_seg_clahe_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("D:/boneage/all_rsna/image preprocessing/clage+seg+thersold/" + img_name, thresold_seg_clahe_img)
    
    
    cv2.imshow('Final Image',merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    cv2.imwrite("D:/boneage/all_rsna/image preprocessing/merge/" + img_name, merge*255.0)


cv2.imwrite("./new_label_data/" + imglist[_j], img_Pred*255.0)


#%%

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
    