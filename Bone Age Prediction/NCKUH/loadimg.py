# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 23:37:41 2023

@author: q56104034
"""
import numpy as np
import os
import nibabel as nib
from nibabel.testing import data_path
import glob
import imageio
import cv2

data_dir = './qoca2/'

df_name = glob.glob(data_dir+'*.nii')

for i in df_name: #i = df_name[0]  i = df_name[694]  
    
    img = nib.load(i)
    print(img.header)
    img_fdata = img.get_fdata()
    fname = i.replace('.nii','')
    (x,y,z) = img.shape
    
    for j in range(z): #j = 0
        image = cv2.resize(img_fdata,(1024,1024))
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        image_norm = cv2.rotate(image_norm, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(image_norm, 1) 
        img = np.uint8(img*255)
        
        # cv2.imshow('My Image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        
        cv2.imwrite('./dd/BONEAGE_' + fname[43:48] + '.png', img)
        # img.imwrite(,img_fdata)

#BONEAGE_00001


print(n1_img.header)
print(n1_img.dataobj)



    image = cv2.imread('./label_dd/BONEAGE_02937.png',0)
    image = cv2.resize(image, (512, 512))
    cv2.imwrite('./label_dd/BONEAGE_02937.png' , image)
