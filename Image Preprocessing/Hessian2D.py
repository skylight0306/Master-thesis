# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 22:05:28 2022

@author: q56104034
"""

import numpy as np
import cv2
import math
from scipy import signal


def Hessian2D(I,Sigma):
    #求输入图像的Hessian矩阵
    #实际上中间包含了高斯滤波，因为二阶导数对噪声非常敏感
    #输入：    
    #   I : 单通道double类型图像
    #   Sigma : 高斯滤波卷积核的尺度  
    #    
    # 输出 : Dxx, Dxy, Dyy: 图像的二阶导数 
    if Sigma<1:
        print("error: Sigma<1")
        return -1
    I=np.array(I,dtype=float)
    Sigma=np.array(Sigma,dtype=float)
    S_round=np.round(3*Sigma)

    [X,Y]= np.mgrid[-S_round:S_round+1,-S_round:S_round+1]

    #构建卷积核：高斯函数的二阶导数
    DGaussxx = 1/(2*math.pi*pow(Sigma,4)) * (X**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
    DGaussxy = 1/(2*math.pi*pow(Sigma,6)) * (X*Y) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))   
    DGaussyy = 1/(2*math.pi*pow(Sigma,4)) * (Y**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
  
    Dxx = signal.convolve2d(I,DGaussxx,boundary='fill',mode='same',fillvalue=0)
    Dxy = signal.convolve2d(I,DGaussxy,boundary='fill',mode='same',fillvalue=0)
    Dyy = signal.convolve2d(I,DGaussyy,boundary='fill',mode='same',fillvalue=0)

    return Dxx,Dxy,Dyy


def eig2image(Dxx,Dxy,Dyy):
    # This function eig2image calculates the eigen values from the
    # hessian matrix, sorted by abs value. And gives the direction
    # of the ridge (eigenvector smallest eigenvalue) .
    # input:Dxx,Dxy,Dyy图像的二阶导数
    # output:Lambda1,Lambda2,Ix,Iy
    #Compute the eigenvectors of J, v1 and v2
    Dxx=np.array(Dxx,dtype=float)
    Dyy=np.array(Dyy,dtype=float)
    Dxy=np.array(Dxy,dtype=float)
    if (len(Dxx.shape)!=2):
        print("len(Dxx.shape)!=2,Dxx不是二维数组！")
        return 0

    tmp = np.sqrt( (Dxx - Dyy)**2 + 4*Dxy**2)

    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp

    mag = np.sqrt(v2x**2 + v2y**2)
    i=np.array(mag!=0)

    v2x[i==True] = v2x[i==True]/mag[i==True]
    v2y[i==True] = v2y[i==True]/mag[i==True]

    v1x = -v2y 
    v1y = v2x

    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    check=abs(mu1)>abs(mu2)
            
    Lambda1=mu1.copy()
    Lambda1[check==True] = mu2[check==True]
    Lambda2=mu2
    Lambda2[check==True] = mu1[check==True]
    
    Ix=v1x
    Ix[check==True] = v2x[check==True]
    Iy=v1y
    Iy[check==True] = v2y[check==True]
    
    return Lambda1,Lambda2,Ix,Iy


def FrangiFilter2D(I):
    I=np.array(I,dtype=float)
    defaultoptions = {'FrangiScaleRange':(1,10), 'FrangiScaleRatio':2, 'FrangiBetaOne':0.5, 'FrangiBetaTwo':15, 'verbose':True,'BlackWhite':True};  
    options=defaultoptions


    sigmas=np.arange(options['FrangiScaleRange'][0],options['FrangiScaleRange'][1],options['FrangiScaleRatio'])
    sigmas.sort()#升序

    beta  = 2*pow(options['FrangiBetaOne'],2)  
    c     = 2*pow(options['FrangiBetaTwo'],2)

    #存储滤波后的图像
    shape=(I.shape[0],I.shape[1],len(sigmas))
    ALLfiltered=np.zeros(shape) 
    ALLangles  =np.zeros(shape) 

    #Frangi filter for all sigmas 
    Rb=0
    S2=0
    for i in range(len(sigmas)):
        #Show progress
        if(options['verbose']):
            print('Current Frangi Filter Sigma: ',sigmas[i])
        
        #Make 2D hessian
        [Dxx,Dxy,Dyy] = Hessian2D(I,sigmas[i])

        #Correct for scale 
        Dxx = pow(sigmas[i],2)*Dxx  
        Dxy = pow(sigmas[i],2)*Dxy  
        Dyy = pow(sigmas[i],2)*Dyy
         
        #Calculate (abs sorted) eigenvalues and vectors  
        [Lambda2,Lambda1,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)  

        #Compute the direction of the minor eigenvector  
        angles = np.arctan2(Ix,Iy)  

        #Compute some similarity measures  
        Lambda1[Lambda1==0] = np.spacing(1)

        Rb = (Lambda2/Lambda1)**2  
        S2 = Lambda1**2 + Lambda2**2
        
        #Compute the output image
        Ifiltered = np.exp(-Rb/beta) * (np.ones(I.shape)-np.exp(-S2/c))
         
        #see pp. 45  
        if(options['BlackWhite']): 
            Ifiltered[Lambda1<0]=0
        else:
            Ifiltered[Lambda1>0]=0
        
        #store the results in 3D matrices  
        ALLfiltered[:,:,i] = Ifiltered 
        ALLangles[:,:,i] = angles

        # Return for every pixel the value of the scale(sigma) with the maximum   
        # output pixel value  
        if len(sigmas) > 1:
            outIm=ALLfiltered.max(2)
        else:
            outIm = (outIm.transpose()).reshape(I.shape)
            
    return outIm


def clahe():
    
    
    dataset_path = 'data/'
    label_path = 'label_data/'    
    image_dir = 'data/'
    label_image_dir = 'label_data/'

    img_size = 512
    image_filenames = glob.glob(dataset_path+'*.bmp')
    label_image_filenames = glob.glob(label_path+'*.bmp')    
    
    for filename in image_filenames : #filename = image_filenames[0]
        image = cv2.imread(filename)
        image = cv2.resize(image,(img_size,img_size))
    
    
        # image = image.astype(np.float64)
        # image/=255 
        label_name = 'label_'+filename
        label_img = cv2.imread('label_'+filename,0)        

        label_img = cv2.resize(label_img, (img_size, img_size))
        label_img[label_img<255] = 0
        label_img[label_img==255] = 1
        label_img_float64 = label_img.astype(np.float64)
    

        seg_image = label_img*image[:,:,0]
        seg_image = label_img*image[:,:,1]
        seg_image = label_img*image[:,:,2]
    
        
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
        clahe_img = clahe.apply(seg_image)
        clahe_img = label_img*clahe_img
        clahe_img = clahe_img.astype(np.float64)
        clahe_img/=255              
        clahe_img[clahe_img<0.7]=0.0      
        
        
        # cv2.imshow('My Image',clahe_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        cv2.imwrite("./clahe_data/" + filename[5:], clahe_img*255.0)



if __name__ == "__main__":
    
    
    dataset_path = 'clahe_data/'
    label_path = 'label_data/'
    
    image_filenames = glob.glob(dataset_path+'*.bmp')
    label_image_filenames = glob.glob(label_path+'*.bmp')



filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
image_own = bilateral_filter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_own.png", image_own)
       

    for filename in image_filenames : #filename = image_filenames[0] filename = './clahe_data/' + '13486809.2020.04.15.bmp' 
        image = cv2.imread(filename,0)
        image = cv2.resize(image,(512,512))


        # blood = cv2.normalize(image.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point
        # outIm=FrangiFilter2D(blood)
        # img=outIm*(5000)
        
        cv2.imwrite("./Hessian_clahe_data/" + filename[11:], img*255.0)

        # cv2.imshow('img',image)
        cv2.imshow('img',image_own)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
        
        
    # img = cv2.resize(img, (512, 512))
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(image)
    clahe_img = label_img*clahe_img
    clahe_img = clahe_img.astype(np.float64)
    clahe_img/=255              
    clahe_img[clahe_img<0.7]=0.0      

    cv2.imshow('img',clahe_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()    
    
    
    
    # label_imagename='./label_data/13486809.2020.04.15.bmp'
    label_img = cv2.imread('label_data'+filename[12:],0)
    label_img = cv2.resize(label_img, (512, 512))
    label_img[label_img<255] = 0
    label_img = label_img.astype(np.float64)
    label_img/=255        



    # img = label_img*img

    
    
    # cv2.imshow('img',img)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()