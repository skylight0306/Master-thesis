import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
 
img_size = 512
class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2


class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1)
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

def segmentation():
    global imglist

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)
    #net = Unet()
        
    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
    net.inc.double_conv[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net.outc=nn.Sequential(nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)),nn.Sigmoid())
    net.to(device=device)
    net.load_state_dict(torch.load("./pretrain_unet_model_30000.pth", map_location=device))
         
    

    
    model = net
    
    class dataset (torch.utils.data.Dataset):
          def __init__(self,is_train):
             self.img_path='./data/'
            #  self.label_path='./Resized_Label_Images/'
             self.is_train=is_train
             if is_train:
                 self.img_dir=os.listdir(self.img_path)[:1]
                #  self.label_dir=os.listdir(self.label_path)[:1236]
             else:
                 self.img_dir=os.listdir(self.img_path)

          def __len__(self):
              return len(self.img_dir)
          def __getitem__(self,index):
              img = cv2.imread(self.img_path+self.img_dir[index],cv2.IMREAD_GRAYSCALE)
              clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
              img = clahe.apply(img)
              imglist.append(self.img_dir[index])
              # label = cv2.imread(self.label_path+self.label_dir[index])

              tensor=transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((img_size, img_size))])
              img = tensor(img)
              # label=tensor(label)
              return img
    val_loader = torch.utils.data.DataLoader(dataset(False), batch_size=1, shuffle=True)
    
    count = 0
    for _j,val_batch in enumerate(val_loader):
        # img,img_label=val_batch
        # img,img_label=img.cuda(),img_label.cuda()
        # img_Pred = torch.sigmoid_(model(img))
        # img_Pred=np.transpose(img_Pred.cpu().detach().numpy()[0],(1,2,0))
        # cv2.imshow('Image',np.transpose(img.cpu().detach().numpy()[0],(1,2,0)))
        # cv2.imshow('Original Mask',np.transpose(img_label.cpu().detach().numpy()[0],(1,2,0)))
        # cv2.imshow('Predicted Mask',img_Pred)
        # cv2.waitKey(2000)

        img=val_batch
        img=img.cuda()
        img_Pred = model(img)
        img_Pred=np.transpose(img_Pred.cpu().detach().numpy()[0],(1,2,0))
        img_Pred[img_Pred>0.5]=1
        img_Pred[img_Pred<=0.5]=0
        # cv2.imshow('Image',np.transpose(img.cpu().detach().numpy()[0],(1,2,0)))
        # cv2.imshow('Original Mask',np.transpose(img_label.cpu().detach().numpy()[0],(1,2,0)))
        # cv2.imshow('Predicted Mask',img_Pred)

        # cv2.imwrite("./test/merge23result/" + , np.transpose(img.cpu().detach().numpy()[0],(1,2,0)) *255.0)
        cv2.imwrite("./label_data/" + imglist[_j], img_Pred*255.0)
        # cv2.waitKey(1000)
        count += 1
      
      
from PIL import Image
import glob
dataset_path = './data/'

# file_source = './boneage-training-dataset/'
# file_destination = './data/'
 
# get_files = os.listdir(dataset_path)
 
# for g in get_files:
#     os.replace(file_source + g, file_destination + g)


def transpose():
    
    image_filenames = glob.glob(dataset_path+'*.png')
    
    
    for i in range(len(image_filenames)): #i=0
        tp=Image.open(image_filenames[i])
        tp.transpose(Image.FLIP_LEFT_RIGHT).save('./rsna_data/'+image_filenames[i][6:])#左右翻轉



if __name__ == '__main__':
    global imglist
    imglist = []
    segmentation()
    

