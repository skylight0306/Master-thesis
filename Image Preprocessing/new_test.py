import os 
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
 
img_size = 512
#%%
# class DownsampleLayer(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(DownsampleLayer, self).__init__()
#         self.Conv_BN_ReLU_2=nn.Sequential(
#             nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#         self.downsample=nn.Sequential(
#             nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )

#     def forward(self,x):
#         """
#         :param x:
#         :return: out输出到深层，out_2输入到下一层，
#         """
#         out=self.Conv_BN_ReLU_2(x)
#         out_2=self.downsample(out)
#         return out,out_2


# class UpSampleLayer(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         # 512-1024-512
#         # 1024-512-256
#         # 512-256-128
#         # 256-128-64
#         super(UpSampleLayer, self).__init__()
#         self.Conv_BN_ReLU_2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
#             nn.BatchNorm2d(out_ch*2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
#             nn.BatchNorm2d(out_ch*2),
#             nn.ReLU()
#         )
#         self.upsample=nn.Sequential(
#             nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )

#     def forward(self,x,out):
#         '''
#         :param x: 输入卷积层
#         :param out:与上采样层进行cat
#         :return:
#         '''
#         x_out=self.Conv_BN_ReLU_2(x)
#         x_out=self.upsample(x_out)
#         cat_out=torch.cat((x_out,out),dim=1)
#         return cat_out


# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
#         #下采样
#         self.d1=DownsampleLayer(3,out_channels[0])#3-64
#         self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
#         self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
#         self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
#         #上采样
#         self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
#         self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
#         self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
#         self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
#         #输出
#         self.o=nn.Sequential(nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm2d(out_channels[0]),
#             nn.ReLU(),
#             nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels[0]),
#             nn.ReLU(),
#             nn.Conv2d(out_channels[0],3,3,1,1)
#         )
#     def forward(self,x):
#         out_1,out1=self.d1(x)
#         out_2,out2=self.d2(out1)
#         out_3,out3=self.d3(out2)
#         out_4,out4=self.d4(out3)
#         out5=self.u1(out4,out_4)
#         out6=self.u2(out5,out_3)
#         out7=self.u3(out6,out_2)
#         out8=self.u4(out7,out_1)
#         out=self.o(out8)
#         return out
#%%


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



def segmentation():
    global imglist

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)
    net = UNet(1,1)
        
    # net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
    net.inc.double_conv[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net.outc=nn.Sequential(nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)),nn.Sigmoid())
    net.to(device=device)
    net.load_state_dict(torch.load("./pretrain_unet_model_0519.pth", map_location=device))
         
    # net = torch.load("./unet.pth")
    # torch.save(net,f'./unet.pth')
    
    model = net
    
    class dataset (torch.utils.data.Dataset):
          def __init__(self,is_train):
             self.img_path='./new_data/'
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
              # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
              # img = clahe.apply(img)
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
        cv2.imwrite("./new_label_data/" + imglist[_j], img_Pred*255.0)
        # cv2.waitKey(1000)
        count += 1
      

if __name__ == '__main__':
    global imglist
    imglist = []
    segmentation()
    

