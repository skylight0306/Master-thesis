import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import tqdm
import glob
img_size=512


'''
from shutil import copy
arr = os.listdir('./label/')

file_path = 'D:/boneage/preprocessing/boneage-training-dataset/'
to_path = './age/'
for filename in arr:
    print(filename)
    from_path = os.path.join(file_path, filename)
    copy(from_path,to_path)
'''
    
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

# # class dual-branch(nn.Module):

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
#         #Downsample
#         self.d1=DownsampleLayer(3,out_channels[0])#3-64
#         self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
#         self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
#         self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
#         #UpSample
#         self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
#         self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
#         self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
#         self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
#         #output
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


class dataset (torch.utils.data.Dataset):
    
      def __init__(self,is_train=True,transform = None):
         # self.img_path='./bonetrain/'
         # self.label_path='./bonelabel/'
         self.img_path='./train_data/'
         self.label_path='./label_data/'
         self.transform = transform
         img=[]
         label_img=[]
         #total 15, 8:2 split
         if is_train:
             # self.img_dir=os.listdir(self.img_path)[:300] #19514153.2020.01.22.
             # self.label_dir=os.listdir(self.label_path)[:300]
             # self.img_dir=glob.glob(self.img_path+'*.bmp')[:300]
             
             img = [glob.glob(self.img_path + e) for e in ['*.bmp', '*.png']]
             label_img = [glob.glob(self.label_path + e) for e in ['*.bmp', '*.png']]
             self.img_dir = (img[0] + img[1])[:3000]
             self.label_dir= (label_img[0] + label_img[1])[:3000]
         else:
             # self.img_dir=os.listdir(self.img_path)[300:]
             # self.label_dir=os.listdir(self.label_path)[300:]
             img = [glob.glob(self.img_path + e) for e in ['*.bmp', '*.png']]
             label_img = [glob.glob(self.label_path + e) for e in ['*.bmp', '*.png']]             
             
             self.img_dir = (img[0] + img[1])[3000:]
             self.label_dir = (label_img[0] + label_img[1])[3000:]
         # self.img_dir.remove('desktop.ini')
         # self.label_dir.remove('desktop.ini')
         self.is_train=is_train
      def __len__(self):
          return len(self.img_dir)

                              
      def __getitem__(self,index):
          # img = cv2.imread(self.img_path + self.img_dir[index],cv2.IMREAD_GRAYSCALE) # img = cv2.imread(img_path + img_dir[0])
          # print(self.img_dir[index])
          img = cv2.imread(self.img_dir[index],0) # img = cv2.imread(img_path + '14170.bmp',0) #img = cv2.imread(img_path + img_dir[0],0)
          img = cv2.resize(img, (img_size, img_size))
          # bn_img = ((img-img.min()) / img.max() - img.min())
          # print('ok')
            # cv2.imshow('img',label)
            
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
          # label = cv2.imread(self.label_path + self.label_dir[index],cv2.IMREAD_GRAYSCALE) #label = cv2.imread(label_path + label_dir[0],0)
          # print(self.label_dir[index])
          label = cv2.imread(self.label_dir[index],0) #label = cv2.imread(label_path + label_dir[0],0)
          
          label = cv2.resize(label, (img_size, img_size))
          # print('ok')
          # tensor=transforms.Compose([transforms.ToTensor(),
          #                            transforms.Resize((img_size, img_size)),
          #                            ]) 
             
          # img_tensor=transforms.Compose([transforms.ToTensor(),
          #                                  transforms.Resize((img_size, img_size)),
          #                                  # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          #                                  transforms.Grayscale(num_output_channels=1)
          #                                  ])    
          
          
          merge = cv2.merge([img, label])
        
        
          trans_toPIL = transforms.ToPILImage()
          # trans_Rotate = transforms.RandomRotation(30)
          trans_horizon = transforms.RandomHorizontalFlip(p=0.3)
          trans_Affine = transforms.RandomAffine(degrees=(-25,25), scale=(0.9, 1.2), translate = (0, 0.1))
          # trans_Crop = transforms.CenterCrop(img_size*0.8)
          trans_Color = transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.7, 1.3))
   
          if self.transform:
              merge = np.asarray(merge)
              merge = trans_toPIL(merge)
              merge = trans_horizon(merge)
              # merge = trans_Rotate(merge)
              merge = trans_Affine(merge)
              merge = trans_Color(merge)
              merge = np.asarray(merge)
              merge = merge.astype(np.float64)
            
          # if self.transform:
          #   augmentation = self.transform(image=img, mask=label)
          #   img = augmentation["image"]
          #   label = augmentation["mask"]
          #   label = torch.unsqueeze(label,0)
          
          # img=img_tensor(img)
  
          # label=tensor(label)
          merge = merge.astype(np.float64)
          merge/=255.0

                    
          # cv2.imshow('img',merge[:,:,1])
          # cv2.waitKey(0)
          # cv2.destroyAllWindows()
          
          img = merge[:,:,0]
          label = merge[:,:,1]
          
          
          
          
          return img,label

# class ToTensor(object):
    

#     def __call__(self, sample):
#         img_name, image, gender, bone_age = sample['name'], sample['image'], sample['gender'], sample['bone_age']


#         image = cv2.resize(image,(img_size,img_size))
    
#         temp = np.zeros((3,img_size, img_size))
#         temp[0, :, :] = image[:, :, 0]
#         temp[1, :, :] = image[:, :, 1]
#         temp[2, :, :] = image[:, :, 2]
#         # temp[3, :, :] = image[:, :, 3]
#         # temp[4, :, :] = image[:, :, 4]
#         # temp[5, :, :] = image[:, :, 5]        
#         image = temp
#         # image = np.expand_dims(image,axis = 0)
        
# #       we need to convert  cuda.longtensors to cuda.floatTensor data type
#         return {'name' : img_name,
#                 'image': torch.from_numpy(image).float(),
#                 'gender': torch.from_numpy(gender).float(),
#                 'bone_age':torch.from_numpy(bone_age).float()}        


def dice_loss(y_pred,y_true,  epsilon=1e-6):  #(y_pred=True_mask,   y_true=Pred_mask)
    batch=y_pred.shape[0]
    #print(y_pred.shape)
    y_pred=y_pred.view(batch,img_size*img_size)
    y_true=y_true.view(batch,img_size*img_size)
    numerator = 2. * torch.sum(y_pred * y_true,1)
    denominator = torch.sum(y_pred + y_true,1)
    return 1 - torch.mean(numerator / (denominator + epsilon))




net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)

net.inc.double_conv[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

net.outc=nn.Sequential(nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)), nn.Sigmoid())

# net= net.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = net.to( dtype=torch.float)
net = net.to(device)
net = net.type(torch.cuda.FloatTensor)
#net = UNet().cuda()
#print(net)
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset(True, transform = True), batch_size=4, shuffle=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-9)
val_loader = torch.utils.data.DataLoader(dataset(False,transform = False), batch_size=2, shuffle=True)

record = []
best_valLoss = 9999
for epoch in range(30000):
    net.train()
    epoch_loss = 0
    loss=0
    
    print("-------", epoch, "-------")
    # for _i,batch in enumerate(train_loader):
    for batch in tqdm(train_loader):
    
        imgs,True_mask=batch
        imgs=imgs.type(torch.cuda.FloatTensor).to(device)
        True_mask = True_mask.type(torch.cuda.FloatTensor).to(device)
        
        imgs = torch.unsqueeze(imgs, 1)
        True_mask = torch.unsqueeze(True_mask, 1)
        with torch.set_grad_enabled(True):
            Pred_mask = net(imgs)
    
            loss = dice_loss(True_mask,Pred_mask)
            epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('-------- Train epoch_loss=',epoch_loss/300.0)
    
    val_loss=0
    net.eval()
    # for _j,val_batch in enumerate(val_loader):
    for val_batch in tqdm(val_loader):
        
        val_imgs,val_True_mask=val_batch


        val_imgs=val_imgs.type(torch.cuda.FloatTensor).to(device)
        val_True_mask = val_True_mask.type(torch.cuda.FloatTensor).to(device)
        
        val_imgs = torch.unsqueeze(val_imgs, 1)
        val_True_mask = torch.unsqueeze(val_True_mask, 1)        
        with torch.set_grad_enabled(False):
            val_Pred_mask = net(val_imgs)
            Loss = dice_loss(val_True_mask,val_Pred_mask)
        val_loss += Loss.item()

    print('--------val eopch_loss=',val_loss/27.0)
    
    valloss = val_loss/(27.0)
    epoch += 1
    
     
    if best_valLoss > valloss :
        best_valLoss = valloss
        best_model = net
        print("record")
        record.append('Train epoch_loss='+str(epoch_loss/300.0)+'Train epoch_loss=' + str(val_loss/27.0))
        
        
        # record.append('Train epoch_loss='+str(epoch_loss/(_i+1))+'Train epoch_loss=' + str(val_loss/(_j+1)))
        # torch.save(net.state_dict(), f"./pretrain_unet_model_{epoch}.pth")
        
        
    # if epoch %10000 == 0 :
    #     record.append('Train epoch_loss='+str(epoch_loss/(_i+1))+'Train epoch_loss=' + str(val_loss/(_j+1)))
    #     torch.save(net.state_dict(), f"./pretrain_unet_model_{epoch}.pth")

# torch.save(net.state_dict(), f"./pretrain_unet_model.pth")

torch.save(best_model.state_dict(), f"./pretrain_unet_model_0519.pth")
