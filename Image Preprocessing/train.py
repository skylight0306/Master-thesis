import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms
import cv2
from PIL import Image
 
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False)
net.outc=nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
class dataset (torch.utils.data.Dataset):
    
      def __init__(self,is_train=True):
         self.img_path='./test/merge23/'
         self.label_path='./test/merge23label/'
         #total 15, 8:2 split
         if is_train:
             self.img_dir=os.listdir(self.img_path)[:12]
             self.label_dir=os.listdir(self.label_path)[:12]
         else:
             self.img_dir=os.listdir(self.img_path)[12:]
             self.label_dir=os.listdir(self.label_path)[12:]
             
         self.is_train=is_train
      def __len__(self):
          return len(self.img_dir)

                              
      def __getitem__(self,index):
          img = cv2.imread(self.img_path + self.img_dir[index])
        #   cv2.imshow("123", img)
        #   while True:
        #     if cv2.getWindowProperty("123", 0) == -1:
        #         break
        #     cv2.waitKey(1)
        #   cv2.destroyWindow("123")
          label = cv2.imread(self.label_path + self.label_dir[index])
        #   cv2.imshow("123", label)
        #   while True:
        #     if cv2.getWindowProperty("123", 0) == -1:
        #         break
        #     cv2.waitKey(1)
        #   cv2.destroyWindow("123")
        #   ret, label = cv2.threshold(label, 240, 255, cv2.THRESH_BINARY)
          tensor=transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((256, 256)),
                                     ])                         
          img=tensor(img)
          label=tensor(label)
          return img,label
          
def dice_loss(y_pred,y_true,  epsilon=1e-6): 
    batch=y_pred.shape[0]
    y_pred=y_pred.view(batch,3,256*256)
    y_true=y_true.view(batch,3,256*256)
    numerator = 2. * torch.sum(torch.sum(y_pred * y_true,2),1)
    denominator = torch.sum(torch.sum(y_pred + y_true,2),1)
    
    return 1 - torch.mean(numerator / (denominator + epsilon))

net.cuda()
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset(True), batch_size=8, shuffle=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-9)
val_loader = torch.utils.data.DataLoader(dataset(False), batch_size=1, shuffle=True)

for epoch in range(60):
    net.train()
    epoch_loss = 0
    loss=0
    print("-------", epoch, "-------")
    for _i,batch in enumerate(train_loader):
        net.train()
        imgs,True_mask=batch
        imgs,True_mask=imgs.cuda(),True_mask.cuda()
        Pred_mask = torch.sigmoid_(net(imgs))

        loss = dice_loss(True_mask,Pred_mask)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('-------- Train epoch_loss=',epoch_loss/(_i+1))

    val_loss=0
    net.eval()
    for _j,val_batch in enumerate(val_loader):
        val_imgs,val_True_mask=val_batch
        val_imgs,val_True_mask=val_imgs.cuda(),val_True_mask.cuda()
        val_Pred_mask = torch.sigmoid_(net(val_imgs))
        Loss = dice_loss(val_True_mask,val_Pred_mask)
        val_loss += Loss.item()

    print('--------val eopch_loss=',val_loss/(_j+1))
    epoch += 1
    
torch.save(net, "unet_model.pth")