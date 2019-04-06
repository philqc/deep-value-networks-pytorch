# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:40:19 2019

@author: Gabriel Hsu
"""

from __future__ import print_function, division

import os
import argparse
import time
import random

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchsummary import summary
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from auxiliary_functions import *
#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")

__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"

#%%Dataset
class WeizmannHorseDataset(Dataset):
    """Weizmann Horse Dataset"""
    
    def __init__(self, img_dir, mask_dir, transform = None):
        """
        Args:
            img_dir(string): Path to the image file (training image)
            mask_dir(string): Path to the mask file (segmentation result)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_names = os.listdir(img_dir)
        self.mask_names = os.listdir(mask_dir)

        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = io.imread(img_name)
        mask = io.imread(mask_name)
        
        if self.transform:
                      
            image = self.transform(image)
            
            # create a channel for mask so as to transform
            mask = self.transform(np.expand_dims(mask, axis=2))
            
#            #Random crop
#            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(24, 24))
#            image = TF.crop(image, i, j, h, w)                
#            mask = TF.crop(mask, i, j, h, w)
                
#            # Random Horizontal flipping
#            if random.random() > 0.5:
#                image = TF.hflip(image)
#                mask = TF.hflip(mask)
                
            #In DVN mask should be continuous
            image = TF.to_tensor(image)      
            mask = TF.to_tensor(mask)
        
        return image, mask
    
#%%Evaluation Metrics
def IOU_batch(y_pred, y_true):
    batch_size = y_pred.shape[0]
    
    scores = torch.zeros(batch_size, 1)
    for i in range(batch_size):
        scores[i] = IOU(y_pred[i], y_true[i])
    
    return scores

#y_pred and y_true are all "torch tensor"
def IOU(y_pred, y_true):
    
    y_pred = torch.flatten(y_pred).reshape(1,-1)
    y_true = torch.flatten(y_true).reshape(1,-1)

    y_concat = torch.cat([y_pred, y_true], 0)

    intersect = torch.sum(torch.min(y_concat, 0)[0]).float()
    union = torch.sum(torch.max(y_concat, 0)[0]).float()
    return (intersect / max(10 ** -8, union))

#%%Model 
def weights_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight.data)
        
class DVN(nn.Module):
     
    #define each layer of neural network
    def __init__(self):
         super(DVN, self). __init__()
         #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
         self.conv1 = nn.Conv2d(4, 64, 5, 1)
         self.bn1 = nn.BatchNorm2d(64)
         self.conv2 = nn.Conv2d(64, 128, 5, 2)
         self.bn2 = nn.BatchNorm2d(128)
         self.conv3 = nn.Conv2d(128, 128, 5, 2)
         self.bn3 = nn.BatchNorm2d(128)
         
         #Linear(in_features, out_features, bias=True)
         self.fc1 = nn.Linear(128*4*4, 1024)
         self.fc2 = nn.Linear(1024, 512)
         self.fc3 = nn.Linear(512, 256)
         self.fc4 = nn.Linear(256, 128)
         self.fc5 = nn.Linear(128, 1)
         
    #define how input will be processed through those layers
    def forward(self, imgs, masks):
         x = torch.cat((imgs, masks), 1)
         x = F.relu(self.bn1(self.conv1(x)))
         x = F.relu(self.bn2(self.conv2(x)))
         x = F.relu(self.bn3(self.conv3(x)))
         #Dont forget to flatten before connect to FC layers
         x = x.view(-1, 128*4*4)
         x = F.relu(self.fc1(x))
         #Apply dropout on the first FC layer as paper mentioned
         x = F.dropout(x, p=0.25)
         x = F.relu(self.fc2(x))
         x = F.dropout(x, p=0.25)
         x = F.relu(self.fc3(x))
         x = F.dropout(x, p=0.25)
         x = F.relu(self.fc4(x))
         x = F.relu(self.fc5(x))

#         x = F.sigmoid(x)
         return x
     
        
#%% Inference 
def generate_train_tuple(model, imgs, masks, isTrain=True):
    """generate training tuple"""
  
    #zero mask for inference
#    init_masks = torch.zeros_like(masks)
    init_masks = torch.randn(masks.shape[0], masks.shape[1], masks.shape[2], masks.shape[3]).to(device)
    k = np.random.uniform()
    
    if isTrain and k >= 0.6:
#        print("Input GT Label")
        iteration = random.randint(0, 30)
        inf_masks = inference(model, imgs, init_masks, num_iterations=iteration, isTrain=isTrain)
    
    elif isTrain:
        iteration = random.randint(0, 30)
        inf_masks = inference(model, imgs, init_masks, masks, num_iterations=iteration, isTrain=isTrain)
    
    else:
#        print("Input inference Label")
        iteration = 30
        inf_masks = inference(model, imgs, init_masks, num_iterations=iteration, isTrain=isTrain)
        
    
    iou = IOU_batch(inf_masks, masks)
    
    return inf_masks, iou
        
        
def inference(model, imgs, init_masks, gt_labels=None, lr=20, num_iterations=30, isTrain=True):
    """Run the inference"""
    
    #Don't update parameters when doing inference
    if not isTrain:
        model.eval()
        
    inf_masks = init_masks.requires_grad_()
    
    #enhance loss or IOU is a regression problem, not classification
    loss_fn = nn.MSELoss()
    
    optim_inf = Adam(inf_masks, lr= 5)
    
    with torch.enable_grad():
        for idx in range(0, num_iterations):              
            inf_IOU = model(imgs, inf_masks)
            
            if gt_labels is not None:
                #make adversarial sample
                true_IOU = IOU_batch(inf_masks, gt_labels).to(device)
                value = loss_fn(inf_IOU, true_IOU)
            else:
                #make inference sample
                value = inf_IOU
                
            if idx==0 or idx==19:
                #monitor if gradient ascent works 
                print("iteration={} \t value={}".format(idx+1, torch.mean(value)))
            
            #calculate the gradient for gradient ascend
            grad = torch.autograd.grad(value, inf_masks, grad_outputs=torch.ones_like(value),
                                           only_inputs=True)
            
            grad = grad[0].detach()
            inf_masks = inf_masks + optim_inf.update(grad)
            inf_masks = torch.clamp(inf_masks, 0, 1)
    
#    inf_masks = inf_masks.detach()
#    inf_masks[inf_masks > 0.5] = 1
#    inf_masks[inf_masks < 1] = 0
            
    return inf_masks
    
#%%Training and Validation
def train(args, model, device, train_loader, optimizer, epochs) :
    
    model.train()
    train_loss = 0
    
    #DVN network is a regression problem
    loss_fn = nn.MSELoss()
    
    #define the operation batch-wise
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        #send the data into GPU or CPU
        imgs, masks = imgs.to(device), masks.to(device)
        
        #generate training tuple
        inf_masks, iou = generate_train_tuple(model, imgs, masks, isTrain=True)
        inf_masks, iou = inf_masks.to(device), iou.to(device)
        
        #clear the gradient in the optimizer in the begining of each backpropagation
        optimizer.zero_grad()
        #get out
        pred_IOU = model(imgs,inf_masks)
       
        loss  = loss_fn(pred_IOU, iou)
        #do backprobagation to get gradient
        loss.backward()
        
        #update the parameters
        optimizer.step()
        
        #train loss
        train_loss+=loss.item()
    
    return train_loss/len(train_loader)
    
def test(args, model, device, test_loader, epochs):
    
    model.eval()
    test_loss = 0
    total_iou = 0
    
    #DVN network is a regression problem
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for imgs, masks in test_loader:            
            imgs, masks = imgs.to(device), masks.to(device)
            
            #generate training tuple
            inf_masks, iou = generate_train_tuple(model, imgs, masks, isTrain=False)
            inf_masks, iou = inf_masks.to(device), iou.to(device)

            #clear the gradient in the optimizer in the begining of each backpropagation
            optimizer.zero_grad()
        
            pred_IOU = model(imgs,inf_masks)
            loss  = loss_fn(pred_IOU, iou)
            
            #calculate IOU of inference
            init_masks = torch.randn(masks.shape[0], masks.shape[1], masks.shape[2], masks.shape[3]).to(device)
            pred_masks = inference(model, imgs, init_masks, num_iterations=30, isTrain=False)
            total_iou+= torch.mean(IOU_batch(pred_masks, masks))
            
            
            #visualize the inference image
            pred_masks = pred_masks.detach().cpu()
            directory = "./inference/epoch" + str(epochs)
            save_grid_imgs(pred_masks, directory)
            
            
            #Average the loss (batch_wise)
            test_loss += loss.item()

    return test_loss/len(test_loader), total_iou/len(test_loader)



#%%Main Function
if __name__ == "__main__":
    
    #Version of Pytorch
    print("Pytorch Version:", torch.__version__)
    

    #Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    #Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #root directory of dataset
    image_dir = './images'
    mask_dir = './masks'
    
    
    #Create FCN
    model = DVN().to(device)
    #print the model summery
    print(model)
    
    
    #Visualize the output of each layer via torchSummary
    summary(model, input_size = [(3,32,32),(1,32,32)] )
    
#%%Dataloaders
        #Use Dataset to resize ,convert to Tensor, and data augmentation
    dataset = WeizmannHorseDataset(image_dir, mask_dir, transform = 
                                         transforms.Compose([
                                               transforms.ToPILImage(),
                                               transforms.Resize(size=(32,32))                                         
                                           ]))
    
    batch_size = args.batch_size
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))        
    train_indices, val_indices = indices[0:200], indices[201:-1]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
    
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

#%%
    
 #optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_loss = []
    validation_loss = []
    inference_iou = []
    best_inference_iou = 0
    
    
    model.load_state_dict(torch.load('./DVN_Horse_best_para'))
    
    start_time = time.time()
    #Start Training
    for epoch in range(1, args.epochs+1):
        #train loss
        t_loss = train(args, model, device, train_loader, optimizer, epoch)
        print('Train Epoch: {} \t Loss: {:.6f}\t'.format(
            epoch, t_loss))
        
        #validation loss
        v_loss, v_inf_iou = test(args, model, device, valid_loader, epoch)
        print('Validation Epoch: {} \t Loss: {:.6f}\t Mean_IOU(%):{}%'.format(
            epoch, v_loss, v_inf_iou))
        
        scheduler.step()
        train_loss.append(t_loss)
        validation_loss.append(v_loss)
        inference_iou.append(inference_iou)
        if v_inf_iou > best_inference_iou:
            best_inference_iou = v_inf_iou
            torch.save(model.state_dict(), "DVN_Horse_best_{}_para".format(best_inference_iou))
        
        print('-------------------------------------------------------')
        
    print("--- %s seconds ---" % (time.time() - start_time))

        
    print("training:", len(train_loader))
    print("validation:", len(valid_loader))
    x = list(range(1, args.epochs+1))
    #plot train/validation loss versus epoch
    plt.figure()
    plt.title("Train/Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.plot(x, train_loss,label="train loss")
    plt.plot(x, validation_loss, color='red', label="validation loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
#%%
    #plot train/validation loss versus epoch
    plt.figure()
    plt.title("Train/Validation IOU")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IOU")
    plt.plot(x, inference_iou, color='red', label="validation iou")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    
    #test set
    print ("Best Inference Mean IOU:", best_inference_iou)
    model.load_state_dict(torch.load('./DVN_Horse_para'))
