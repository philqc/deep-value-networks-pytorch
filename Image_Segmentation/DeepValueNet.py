# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:36:23 2019
@author: Gabriel Hsu
"""

from __future__ import print_function, division

import os
import threading
from queue import Queue, Empty

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.optim as optim
from torchsummary import summary

#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")


__author__ = "HSU CHIH-CHAO, University of Montreal"
#%% data preprocessing 

#build the dataset, generate the "training tuple"
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
            
            #create a channel for mask so as to transform
            mask = self.transform(np.expand_dims(mask, axis=2))
            
        return image, mask
        
#%% extended domain oracle value function
#define the oracle function for image segmentation(F1, IOU, or Dice Score) (tensor?) 
        
def f1_score_batch(y_pred, y_true):
    batch_size = y_pred.shape[0]
    scores = torch.zeros(batch_size, 1)
    for i in range(batch_size):
        scores[i] = f1_score(y_pred[i], y_true[i])
    
    return scores

#y_pred and y_true are all "torch tensor"
def f1_score(y_pred, y_true):
    
    y_pred = torch.flatten(y_pred).reshape(1,-1)
    y_true = torch.flatten(y_true).reshape(1,-1)

    y_concat = torch.cat([y_pred, y_true], 0)

    intersect = torch.sum(torch.min(y_concat, 0)[0])
    union = torch.sum(torch.max(y_concat, 0)[0])
    return -1*intersect / max(10 ** -8, union)

#%%
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        
#define the DVN for Image segmentation (Can be any type of network u like)
class DeepValueNet(nn.Module):
    
     #define each layer of neural network
     def __init__(self):
         super(DeepValueNet, self). __init__()
         #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
         self.conv1 = nn.Conv2d(4, 64, 5, 1)
         self.conv2 = nn.Conv2d(64, 128, 5, 2)
         self.conv3 = nn.Conv2d(128, 128, 5, 2)
         #Linear(in_features, out_features, bias=True)
         self.fc1 = nn.Linear(128*4*4, 384)
         self.fc2 = nn.Linear(384, 192)
         self.fc3 = nn.Linear(192, 1)
     
     #define how input will be processed through those layers
     def forward(self, x):
         x = F.relu(self.conv1(x))
         x = F.relu(self.conv2(x))
         x = F.relu(self.conv3(x))
         #dont forget to flatten before connect to FC layers
         x = x.view(-1, 128*4*4)
         x = F.relu(self.fc1(x))
         #apply dropout on the first FC layer as paper mentioned
         x = F.dropout(x, p=0.75)
         x = F.relu(self.fc2(x))
         x = F.relu(self.fc3(x))
         #assume the oracle value function is IOU (range from 0 ~　１)
         x = F.sigmoid(x)
         
         return x

#%%      
def show_sample(img ,label, f1_score):
#    plt.subplot(2,1,1)
#    plt.imshow(np.squeeze(img.to('cpu').numpy()))
    print('f1 =', f1_score)   

def cross_entropy_loss(y_pred, y_true):
    t1 = -1*y_true*torch.log(y_pred)
    t2 = (1-y_true)*torch.log(1-y_pred)
    
    return torch.sum(t1-t2) /16
    
         
#define the training method  
def train(imgs, masks, model, device, batch_size, optimizer, epochs) :
    
    model.train()
    
    #split the dataset
    train_imgs = imgs[0:200]
    train_masks = masks[0:200]
    
    val_imgs = imgs[201:300]
    val_masks = masks[201:300]
    
    test_imgs = imgs[301:]
    test_masks = masks[301:]
    
    data_size = imgs.shape[0]
    
    ls = nn.BCELoss()
    
    #Training Process
    for epoch in range(0, epochs):
        #shuffle the training dataset
        train_loss = 0
        print('epoch:', epoch)
        queue = create_sample_queue(model, train_imgs, train_masks, batch_size)
        model.zero_grad()
        optimizer.zero_grad()
        while True:
            if queue.empty() != True:
                #get training tuple from queue
                image, label, f1_score = queue.get(timeout=10)
#                show_sample(image, label, f1_score)
                image, label, f1_score = image.to(device), label.to(device), f1_score.to(device)
                input_data = torch.cat((image,label), 1)
                
                #concatenate input as a 4 channel image 
                output = model(input_data)
                loss = ls(output, f1_score)
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()
            else:
                break
    
#        train_loss /= data_size
        print("Training loss = ", train_loss)
    
    return train_loss
    
#%% The functions for creating training tuple
         
#define Inference method for prediction
def inference(model, imgs, init_masks, gt_labels=None, learning_rate=0.001, num_iterations=20):
    """Run the inference"""
    
#    model.eval()
#    
#    #figure out to(device)
#    imgs = imgs.to(device)
    
    ls = nn.BCELoss()
    pred_masks = init_masks
    
    with torch.enable_grad():
        for idx in range(0, num_iterations):
            input_data = Variable(torch.cat((imgs, pred_masks), 1), requires_grad = True)
            prediction = model(input_data)
            if gt_labels is not None:
                 v = f1_score_batch(pred_masks, gt_labels)
                 loss = ls(prediction.to(device), v.to(device))
#                 print('loss', loss.item())
                 value = loss
            else:
                value = prediction
                print(value)
#                print (value)
                
            grad = torch.autograd.grad(value, input_data, grad_outputs=torch.ones_like(value),
                                           only_inputs=True)
            grad = grad[0].detach()
#            print(grad[:,3:4,:,:].shape)
#            print(pred_masks.shape)
            pred_masks += learning_rate * grad[:,3:4,:,:]
            pred_masks = torch.clamp(pred_masks, 0, 1)
#            pred_masks[pred_masks < 0.] = 0
#            pred_masks[pred_masks > 0.] = 1
    
    return pred_masks 
    
    

#generate training tuples during training
def generate_examples(model, imgs, masks, train = False, val = False):
    
    """generate training tuple (adversarial or normal inference)"""
    
    #fix the always zero label
    init_masks = torch.randn(masks.shape[0], masks.shape[1], masks.shape[2], masks.shape[3]).to(device)
    
    #50% chance to get adversarial training sample
    if train and np.random.rand() >= 0.5:
        #Initialize 50% Ground truth y_pred, 50% from zero matrices
        gt_sample_choice = torch.randn(masks.shape[0]) > 0.5
        init_masks[gt_sample_choice] = masks[gt_sample_choice]
        pred_masks = inference(model, imgs, init_masks, masks, num_iterations = 1)
        
    else:
        pred_masks = inference(model, imgs, init_masks)
        
    
    #create correspond f1_scores for pred_masks
    scores = f1_score_batch(pred_masks, masks)

        
    return pred_masks, scores
    
#create syncrhonized queue to accumulate the sample:
def create_sample_queue(model, train_imgs, train_masks, batch_size, num_threads = 5):
    #need to reconsider the maxsize
    tuple_queue = Queue()
    indices_queue = Queue()
    for idx in np.arange(0, train_imgs.shape[0], batch_size):
        indices_queue.put(idx)

    #parallel work here
    def generate():
        try:
            while True:
                #get a batch
                idx = indices_queue.get_nowait()
#                print(idx)
                imgs = train_imgs[idx: min(train_imgs.shape[0], idx + batch_size)]
                masks = train_masks[idx: min(train_masks.shape[0], idx + batch_size)]

                #generate data (training tuples)
                pred_masks, f1_scores = generate_examples(model, imgs, masks, train = True)
                tuple_queue.put((imgs, pred_masks, f1_scores))
                
        except Empty:
            #put empty object as a end signal
            print("empty detect")
            
        
    for _ in range(num_threads):
        thread = threading.Thread(target = generate)
        thread.start()
        thread.join()
    
    return tuple_queue

#%%
#predict the image segmentation result(binoptimizer, epoch):
def predit(x, model, device):
    return 0

#%%
if __name__ == "__main__":
    
    #args
    
    use_cuda = torch.cuda.is_available()
    #Use GPU if it is available
    device = torch.device("cuda" if use_cuda else "cpu")
    
    image_dir = './images'
    mask_dir = './masks'
    

    #Use Dataset to resize and convert to Tensor
    WhorseDataset = WeizmannHorseDataset(image_dir, mask_dir, transform = 
                                         transforms.Compose([
                                               transforms.ToPILImage(),
                                               transforms.Resize(size=(32,32)),
                                               transforms.ToTensor()
                                           ]))
    
    data_size = len(WhorseDataset)
    loader = DataLoader(WhorseDataset, batch_size = data_size, shuffle=True)
    
    #all data and label(Tensor)
    imgs = next(iter(loader))[0]
    masks = next(iter(loader))[1]
    print(imgs[0:4].shape)
    print(masks[0:4].shape)
    print(torch.cat((imgs[0:4],masks[0:4]),1).shape)
    #Create DVN 
    DVN = DeepValueNet().to(device)
    DVN.apply(weights_init)
#    print the model summery
    print(DVN)
    
    #Visualize the output of each layer via torchSummary
    summary(DVN, (4, 32, 32))

    
#   inference test
    pred_mask = inference(DVN, imgs[0:16].to(device), torch.zeros_like(masks[0:16]).to(device), gt_labels = None, learning_rate=0.5, num_iterations=20)

    

#%%  


#    choose the optimizer 
    optimizer = optim.Adam(DVN.parameters(), lr=0.5)
    #training
    train(imgs.to(device), masks.to(device), DVN, device, 16, optimizer, 200)

    