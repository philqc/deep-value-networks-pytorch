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
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF

from distutils.version import LooseVersion
#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")


__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"


# build the dataset, generate the "training tuple"
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
            
            #Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(24, 24))
            image = TF.crop(image, i, j, h, w)                
            mask = TF.crop(mask, i, j, h, w)
                
            # Random Horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            image = TF.to_tensor(image)      
            mask = TF.to_tensor(mask)
        
        return image, mask
        
#%% extended domain oracle value function
#define the oracle function for image segmentation(F1, IOU, or Dice Score) (tensor?) 
        
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
    return intersect / max(10 ** -8, union)

#%%
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        
        
#define the DVN for Image segmentation (Can be any type of network u like)
class FCN(nn.Module):
     
    #define each layer of neural network
    def __init__(self):
         super(FCN, self). __init__()
          #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
         self.conv1 = nn.Conv2d(3, 64, 5, 1, padding=2)
         self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
         self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)
         
         #Deconvolution
         #nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
         self.deconv1 = nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1)
         self.deconv2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
         
    #define how input will be processed through those layers
    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = F.relu(self.conv2(x))
         x = F.relu(self.conv3(x))
         x = F.relu(self.deconv1(x)) 
         x = self.deconv2(x) 
         return x

#%%      
#define the training method  
def train(args, model, device, train_loader, optimizer, epochs) :
    
    model.train()
    train_loss = 0
    total_iou = 0
    #define the operation batch-wise
    for batch_idx, (data, target) in enumerate(train_loader):
        #send the data into GPU or CPU
        data, target = data.to(device), target.to(device)
        target[target > 0] = 1
        
        #visualization purpose
#        if epochs == 101:
#            i = random.randint(1,5)
#            print(i)
#            plt.figure()
#            plt.imshow(data[i].to('cpu').numpy().transpose((1, 2, 0)))
#            plt.figure()
#            plt.imshow(np.squeeze(target[i].to('cpu').numpy().transpose((1, 2, 0))))
        
        #clear the gradient in the optimizer in the begining of each backpropagation
        optimizer.zero_grad()
        #get out
        output = model(data)
        #define loss
        n, c, h, w = output.size()
        log_p = F.log_softmax(output, dim=1)
        
        #prediction (pick higher probability after log softmax)
        pred = torch.argmax(log_p, dim=1)
        
        #visualization purpose
#        if epochs == 101:
#                i = random.randint(1,5)
#                print(i)
#                plt.figure()
#                plt.imshow(pred[i].to('cpu').numpy())

        total_iou += torch.mean(IOU_batch(pred.detach(), target.long()))
        
        #adapted from cross_entropy_2d
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2,3)
        target = target.transpose(1, 2).transpose(2,3)
        log_p = log_p[target.repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]

        loss  = F.nll_loss(log_p, target.long())
        #do backprobagation to get gradient
        loss.backward()
        
        #update the parameters
        optimizer.step()
        
        #train loss
        train_loss+=loss.item()
    
    
    return train_loss/len(train_loader), total_iou/len(train_loader)

#Define Test Method
def testing(args, model, device, test_loader, epochs):
    model.eval()
    test_loss = 0
    total_iou = 0
    #What is this ?
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target[target > 0] = 1
            
            output = model(data)
            #prediction (pick higher probability after log softmax)
            n, c, h, w = output.size()
            log_p = F.log_softmax(output, dim=1)
            pred = torch.argmax(log_p, dim=1)
            iou = IOU_batch(pred, target.long())
            total_iou += torch.mean(iou)

#        visualization purpose
#            if epochs % 25== 0:
#                i = iou.max(0)[1].item()
#                print('Epochs:', epochs)
#                plt.figure()
#                print(iou[i])
#                plt.imshow(data[i].to('cpu').numpy().transpose((1, 2, 0)))
#                plt.figure()
#                plt.imshow(np.squeeze(target[i].to('cpu').numpy().transpose((1, 2, 0))))
#                plt.figure()
#                plt.imshow(pred[i].to('cpu').numpy())
                
            
            log_p = log_p.transpose(1, 2).transpose(2,3)
            target = target.transpose(1, 2).transpose(2,3)
            log_p = log_p[target.repeat(1, 1, 1, c) >= 0]
            log_p = log_p.view(-1, c)
            mask = target >= 0
            target = target[mask]
            
            loss  = F.nll_loss(log_p, target.long())
            #Average the loss (batch_wise)
            test_loss += loss.item()

    return test_loss/len(test_loader), total_iou/len(test_loader)


if __name__ == "__main__":
    
    # Version of Pytorch
    print("Pytorch Version:", torch.__version__)
    

    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
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
    
    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #root directory of dataset
    image_dir = './images'
    mask_dir = './masks'
    
    
    #Create FCN
    model = FCN().to(device)
    #print the model summery
    print(model)
    
    #Visualize the output of each layer via torchSummary
    summary(model, input_size = (3,24,24))
    

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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    
    train_loss = []
    train_iou = []
    validation_loss = []
    validation_iou = []
    best_val_iou = 0
    
    start_time = time.time()
    # Start Training
    for epoch in range(1, args.epochs+1):
        #train loss
        t_loss, t_mean_iou = train(args, model, device, train_loader, optimizer, epoch)
        print('Train Epoch: {} \t Loss: {:.6f}\t Mean_IOU(%):{}%'.format(
            epoch, t_loss, t_mean_iou))
        
        # validation loss
        v_loss, v_mean_iou = testing(args, model, device, valid_loader, epoch)
        print('Validation Epoch: {} \t Loss: {:.6f}\t Mean_IOU(%):{}%'.format(
            epoch, v_loss, v_mean_iou))
        
        train_loss.append(t_loss)
        train_iou.append(t_mean_iou)
        validation_loss.append(v_loss)
        validation_iou.append(v_mean_iou)
        best_val_iou = max(best_val_iou, v_mean_iou)
        
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
    
    #plot train/validation loss versus epoch
    plt.figure()
    plt.title("Train/Validation IOU")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IOU")
    plt.plot(x, train_iou,label="train iou")
    plt.plot(x, validation_iou, color='red', label="validation iou")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    
    # test set
    print("Best Validation Mean IOU:", best_val_iou)
   
    
    # Save the trained model(which means parameters)
    if args.save_model:
        torch.save(model.state_dict(), "FCN_Horse")
    

