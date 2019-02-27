# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:36:23 2019
@author: Gabriel Hsu
"""

from __future__ import print_function, division

import os
import threading
from queue import Queue, Empty
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import pickle
import pdb
import torch.optim as optim
#from torchsummary import summary
from auxiliary_functions import *
import random
#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")


__author__ = "HSU CHIH-CHAO, University of Montreal"
#%% data preprocessing 


# build the dataset, generate the "training tuple"
class WeizmannHorseDataset(Dataset):
    """ Weizmann Horse Dataset """
    
    def __init__(self, img_dir, mask_dir, test_set, transform=None):
        """
        Args:
            img_dir(string): Path to the image file (training image)
            mask_dir(string): Path to the mask file (segmentation result)
            test_set(bool): if we want test set or train set
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        all_img_names = os.listdir(img_dir)
        all_mask_names = os.listdir(mask_dir)

        self.img_names = []
        self.mask_names = []
        for i, name in enumerate(all_img_names):
            img_number = ''.join([n for n in name if n.isdigit()])
            if int(img_number) >= 200:
                if test_set:
                    self.img_names.append(name)
                    self.mask_names.append(all_mask_names[i])
            else:
                self.img_names.append(name)
                self.mask_names.append(all_mask_names[i])

        assert len(self.mask_names) == len(self.img_names)
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


class ConvNet(nn.Module):

    # define each layer of neural network
    def __init__(self, non_linearity=nn.ReLU()):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(4, 64, 5, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.bn3 = nn.BatchNorm2d(128)
        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(128 * 4 * 4, 384)

        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 1)
        self.non_linearity = non_linearity

    # define how input will be processed through those layers
    def forward(self, x, y):
        # We first concatenate the img and the mask
        z = torch.cat((x, y), 1)
        z = self.non_linearity(self.bn1(self.conv1(z)))
        z = self.non_linearity(self.bn2(self.conv2(z)))
        z = self.non_linearity(self.bn3(self.conv3(z)))
        # don't forget to flatten before connect to FC layers
        z = z.view(-1, 128 * 4 * 4)
        z = self.non_linearity(self.fc1(z))
        # apply dropout on the first FC layer as paper mentioned
        z = F.dropout(z, p=0.75)
        z = self.non_linearity(self.fc2(z))
        z = self.non_linearity(self.fc3(z))
        return z


class DeepValueNetwork:

    def __init__(self, dataset, use_cuda, add_adversarial=True,
                 add_ground_truth=False, stratified_sampling=False, batch_size=16, batch_size_eval=16,
                 learning_rate=1e-3, inf_lr=0.5, feature_dim=(32, 32), label_dim=(32, 32),
                 non_linearity=nn.ReLU()):
        """
        Parameters
        ----------
        use_cuda: boolean
            true if we are using gpu, false if using cpu
        learning_rate : float
            learning rate for updating the value network parameters
        inf_lr : float
            learning rate for the inference procedure
        add_adversarial: bool
            Generate adversarial tuples while training.
            (Usually outperforms stratified sampling and adding ground truth)
        stratified_sampling: bool (Not yet implemented)
            Sample y proportional to its exponential oracle value.
            Sample from the exponentiated value distribution using stratified sampling.
        add_ground_truth: bool
            Simply add the ground truth outputs y* with some probably p while training.
        """

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.add_adversarial = add_adversarial
        self.add_ground_truth = add_ground_truth
        self.stratified_sampling = stratified_sampling
        if self.stratified_sampling:
            raise ValueError('Stratified sampling is not yet implemented!')
        if self.add_ground_truth and self.add_adversarial:
            raise ValueError('Adversarial examples and Adding Ground Truth are both set to true !')

        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.inf_lr = inf_lr

        # Deep Value Network is just a ConvNet
        self.model = ConvNet(non_linearity).to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval

        self.n_train = int(len(dataset) * 0.80)
        self.n_valid = len(dataset) - self.n_train

        print('Using a {} train {} validation split'.format(self.n_train, self.n_valid))
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(indices[:self.n_train]),
            pin_memory=use_cuda
        )

        self.valid_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_eval,
            sampler=SubsetRandomSampler(indices[self.n_train:]),
            pin_memory=use_cuda
        )

        # turn on/off
        self.training = False

    def get_oracle_value(self, pred_labels, gt_labels):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the IOU (intersection
        over union) when training, and the discrete IOU
        when validating/testing
        """
        if pred_labels.shape != gt_labels.shape:
            raise ValueError('Invalid labels shape: gt = ', gt_labels.shape, 'pred = ', pred_labels.shape)

        if not self.training:
            # No relaxation, 0-1 only
            pred_labels = torch.where(pred_labels >= 0.5,
                                      torch.ones(1).to(self.device),
                                      torch.zeros(1).to(self.device))
            pred_labels = pred_labels.float()

        pred_labels = torch.flatten(pred_labels).reshape(pred_labels.size()[0], -1)
        gt_labels = torch.flatten(gt_labels).reshape(gt_labels.size()[0], -1)

        intersect = torch.sum(torch.min(pred_labels, gt_labels), dim=1)
        union = torch.sum(torch.max(pred_labels, gt_labels), dim=1)

        # for numerical stability
        epsilon = torch.full(union.size(), 10 ** -8).to(self.device)

        iou = intersect / torch.max(epsilon, union)
        # we want a (Batch_size x 1) tensor
        iou = iou.view(-1, 1)
        #pdb.set_trace()
        return iou

    def get_ini_labels(self, x, gt_labels=None):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = torch.zeros(x.size()[0], 1, self.label_dim[0], self.label_dim[1],
                        dtype=torch.float32, device=self.device)

        if gt_labels is not None:
            # 50%: Start from GT; rest: start from zeros
            gt_indices = torch.rand(gt_labels.shape[0]).float().to(self.device) > 0.5
            y[gt_indices] = gt_labels[gt_indices]

        # Set requires_grad=True after in_place operation (changing the indices)
        y.requires_grad = True
        return y

    def generate_output(self, x, gt_labels):
        """
        Generate an output y to compute
        the loss v(y, y*) --> we can use different
        techniques to generate the output
        1) Gradient based inference
        2) Simply add the ground truth outputs
        2) Generating adversarial tuples
        3) TODO: Stratified Sampling: Random samples from Y, biased towards y*
        """

        if self.add_adversarial and self.training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            pred_labels = self.inference(x, init_labels, gt_labels=gt_labels, num_iterations=1)
        elif self.add_ground_truth and self.training and np.random.rand() >= 0.5:
            # In training: If add_ground_truth=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            pred_labels = self.inference(x, init_labels)

        return pred_labels

    def inference(self, x, y, gt_labels=None, num_iterations=20):

        if self.training:
            self.model.eval()

        #optim_inf = SGD(y, lr=3, momentum=0.9)
        optim_inf = Adam(y, lr=0.5)

        with torch.enable_grad():

            for i in range(num_iterations):

                if gt_labels is not None:  # Adversarial
                    oracle = self.get_oracle_value(y, gt_labels)
                    output = self.model(x, y)
                    # this is the BCE loss with logits
                    value = self.loss_fn(output, oracle)
                else:
                    output = self.model(x, y)
                    value = torch.sigmoid(output)

                grad = torch.autograd.grad(value, y, grad_outputs=torch.ones_like(value),
                                           only_inputs=True)

                y_grad = grad[0].detach()
                #pdb.set_trace()
                y = y + optim_inf.update(y_grad)
                # Project back to the valid range
                y = torch.clamp(y, 0, 1)

        if self.training:
            self.model.train()

        return y

    def train(self, ep):

        self.model.train()
        self.training = True

        time_start = time.time()
        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()

            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels = self.generate_output(inputs, targets)
            oracle = self.get_oracle_value(pred_labels, targets)
            output = self.model(inputs, pred_labels)

            loss = self.loss_fn(output, oracle)
            t_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 2 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                      'Avg_Loss = {:.5f}; IOU = {:.2f}%'
                      ''.format(ep, t_size, self.n_train, 100 * t_size / self.n_train,
                                (self.n_train / t_size) * (time.time() - time_start), t_loss / t_size, 100*oracle.mean()),
                      end='')

        t_loss /= t_size
        self.training = False
        print('')
        return t_loss

    def valid(self, loader, test_set=False):

        self.model.eval()
        self.training = False

        loss, t_size = 0, 0
        mean_iou = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels = self.generate_output(inputs, targets)
                oracle = self.get_oracle_value(pred_labels, targets)
                output = self.model(inputs, pred_labels)

                loss += self.loss_fn(oracle, output)

                mean_iou.append(oracle.mean())

        mean_iou = torch.stack(mean_iou)
        mean_iou = torch.mean(mean_iou)
        loss /= t_size

        if test_set:
            print('Test set: Avg_Loss = {:.2f};  IOU = {:.2f}%'
                  ''.format(loss.item(), 100 * mean_iou))
        else:
            print('Validation set: Avg_Loss = {:.2f}; IOU = {:.2f}%'
                  ''.format(loss.item(), 100 * mean_iou))

        return loss.item(), mean_iou

        
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
    return intersect / max(10 ** -8, union)

#%%
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        
#define the DVN for Image segmentation (Can be any type of network u like)

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
    
    ls = nn.BCEWithLogitsLoss()
    
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
def inference(model, imgs, init_masks, gt_labels=None, learning_rate=0.5, num_iterations=20):
    """Run the inference"""
    
#    model.eval()
#    
#    #figure out to(device)
#    imgs = imgs.to(device)
    
    ls = nn.BCEWithLogitsLoss()
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
                value = torch.sigmoid(prediction)

                print('value =', value)
#                print (value)
                
            grad = torch.autograd.grad(value, input_data, grad_outputs=torch.ones_like(value),
                                           only_inputs=True)
            grad = grad[0].detach()
#            print(grad[:,3:4,:,:].shape)
#            print(pred_masks.shape)
            pred_masks += learning_rate * grad[:,3:4,:,:]
            pred_masks = torch.clamp(pred_masks, 0, 1)
    
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


if __name__ == "__main__":
    
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()
    
    image_dir = './images'
    mask_dir = './masks'

    # Use Dataset to resize and convert to Tensor
    WhorseDataset = WeizmannHorseDataset(image_dir, mask_dir, test_set=False,
                                         transform=transforms.Compose([
                                               transforms.ToPILImage(),
                                               transforms.Resize(size=(32, 32)),
                                               transforms.ToTensor()]))

    DVN = DeepValueNetwork(WhorseDataset, use_cuda,
                           add_adversarial=True, add_ground_truth=False,
                           batch_size=16, batch_size_eval=16,
                           non_linearity=nn.ReLU())

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(DVN.optimizer, step_size=30, gamma=0.1)

    results = {'name': 'DVN_Whorse', 'loss_train': [],
               'loss_valid': [], 'IOU_valid': []}

    save_results_file = os.path.join(dir_path, results['name'] + '.pkl')

    for epoch in range(60):
        loss_train = DVN.train(epoch)
        loss_valid, IOU_valid = DVN.valid(DVN.valid_loader)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['IOU_valid'].append(IOU_valid)

        with open(save_results_file, 'wb') as fout:
            pickle.dump(results, fout)

    plot_results(results)
    torch.save(DVN.model.state_dict(), dir_path + '/' + results['name'] + '.pth')