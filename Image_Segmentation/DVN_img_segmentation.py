# -*- coding: utf-8 -*-
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
import torchvision.transforms.functional as TF
import pickle
import pdb
import torch.optim as optim
from scipy.misc import imsave, imresize
#from torchsummary import summary
from auxiliary_functions import *
import random
#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"


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
        self.normalize = None
        self.to_tensor = transforms.ToTensor()
        
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

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(24, 24))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            image, mask = self.to_tensor(image), self.to_tensor(mask)
            image = self.normalize(image)

        return image, mask

    def compute_mean_and_stddev(self):
        n_images = len(self.img_names)
        masks, images = [], []

        # ToTensor transforms the images/masks in range [0, 1]
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=(32, 32)),
                                        transforms.ToTensor()])

        for i in range(n_images):
            mask_name = os.path.join(self.mask_dir, self.mask_names[i])
            img_name = os.path.join(self.img_dir, self.img_names[i])
            mask = io.imread(mask_name)
            image = io.imread(img_name)

            image = transform(image)
            # create a channel for mask so as to transform
            mask = transform(np.expand_dims(mask, axis=2))

            masks.append(mask)
            images.append(image)

        # after torch.stack, we should have n_images x 1 x 32 x 32 for mask
        # and have n_images x 3 x 32 x 32 for images
        images, masks = torch.stack(images), torch.stack(masks)

        # compute mean and std_dev of images
        # put the images in n_images x 3 x (32x32) shape
        images = images.view(images.size(0), images.size(1), -1)
        mean_imgs = images.mean(2).sum(0) / n_images
        std_imgs = images.std(2).sum(0) / n_images
        ############################################

        # Find mean_mask for visualization purposes
        height, width = masks.shape[2], masks.shape[3]
        # flatten
        masks = masks.view(n_images, 1, -1)
        mean_mask = torch.mean(masks, dim=0)
        # go back to 32 x 32 view
        mean_mask = mean_mask.view(1, 1, height, width)

        print_mask = False
        if print_mask:
            img_to_show = mean_mask.squeeze(0)
            img_to_show = img_to_show.squeeze(0)
            show_img(img_to_show, black_and_white=True)

        return mean_imgs, std_imgs, mean_mask


class ConvNet(nn.Module):

    def __init__(self, non_linearity='relu', use_batch_norm=False):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(4, 64, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)

        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else None
        self.bn2 = nn.BatchNorm2d(128) if use_batch_norm else None
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else None

        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(128 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 1)

        non_linearity = non_linearity.lower()
        if non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        else:
            raise ValueError('Unknown activation Convnet:', non_linearity)

        self.use_batch_norm = use_batch_norm
        # apply dropout on the first FC layer as paper mentioned
        self.dropout = nn.Dropout(p=0.25)

        # define how input will be processed through those layers

    def forward(self, x, y):
        # We first concatenate the img and the mask
        z = torch.cat((x, y), 1)
        if self.use_batch_norm:
            z = self.non_linearity(self.bn1(self.conv1(z)))
            z = self.non_linearity(self.bn2(self.conv2(z)))
            z = self.non_linearity(self.bn3(self.conv3(z)))
        else:
            z = self.non_linearity(self.conv1(z))
            z = self.non_linearity(self.conv2(z))
            z = self.non_linearity(self.conv3(z))

        # flatten before FC layers
        z = z.view(-1, 128 * 6 * 6)
        z = self.non_linearity(self.fc1(z))
        z = self.dropout(z)
        z = self.non_linearity(self.fc2(z))
        z = self.fc3(z)
        return z


class DeepValueNetwork:

    def __init__(self, dataset, dir_path, use_cuda, adversarial_sampling=True,
                 gt_sampling=False, stratified_sampling=False, batch_size=16, batch_size_eval=16,
                 learning_rate=0.01, inf_lr=50, momentum_inf=0, feature_dim=(24, 24), label_dim=(24, 24),
                 non_linearity='relu', n_steps_inf=30, n_steps_inf_adversarial=1, lr_decay_inf_iteration=1,
                 lr_decay_inf_epoch=1, optimizer_inf='sgd'):
        """
        Parameters
        ----------
        use_cuda: boolean
            true if we are using gpu, false if using cpu
        learning_rate : float
            learning rate for updating the value network parameters
            default: 0.01 in DVN paper
        inf_lr : float
            learning rate for the inference procedure
        adversarial_sampling: bool
            Generate adversarial tuples while training.
            (Usually outperforms stratified sampling and adding ground truth)
        stratified_sampling: bool (Not yet implemented)
            Sample y proportional to its exponential oracle value.
            Sample from the exponentiated value distribution using stratified sampling.
        gt_sampling: bool
            Simply add the ground truth outputs y* with some probably p while training.
        """

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.adversarial_sampling = adversarial_sampling
        self.gt_sampling = gt_sampling
        self.stratified_sampling = stratified_sampling
        if self.stratified_sampling:
            raise ValueError('Stratified sampling is not yet implemented!')
        if self.gt_sampling and self.adversarial_sampling:
            raise ValueError('Adversarial examples and Adding Ground Truth are both set to true !')

        self.feature_dim = feature_dim
        self.label_dim = label_dim

        # Inference hyperparameters
        self.n_steps_inf_adversarial = n_steps_inf_adversarial
        self.n_steps_inf = n_steps_inf
        self.lr_decay_inf_epoch = lr_decay_inf_epoch
        self.lr_decay_inf_iteration = lr_decay_inf_iteration
        self.optimizer_inf = optimizer_inf
        self.inf_lr = inf_lr
        self.momentum_inf = momentum_inf
        ################################

        # for visualization purpose in "inference" function
        self.filename_train_other, self.filename_train_inf, self.filename_valid = 0, 0, 0
        # if directory doesn't exist, create it
        dir_predict_imgs = dir_path + '/pred/'
        self.dir_train_inf_img = dir_predict_imgs + '/train_inference/'
        self.dir_train_adversarial_img = dir_predict_imgs + '/train_adversarial/'
        self.dir_valid_img = dir_predict_imgs + '/valid/'
        for directory in [self.dir_train_inf_img, self.dir_train_adversarial_img, self.dir_valid_img]:
            if not os.path.isdir(directory):
                os.makedirs(directory)

        # Deep Value Network is just a ConvNet
        self.model = ConvNet(non_linearity, use_batch_norm=True).to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Paper use SGD for convnet with learning rate = 0.01
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval

        self.n_train = int(len(dataset) * 0.80)
        self.n_valid = len(dataset) - self.n_train

        print('Using a {} train {} validation split'.format(self.n_train, self.n_valid))
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(indices[:self.n_train]),
            pin_memory=use_cuda
        )

        self.valid_loader = DataLoader(
            dataset,
            batch_size=batch_size_eval,
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
            # 50% of initialization labels: Start from GT; rest: start from zeros
            gt_indices = torch.rand(gt_labels.shape[0]).float().to(self.device) > 0.5
            y[gt_indices] = gt_labels[gt_indices]

        # Set requires_grad=True after in_place operation (changing the indices)
        y.requires_grad = True
        return y

    def generate_output(self, x, gt_labels, ep=0):
        """
        Generate an output y to compute
        the loss v(y, y*) --> we can use different
        techniques to generate the output
        1) Gradient based inference
        2) Simply add the ground truth outputs
        2) Generating adversarial tuples
        3) TODO: Stratified Sampling: Random samples from Y, biased towards y*
        """

        if self.adversarial_sampling and self.training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            pred_labels = self.inference(x, init_labels, self.n_steps_inf_adversarial,
                                         gt_labels=gt_labels, ep=ep)
        elif self.gt_sampling and self.training and np.random.rand() >= 0.5:
            # In training: If add_ground_truth=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            pred_labels = self.inference(x, init_labels, self.n_steps_inf, ep=ep)

        return pred_labels

    def inference(self, x, y, num_iterations, gt_labels=None, ep=0):

        if self.training:
            self.model.eval()

        if self.optimizer_inf.lower() == 'sgd':
            optim_inf = SGD(y, lr=self.inf_lr, momentum=self.momentum_inf)
        elif self.optimizer_inf.lower() == 'adam':
            optim_inf = Adam(y, lr=self.inf_lr)
        else:
            raise ValueError('Error: Unknown optimizer for inference:', self.optimizer_inf)

        # For hyperparameter search, check if we should reduce learning rate every epoch
        # as gradients become more significant
        optim_inf.lr /= (self.lr_decay_inf_epoch ** ep)

        with torch.enable_grad():

            for i in range(num_iterations):

                optim_inf.lr /= self.lr_decay_inf_iteration

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
                y_new = y + optim_inf.update(y_grad)
                # Project back to the valid range
                y_new = torch.clamp(y_new, 0, 1)
                #pdb.set_trace()
                y = y_new
                # visualize all the inference process while training
                if i == num_iterations - 1:
                    img = y_new.detach().cpu()
                    if self.training:
                        if gt_labels is not None:
                            directory = self.dir_train_adversarial_img + str(self.filename_train_other)
                            self.filename_train_other += 1
                        else:
                            directory = self.dir_train_inf_img + str(self.filename_train_inf)
                            self.filename_train_inf += 1
                    else:
                        directory = self.dir_valid_img + str(self.filename_valid)
                        self.filename_valid += 1
                    save_grid_imgs(img, directory)

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

            pred_labels = self.generate_output(inputs, targets, ep)
            oracle = self.get_oracle_value(pred_labels, targets)
            output = self.model(inputs, pred_labels)

            loss = self.loss_fn(output, oracle)
            t_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 2 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                      'Avg_Loss = {:.5f}; Pred_IOU = {:.2f}%; Real_IOU = {:.2f}%'
                      ''.format(ep, t_size, self.n_train, 100 * t_size / self.n_train,
                                (self.n_train / t_size) * (time.time() - time_start), t_loss / t_size,
                                100 * torch.sigmoid(output).mean(), 100 * oracle.mean()),
                      end='')

        t_loss /= t_size
        self.training = False
        print('')
        return t_loss

    def valid(self, loader, test_set=False, ep=0):

        self.model.eval()
        self.training = False

        loss, t_size = 0, 0
        mean_iou = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels = self.generate_output(inputs, targets, ep)
                oracle = self.get_oracle_value(pred_labels, targets)
                output = self.model(inputs, pred_labels)

                loss += self.loss_fn(output, oracle)
                mean_iou.append(oracle.mean())

        mean_iou = torch.stack(mean_iou)
        mean_iou = torch.mean(mean_iou)
        mean_iou = mean_iou.cpu().numpy()
        # loss /= t_size

        str_first = 'Test set' if test_set else 'Validation set'
        print('{}: Loss = {:.5f}; Pred_IOU = {:.2f}%, Real_IOU = {:.2f}%'
              ''.format(str_first, loss.item(), 100 * torch.sigmoid(output).mean(), 100 * mean_iou))

        return loss.item(), mean_iou


# create synchronized queue to accumulate the sample:
def create_sample_queue(model, train_imgs, train_masks, batch_size, num_threads = 5):
    # need to reconsider the maxsize
    tuple_queue = Queue()
    indices_queue = Queue()
    for idx in np.arange(0, train_imgs.shape[0], batch_size):
        indices_queue.put(idx)

    # parallel work here
    def generate():
        try:
            while True:
                # get a batch
                idx = indices_queue.get_nowait()
#                print(idx)
                imgs = train_imgs[idx: min(train_imgs.shape[0], idx + batch_size)]
                masks = train_masks[idx: min(train_masks.shape[0], idx + batch_size)]

                # generate data (training tuples)
                # pred_masks, f1_scores = generate_examples(model, imgs, masks, train=True)
                # tuple_queue.put((imgs, pred_masks, f1_scores))
                
        except Empty:
            # put empty object as a end signal
            print("empty detect")

    for _ in range(num_threads):
        thread = threading.Thread(target = generate)
        thread.start()
        thread.join()
    
    return tuple_queue


def run_the_model(dataset, dir_path, use_cuda, save_model, early_stopping, batch_size, batch_size_eval,
                  adversarial_sampling=False, gt_sampling=False, stratified_sampling=False,
                  optimizer_inf='adam', inf_lr=5, momentum_inf=np.nan, lr_decay_inf_iteration=1,
                  lr_decay_inf_epoch=1, n_steps_inf=20, n_steps_inf_adversarial=1,
                  step_size_scheduler_main=30, gamma_scheduler_main=0.1, non_linearity='relu'):

    DVN = DeepValueNetwork(dataset, dir_path, use_cuda,
                           adversarial_sampling=adversarial_sampling, gt_sampling=gt_sampling,
                           stratified_sampling=stratified_sampling, optimizer_inf=optimizer_inf,
                           batch_size=batch_size, batch_size_eval=batch_size_eval, inf_lr=inf_lr,
                           momentum_inf=momentum_inf, lr_decay_inf_iteration=lr_decay_inf_iteration,
                           lr_decay_inf_epoch=lr_decay_inf_epoch, n_steps_inf=n_steps_inf,
                           n_steps_inf_adversarial=n_steps_inf_adversarial, non_linearity=non_linearity)

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(DVN.optimizer, step_size=step_size_scheduler_main,
                                                gamma=gamma_scheduler_main)

    results = {'name': 'DVN_Whorse', 'loss_train': [],
               'loss_valid': [], 'IOU_valid': [], 'batch_size': batch_size,
               'batch_size_eval': batch_size_eval, 'optimizer_inf': optimizer_inf,
               'non_linearity_convnet': non_linearity,
               'adversarial_sampling': adversarial_sampling, 'gt_sampling': gt_sampling,
               'stratified_sampling': stratified_sampling, 'inf_lr': inf_lr, 'n_steps_inf': n_steps_inf,
               'lr_decay_inf_iteration': lr_decay_inf_iteration, 'lr_decay_inf_epoch': lr_decay_inf_epoch,
               'momentum_inf': momentum_inf, 'step_size_scheduler_main': step_size_scheduler_main,
               'gamma_scheduler_main': gamma_scheduler_main, 'n_steps_inf_adversarial': n_steps_inf_adversarial}

    results_path = dir_path + '/results/'
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Increment a counter so that previous results with the same args will not
    # be overwritten. Comment out the next four lines if you only want to keep
    # the most recent results.
    i = 0
    while os.path.exists(results_path + str(i) + '.pkl'):
        i += 1
    results_path = results_path + str(i)

    stop_after_x = 25

    for epoch in range(300):
        loss_train = DVN.train(epoch)
        loss_valid, iou_valid = DVN.valid(DVN.valid_loader, ep=epoch)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['IOU_valid'].append(iou_valid)

        with open(results_path + '.pkl', 'wb') as fout:
            pickle.dump(results, fout)

        # Early stopping #
        if early_stopping:
            if epoch > stop_after_x \
                    and np.mean(results['loss_valid'][-10:]) > np.mean(results['loss_valid'][-stop_after_x:]) - 1e-3 \
                    and loss_valid > np.mean(results['loss_valid'][-stop_after_x:]) - 1e-3 \
                    and np.mean(results['IOU_valid'][-5:]) < 0.005 + np.mean(results['IOU_valid'][-stop_after_x:]):
                print(
                    'Loss has not improved since %s iterations; mean_last_loss = %.5f; mean_last_10 = %.5f; new = %.5f'
                    % (stop_after_x, np.mean(results['loss_valid'][-stop_after_x:]),
                       np.mean(results['loss_valid'][-10:]), loss_valid))
                print('--> We stop')
                break
            elif epoch > 5 and np.isclose(np.mean(results['IOU_valid'][:5]), iou_valid, atol=1e-6).any():
                print('Inference step is not working, Valid IOU stays the same across epochs\n --> We stop')
                break

    if save_model:
        torch.save(DVN.model.state_dict(), results_path + '.pth')


def random_hyper_parameter_search(dataset, dir_path, use_cuda, n_search,
                                  save_model, early_stopping):

    ls_batch_size = [8, 16, 32]
    batch_size_eval = 40

    ls_step_size_scheduler_main = [20, 30, 40, 50, 75, 100, 150, 200, 300]
    ls_gamma_scheduler_main = [0.1, 0.25, 0.5, 0.75, 1]

    ls_n_steps_inf = [30, 30, 35, 40, 50, 75, 100]
    ls_n_steps_inf_adversarial = [1, 1, 1, 2, 3, 5, 10]
    ls_lr_decay_inf_iteration = [1, 1, 1.05, 1.1, 1.15, 1.25, 1.5, 2]
    ls_lr_decay_inf_epoch = [1, 1, 1, 1.1, 1.15, 1.2]

    ls_optim_inf = ['adam', 'sgd']
    optim_inf = random.choice(ls_optim_inf)

    for i in range(n_search):
        print('\n----------------------------------------------')
        print(i, ' iteration of random search \n')
        adversarial_sampling = True if np.random.rand() > 0.5 else False
        gt_sampling = not adversarial_sampling

        if optim_inf == 'adam':
            ls_inf_lr = [1, 5, 10, 25, 50, 100, 250, 500]
            momentum_inf = 0
        else:
            ls_inf_lr = [5, 10, 25, 50, 100, 250, 500, 1000]
            momentum_inf = np.random.rand()

        run_the_model(dataset, dir_path, use_cuda, save_model, early_stopping,
                      batch_size=random.choice(ls_batch_size),
                      batch_size_eval=batch_size_eval, adversarial_sampling=adversarial_sampling,
                      gt_sampling=gt_sampling, optimizer_inf=optim_inf, inf_lr=random.choice(ls_inf_lr),
                      momentum_inf=momentum_inf,
                      lr_decay_inf_iteration=random.choice(ls_lr_decay_inf_iteration),
                      lr_decay_inf_epoch=random.choice(ls_lr_decay_inf_epoch),
                      n_steps_inf=random.choice(ls_n_steps_inf),
                      n_steps_inf_adversarial=random.choice(ls_n_steps_inf_adversarial),
                      step_size_scheduler_main=random.choice(ls_step_size_scheduler_main),
                      gamma_scheduler_main=random.choice(ls_gamma_scheduler_main)
                      )


def start():

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    image_dir = dir_path + '/images'
    mask_dir = dir_path + '/masks'

    # Use Dataset to resize and convert to Tensor
    WhorseDataset = WeizmannHorseDataset(image_dir, mask_dir, test_set=False)

    mean_imgs, std_imgs, mean_mask = WhorseDataset.compute_mean_and_stddev()
    print('mean_imgs =', mean_imgs, 'std_dev_imgs =', std_imgs)

    WhorseDataset.transform = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(size=(32, 32))])

    WhorseDataset.normalize = transforms.Normalize(mean_imgs, std_imgs)

    # Launch hyperparameter search
    random_hyper_parameter_search(WhorseDataset, dir_path, use_cuda, n_search=1,
                                  save_model=False, early_stopping=False)

    # plot_results(results, iou=True)


if __name__ == "__main__":
    start()


