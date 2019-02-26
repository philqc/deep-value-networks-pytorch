from load_bibtex import *
from auxiliary_functions import *
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from feature_network import FeatureMLP
import pdb


class EnergyNetworkMLC(nn.Module):

    def __init__(self, weights_last_layer_mlp, feature_dim=150, label_dim=159,
                 num_pairwise=16, non_linearity=nn.Softplus()):
        """
        BASED on Tensorflow implementation of Michael Gygli
        see https://github.com/gyglim/dvn

        2 hidden layers with num_hidden neurons
        output is of label_dim

        Parameters
        ----------
        feature_dim : int
            dimensionality of the input features
            feature_dim = 1836 for bibtex
        label_dim : int
            dimensionality of the output labels
            label_dim = 159 for bibtex
        num_pairwise : int
            number of pairwise units for the global (label interactions) part
        non_linearity: pytorch nn. functions
            type of non-linearity to apply
        weights_last_layer_mlp:
            weights from the third layer of the Feature extraction MLP
             which will serve to initialize the B matrix
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.num_pairwise = num_pairwise

        self.non_linearity = non_linearity

        self.B = torch.nn.Parameter(torch.transpose(-weights_last_layer_mlp, 0, 1))

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(self.label_dim, self.num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / self.label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(self.num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / self.num_pairwise))

    def forward(self, x, y):

        # Local energy
        e_local = torch.mm(x, self.B)
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        e_local = e_local.view(e_local.size()[0], 1)

        # Label energy
        e_label = self.non_linearity(torch.mm(y, self.C1))
        e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label, e_local)

        return e_global


class SPEN:
    def __init__(self, inputs, targets, use_cuda, batch_size, batch_size_eval,
                 path_feature_extractor, input_dim=1836, label_dim=159, num_pairwise=16,
                 learning_rate=0.1, weight_decay=1e-4, non_linearity=nn.Softplus()):

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.num_pairwise = num_pairwise
        self.label_dim = label_dim
        self.training = False

        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval

        # F(x) model that makes a feature extraction of the inputs
        self.feature_extractor = FeatureMLP(label_dim, input_dim, only_feature_extraction=True)
        self.feature_extractor.load_state_dict(torch.load(path_feature_extractor))
        self.feature_extractor.eval()
        self.feature_dim = self.feature_extractor.n_hidden_units

        # The Energy Network
        self.model = EnergyNetworkMLC(self.feature_extractor.fc3.weight, self.feature_dim, label_dim,
                                      num_pairwise, non_linearity).to(self.device)

        # Squared loss
        self.loss_fn = nn.MSELoss()
        # Log loss
        #self.loss_fn = nn.BCEWithLogitsLoss()

        # From SPEN paper, for training we used SGD + momentum
        # with momentum = 0.9, and learning rate + weight decay
        # are decided using the validation set
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                         momentum=0.9, weight_decay=weight_decay)

        self.n_train = int(len(inputs) * 0.90)
        self.n_valid = len(inputs) - self.n_train

        indices = list(range(len(inputs)))
        #random.shuffle(indices)

        train_data = MyDataset(inputs, targets)

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(indices[:self.n_train]),
            pin_memory=use_cuda
        )

        self.valid_loader = DataLoader(
            train_data,
            batch_size=self.batch_size_eval,
            sampler=SubsetRandomSampler(indices[self.n_train:]),
            pin_memory=use_cuda
        )

    def get_ini_labels(self, x):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = torch.zeros(x.size()[0], self.label_dim,
                        dtype=torch.float32, device=self.device)

        y.requires_grad = True
        return y

    def inference(self, x, gt_labels, num_iterations=30):

        if self.training:
            self.model.eval()

        y_pred = self.get_ini_labels(x)

        # From SPEN paper, inference we use SGD with momentum
        # where momentum=0.95 and learning_rate = 0.1
        optim_inf = SGD(y_pred, lr=0.1, momentum=0.95)

        with torch.enable_grad():

            for i in range(num_iterations):

                pred_energy = self.model(x, y_pred)

                # Max-margin surrogate objective (with E(ground_truth) missing)
                loss = pred_energy - self.loss_fn(y_pred, gt_labels)

                grad = torch.autograd.grad(loss, y_pred, grad_outputs=torch.ones_like(loss),
                                           only_inputs=True)
                y_grad = grad[0].detach()
                # Gradient descent to find the y_prediction that minimizes the energy
                y_pred = y_pred - optim_inf.update(y_grad)

                # Project back to the valid range
                y_pred = torch.clamp(y_pred, 0, 1)

        if self.training:
            self.model.train()

        return y_pred

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

            f_x = self.feature_extractor(inputs)

            pred_labels = self.inference(f_x, targets)
            pred_energy = self.model(f_x, pred_labels)
            # Energy ground truth
            gt_energy = self.model(f_x, targets)

            # Max-margin Loss
            pre_loss = self.loss_fn(pred_labels, targets) - pred_energy + gt_energy
            loss = torch.max(pre_loss, torch.zeros(pre_loss.size()))
            # Take the mean over all losses of the mini batch
            loss = torch.mean(loss)
            t_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; Avg_Loss = {:.5f}'
                      ''.format(ep, t_size, self.n_train, 100 * t_size / self.n_train,
                                (self.n_train / t_size) * (time.time() - time_start), t_loss / t_size),
                      end='')

        t_loss /= t_size
        self.training = False
        print('')
        return t_loss

    def valid(self, loader, test_set=False):

        self.model.eval()
        self.training = False
        loss, t_size = 0, 0
        mean_f1 = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                f_x = self.feature_extractor(inputs)

                pred_labels = self.inference(f_x, targets)
                pred_energy = self.model(f_x, pred_labels)
                # Energy ground truth
                gt_energy = self.model(f_x, targets)

                # Max-margin Loss
                pre_loss = self.loss_fn(pred_labels, targets) - pred_energy + gt_energy
                # Take the mean over all losses of the mini batch
                pre_loss = torch.max(pre_loss, torch.zeros(pre_loss.size()))
                loss += torch.mean(pre_loss)

                # round prediction to binary 0/1
                pred_labels = pred_labels.round().int()

                f1 = compute_f1_score(targets, pred_labels)
                mean_f1.append(f1)

        mean_f1 = np.mean(mean_f1)
        loss /= t_size

        if test_set:
            print('Test set: Avg_Loss = {:.2f}; F1_Score = {:.2f}%'
                  ''.format(loss.item(), 100 * mean_f1))
        else:
            print('Validation set: Avg_Loss = {:.2f}; F1_Score = {:.2f}%'
                  ''.format(loss.item(), 100 * mean_f1))

        return loss.item(), mean_f1


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    path_feature_extractor = dir_path + '/bibtex_feature_network.pth'

    print('Loading the training set...')
    train_labels, train_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'train')

    Spen = SPEN(train_inputs, train_labels, use_cuda,
                batch_size=64, batch_size_eval=64,
                path_feature_extractor=path_feature_extractor)

    results = {'name': 'SPEN_bibtex', 'loss_train': [],
               'loss_valid': [], 'f1_valid': []}

    save_results_file = os.path.join(dir_path, results['name'] + '.pkl')

    scheduler = torch.optim.lr_scheduler.StepLR(Spen.optimizer, step_size=30, gamma=0.1)

    for epoch in range(60):
        loss_train = Spen.train(epoch)
        loss_valid, f1_valid = Spen.valid(Spen.valid_loader)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)

        with open(save_results_file, 'wb') as fout:
            pickle.dump(results, fout)

    plot_results(results)
    torch.save(Spen.model.state_dict(), dir_path + '/' + results['name'] + '.pth')

    do_test_set = True
    if do_test_set:
        # Testing phase
        print('Loading Test set...')
        test_labels, test_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'test')
        test_data = MyDataset(test_inputs, test_labels)
        test_loader = DataLoader(
            test_data,
            batch_size=Spen.batch_size_eval,
            pin_memory=use_cuda
        )
        print('Computing the F1 Score on the test set...')
        loss_test, f1_test = Spen.valid(test_loader, test_set=True)








