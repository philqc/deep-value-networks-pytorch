from Multilabel_classification.load_bibtex import *
from auxiliary_functions import *
import numpy as np
import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import random
import pdb
from Multilabel_classification.feature_network import FeatureMLP


class EnergyNetwork(nn.Module):

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

        self.non_linearity = non_linearity

        self.B = torch.nn.Parameter(torch.transpose(-weights_last_layer_mlp, 0, 1))

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

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
    def __init__(self, use_cuda, path_feature_extractor, input_dim=1836, label_dim=159, num_pairwise=16,
                 learning_rate=0.001, weight_decay=1e-4, non_linearity=nn.Softplus()):

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.label_dim = label_dim
        self.training = False

        # F(x) model that makes a feature extraction of the inputs
        self.feature_extractor = FeatureMLP(label_dim, input_dim, only_feature_extraction=True)
        self.feature_extractor.load_state_dict(torch.load(path_feature_extractor))
        self.feature_extractor.eval()
        self.feature_dim = self.feature_extractor.n_hidden_units

        # The Energy Network
        self.model = EnergyNetwork(self.feature_extractor.fc3.weight, input_dim, label_dim,
                                   num_pairwise, non_linearity).to(self.device)

        # Squared loss
        self.loss_fn = nn.MSELoss(reduction='sum')
        # Log loss
        # self.loss_fn = nn.BCEWithLogitsLoss()

        # From SPEN paper, for training we used SGD + momentum
        # with momentum = 0.9, and learning rate + weight decay
        # are decided using the validation set
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
        #                                 momentum=0.9, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5,
                                          weight_decay=weight_decay)

    def get_ini_labels(self, x):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = torch.zeros(x.size()[0], self.label_dim, dtype=torch.float32, device=self.device)
        y.requires_grad = True
        return y

    def inference(self, x, gt_labels=None, num_iterations=30):

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
                if gt_labels is None:
                    loss = pred_energy
                else:
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

    def train(self, loader, ep):

        self.model.train()
        self.training = True
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

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
                      ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size),
                      end='')

        t_loss /= t_size
        self.training = False
        print('')
        return t_loss

    def valid(self, loader, test_set=False):

        self.model.eval()
        self.training = False
        t_loss, t_size = 0, 0
        mean_f1 = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                f_x = self.feature_extractor(inputs)

                pred_labels = self.inference(f_x)
                # Energy ground truth
                gt_energy = self.model(f_x, targets)
                pred_energy = self.model(f_x, pred_labels)

                # Max-margin Loss
                pre_loss = self.loss_fn(pred_labels, targets) - pred_energy + gt_energy
                # Take the mean over all losses of the mini batch
                pre_loss = torch.max(pre_loss, torch.zeros(pre_loss.size()))
                loss = torch.mean(pre_loss)
                t_loss += loss
                # round prediction to binary 0/1
                pred_labels = pred_labels.round().int()

                f1 = compute_f1_score(targets, pred_labels)
                for f in f1:
                    mean_f1.append(f)

        mean_f1 = np.mean(mean_f1)
        loss /= t_size

        str_first = 'Test set' if test_set else 'Validation set'
        print('{}: Avg_Loss = {:.2f}; F1_Score = {:.2f}%'
              ''.format(str_first, loss.item(), 100 * mean_f1))

        return loss.item(), mean_f1


def run_test_set(dir_path, path_feature_extractor):

    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    print('Loading Test set...')
    test_labels, test_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'test')
    test_inputs = normalize_inputs(test_inputs, dir_path, load=True)
    test_data = MyDataset(test_inputs, test_labels)
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        pin_memory=use_cuda
    )

    Spen = SPEN(use_cuda, path_feature_extractor=path_feature_extractor)

    Spen.model.load_state_dict(torch.load('Spen_bibtex.pth'))

    print('Computing the F1 Score on the test set...')
    loss_test, f1_test = Spen.valid(test_loader, test_set=True)


def run_the_model(dir_path, path_feature_extractor):

    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    print('Loading the training set...')
    train_labels, train_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'train')
    train_inputs = normalize_inputs(train_inputs, dir_path, load=False)
    train_data = MyDataset(train_inputs, train_labels)

    n_train = int(len(train_inputs) * 0.95)
    indices = list(range(len(train_inputs)))
    # don't shuffle here because we want to use same train/valid split as feature extractor
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        sampler=SubsetRandomSampler(indices[:n_train]),
        pin_memory=use_cuda
    )
    valid_loader = DataLoader(
        train_data,
        batch_size=32,
        sampler=SubsetRandomSampler(indices[n_train:]),
        pin_memory=use_cuda
    )

    print('Using a {} train {} validation split'.format(n_train, len(train_inputs) - n_train))

    Spen = SPEN(use_cuda, path_feature_extractor=path_feature_extractor)

    results = {'name': 'SPEN_bibtex', 'loss_train': [],
               'loss_valid': [], 'f1_valid': []}

    save_results_file = os.path.join(dir_path, results['name'] + '.pkl')

    scheduler = torch.optim.lr_scheduler.StepLR(Spen.optimizer, step_size=10, gamma=0.1)

    best_f1_valid = 0
    for epoch in range(25):
        loss_train = Spen.train(train_loader, epoch)
        loss_valid, f1_valid = Spen.valid(valid_loader)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)

        with open(save_results_file, 'wb') as fout:
            pickle.dump(results, fout)

        if f1_valid > best_f1_valid:
            best_f1_valid = f1_valid
            if epoch > 0:
                print('--- Saving model at F1 = {:.2f} ---'.format(100 * best_f1_valid))
                torch.save(Spen.model.state_dict(), dir_path + '/' + results['name'] + '.pth')

    plot_results(results, False)


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_feature_extractor = dir_path + '/bibtex_feature_network.pth'

    #run_the_model(dir_path, path_feature_extractor)

    #run_test_set(dir_path, path_feature_extractor)









