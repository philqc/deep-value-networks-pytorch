from Multilabel_classification.load_bibtex import *
from auxiliary_functions import *
import numpy as np
import os
import random
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import pdb

# Optimal parameters for Adversarial sampling from Gygli
# at https://github.com/gyglim/dvn/blob/master/reproduce_results.py
# Optimal parameters for Ground truth sampling found by us by a small
# hyperparameter search (Second layer is B matrix in SPEN)
optim_params_gt = {'lr': 1e-3, 'optim': 'adam', 'inf_lr': 0.5, 'add_second_layer': True,
                   'n_epochs_reduce': 150, 'weight_decay': 1e-4, 'num_hidden': 150}

optim_params_adv = {'lr': 0.1, 'optim': 'sgd', 'inf_lr': 0.5, 'add_second_layer': False,
                    'n_epochs_reduce': 150, 'weight_decay': 1e-3, 'num_hidden': 150}


class EnergyNetwork(nn.Module):

    def __init__(self, feature_dim, label_dim, num_hidden,
                 num_pairwise, add_second_layer, non_linearity=nn.Softplus()):
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
        num_hidden : int
            number of hidden units for the linear part
        num_pairwise : int
            number of pairwise units for the global (label interactions) part
        non_linearity: pytorch nn. functions
            type of non-linearity to apply
        """
        super().__init__()

        self.non_linearity = non_linearity

        self.fc1 = nn.Linear(feature_dim, num_hidden)
        
        if add_second_layer:
            self.fc2 = nn.Linear(num_hidden, num_hidden)

            # Add what corresponds to the b term in SPEN
            # Eq. 4 in http://www.jmlr.org/proceedings/papers/v48/belanger16.pdf
            # X is Batch_size x num_hidden
            # B is num_hidden x L, so XB gives a Batch_size x L label
            # and then we multiply by the label and sum over the L labels, 
            # and we get Batch_size x 1 output  for the local energy 
            self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
            # using same initialization as DVN paper
            torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
        else:
            self.fc2 = nn.Linear(num_hidden, label_dim)
            self.B = None
        
        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

    def forward(self, x, y):

        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
    
        # Local energy
        if self.B is not None:
            e_local = torch.mm(x, self.B)          
        else:
            e_local = x
            
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        e_local = e_local.view(e_local.size()[0], 1)

        # Label energy
        e_label = self.non_linearity(torch.mm(y, self.C1))
        e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label, e_local)

        return e_global


class DeepValueNetwork:

    def __init__(self, use_cuda, mode_sampling=Sampling.GT, optim='sgd',
                 learning_rate=1e-1, weight_decay=1e-3, inf_lr=0.5, num_hidden=None, add_second_layer=False,
                 feature_dim=1836, label_dim=159, num_pairwise=16, non_linearity=nn.Softplus()):
        """
        Parameters
        ----------
        use_cuda: boolean
            true if we are using gpu, false if using cpu
        learning_rate : float
            learning rate for updating the value network parameters
        inf_lr : float
            learning rate for the inference procedure
        mode_sampling: int
            Sampling.ADV:
                Generate adversarial tuples while training.
                (Usually outperforms stratified sampling and adding ground truth)
            Sampling.STRAT: Not yet implemented)
                Sample y proportional to its exponential oracle value.
                Sample from the exponentiated value distribution using stratified sampling.
            Sampling.GT:
                Simply add the ground truth outputs y* with some probably p while training.
        """

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.mode_sampling = mode_sampling
        if self.mode_sampling == Sampling.STRAT:
            raise ValueError('Stratified sampling is not yet implemented!')

        self.label_dim = label_dim
        self.inf_lr = inf_lr

        if num_hidden:
            self.num_hidden = num_hidden
        else:
            # SPEN uses 150 see https://github.com/davidBelanger/SPEN/blob/master/mlc_cmd.sh (feature_hid_size)
            self.num_hidden = 200

        # Deep Value Network is just a SPEN
        self.model = EnergyNetwork(feature_dim, label_dim, num_hidden,
                                   num_pairwise, add_second_layer, non_linearity).to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        if optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer: {}'.format(optim))

        # turn on/off
        self.training = False

    def get_oracle_value(self, pred_labels, gt_labels):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the F1 Score when training.
        and the discrete F1 when validating/testing
        """
        if pred_labels.shape != gt_labels.shape:
            raise ValueError('Invalid labels shape: gt = ', gt_labels.shape, 'pred = ', pred_labels.shape)

        if not self.training:
            # No relaxation, 0-1 only
            pred_labels = torch.where(pred_labels >= 0.5,
                                      torch.ones(1).to(self.device),
                                      torch.zeros(1).to(self.device))
            pred_labels = pred_labels.float()

        intersect = torch.sum(torch.min(pred_labels, gt_labels), dim=1)
        union = torch.sum(torch.max(pred_labels, gt_labels), dim=1)

        # for numerical stability
        epsilon = torch.full(union.size(), 10 ** -8).to(self.device)

        f1 = 2 * intersect / (intersect + torch.max(epsilon, union))
        # we want a (Batch_size x 1) tensor
        f1 = f1.view(-1, 1)
        return f1

    def get_ini_labels(self, x, gt_labels=None):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = torch.zeros(x.size()[0], self.label_dim, dtype=torch.float32, device=self.device)

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

        if self.mode_sampling == Sampling.ADV and self.training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            pred_labels = self.inference(x, init_labels, gt_labels=gt_labels, num_iterations=1)
        elif self.mode_sampling == Sampling.GT and self.training and np.random.rand() >= 0.5:
            # In training: If gt_sampling=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            pred_labels = self.inference(x, init_labels)

        return pred_labels

    def inference(self, x, y, gt_labels=None, num_iterations=20):

        if self.training:
            self.model.eval()

        optim_inf = SGD(y, lr=self.inf_lr, momentum=0)

        with torch.enable_grad():

            for i in range(num_iterations):

                if gt_labels is not None:  # Adversarial
                    output = self.model(x, y)
                    oracle = self.get_oracle_value(y, gt_labels)
                    # this is the BCE loss with logits
                    value = self.loss_fn(output, oracle)
                else:
                    output = self.model(x, y)
                    value = torch.sigmoid(output)

                grad = torch.autograd.grad(value, y, grad_outputs=torch.ones_like(value), only_inputs=True)

                y_grad = grad[0].detach()
                y = y + optim_inf.update(y_grad)
                # Project back to the valid range
                y = torch.clamp(y, 0, 1)

        if self.training:
            self.model.train()

        return y

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

            pred_labels = self.generate_output(inputs, targets)
            output = self.model(inputs, pred_labels)
            oracle = self.get_oracle_value(pred_labels, targets)
            loss = self.loss_fn(output, oracle)
            t_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; Avg_Loss = {:.5f}; '
                      'Pred_F1 = {:.2f}%; Real_F1 = {:.2f}%'
                      ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size,
                                100 * torch.sigmoid(output).mean(), 100 * oracle.mean()),
                      end='')

        t_loss /= t_size
        self.training = False
        print('')
        return t_loss

    def valid(self, loader, test_set=False):

        self.model.eval()
        self.training = False

        loss, t_size = 0, 0
        mean_f1, mean_output = [], []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels = self.generate_output(inputs, targets)
                output = self.model(inputs, pred_labels)
                oracle = self.get_oracle_value(pred_labels, targets)
                loss += self.loss_fn(output, oracle)

                for o in oracle:
                    mean_f1.append(o)
                for o in output:
                    mean_output.append(torch.sigmoid(o))

        mean_f1, mean_output = torch.stack(mean_f1), torch.stack(mean_output)
        mean_f1, mean_output = torch.mean(mean_f1), torch.mean(mean_output)
        loss /= t_size

        str_first = 'Test set' if test_set else 'Validation set'
        print('{}: Avg_Loss = {:.2f}; Pred_F1 = {:.2f}%; Real_F1 = {:.2f}%'
              ''.format(str_first, loss.item(), 100 * mean_output, 100 * mean_f1))

        return loss.item(), mean_f1


def run_test_set(mode_sampling):

    dir_path = os.path.dirname(os.path.realpath(__file__))
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

    if mode_sampling == Sampling.GT:
        params = optim_params_gt
    else:
        params = optim_params_adv

    DVN = DeepValueNetwork(use_cuda, optim=params['optim'], inf_lr=params['inf_lr'], learning_rate=params['lr'],
                           weight_decay=params['weight_decay'], num_hidden=params['num_hidden'],
                           add_second_layer=params['add_second_layer'])

    if mode_sampling == Sampling.GT:
        DVN.model.load_state_dict(torch.load('DVN_Inference_and_Ground_Truth.pth'))
    else:
        DVN.model.load_state_dict(torch.load('DVN_Inference_and_Adversarial.pth'))

    print('Computing the F1 Score on the test set...')
    loss_test, f1_test = DVN.valid(test_loader, test_set=True)


def run_the_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    print('Loading the training set...')
    train_labels, train_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'train')
    train_inputs = normalize_inputs(train_inputs, dir_path, load=False)
    train_data = MyDataset(train_inputs, train_labels)

    n_train = int(len(train_inputs) * 0.95)
    indices = list(range(len(train_inputs)))
    random.shuffle(indices)
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

    # Choose ground truth sampling or adversarial sampling
    mode_sampling = Sampling.ADV
    str_res = 'Ground_Truth' if mode_sampling == Sampling.GT else 'Adversarial'
    print('Using {} Sampling'.format(str_res))

    if mode_sampling == Sampling.GT:
        params = optim_params_gt
    else:
        params = optim_params_adv

    DVN = DeepValueNetwork(use_cuda, mode_sampling, optim=params['optim'],
                           inf_lr=params['inf_lr'], learning_rate=params['lr'], weight_decay=params['weight_decay'],
                           num_hidden=params['num_hidden'], add_second_layer=params['add_second_layer'])

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(DVN.optimizer, step_size=params['n_epochs_reduce'], gamma=0.1)
    total_epochs = int(params['n_epochs_reduce'] * 2.5)

    results = {'name': 'DVN_Inference_and_' + str_res, 'loss_train': [],
               'loss_valid': [], 'f1_valid': []}

    save_results_file = os.path.join(dir_path, results['name'] + '.pkl')

    for epoch in range(total_epochs):
        loss_train = DVN.train(train_loader, epoch)
        loss_valid, f1_valid = DVN.valid(valid_loader)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)

        with open(save_results_file, 'wb') as fout:
            pickle.dump(results, fout)

    plot_results(results, iou=False)
    torch.save(DVN.model.state_dict(), dir_path + '/' + results['name'] + '.pth')


if __name__ == "__main__":

    #run_the_model()

    # Test the pretrained models (Ground Truth and Adversarial)
    run_test_set(mode_sampling=Sampling.ADV)




