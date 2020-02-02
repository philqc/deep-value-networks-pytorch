import time
import torchvision.models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import os
import pickle

from src.visualization_utils import (
    show_grid_imgs, plot_results
)
from src.utils import Sampling, SGD, create_path_that_doesnt_exist
from .utils import calculate_hamming_loss, plot_hamming_loss
from .load_flickr import (
    show_pred_labels, inv_normalize, FlickrTaggingDataset, FlickrTaggingDatasetFeatures
)


class TopLayer(nn.Module):
    """ TopLayer from Graber & al. 2018:
    Deep Structured Prediction with Nonlinear Output Transformations
    """
    def __init__(self, label_dim):
        super().__init__()
        self.fc1 = nn.Linear(label_dim, 1152)
        self.fc2 = nn.Linear(1152, 1)

    def forward(self, y):
        y = F.hardtanh(self.fc1(y))
        y = self.fc2(y)
        return y


class ConvNet(nn.Module):
    def __init__(self, use_unary, use_features, label_dim, num_hidden, num_pairwise,
                 add_second_layer, non_linearity=nn.Hardtanh()):
        super().__init__()

        self.non_linearity = non_linearity
        self.top = TopLayer(label_dim)
        if use_features:
            if add_second_layer:
                self.unary_model = nn.Linear(4096, num_hidden)
                self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
                # using same initialization as DVN paper
                # torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
            else:
                self.unary_model = nn.Linear(4096, label_dim)
                self.B = None
        elif use_unary:
            if add_second_layer:
                self.unary_model = nn.Linear(label_dim, num_hidden)
                self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
                # using same initialization as DVN paper
                torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
            else:
                self.unary_model = None
                self.B = None
        else:
            # Load pretrained AlexNet on ImageNet
            self.unary_model = torchvision.models.alexnet(pretrained=True)

            # Replace the last FC layer
            tmp = list(self.unary_model.classifier)

            if add_second_layer:
                tmp[-1] = nn.Linear(4096, num_hidden)

                self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
                # using same initialization as DVN paper
                torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
            else:
                tmp[-1] = nn.Linear(4096, label_dim)
                self.B = None
            self.unary_model.classifier = nn.Sequential(*tmp)

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

    def forward(self, x, y):

        # First, send image through AlexNet if not using features
        # else use directly 4096 features computed on Unary model
        if self.unary_model is not None:
            x = self.unary_model(x)

        # Local energy
        if self.B is not None:
            e_local = self.non_linearity(x)
            e_local = torch.mm(e_local, self.B)
        else:
            e_local = x
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        e_local = e_local.view(e_local.size()[0], 1)

        # Label energy
        e_label = self.top(y)
        # e_label = self.non_linearity(torch.mm(y, self.C1))
        # e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label, e_local)
        return e_global


class DeepValueNetwork:

    def __init__(self, use_top_layer, use_bce, use_unary, use_features, use_f1_score, use_cuda,
                 mode_sampling=Sampling.GT,
                 add_second_layer=False, learning_rate=0.01, momentum=0, weight_decay=1e-4, shuffle_n_size=False,
                 inf_lr=0.50, num_hidden=48, num_pairwise=32, label_dim=24, n_steps_inf=30, n_steps_adv=1):
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
        self.use_features = use_features
        self.use_unary = use_unary
        self.use_f1_score = use_f1_score
        self.use_bce = use_bce
        self.use_top_layer = use_top_layer

        # Inference hyperparameters
        self.n_steps_adv = n_steps_adv
        self.n_steps_inf = n_steps_inf
        self.inf_lr = inf_lr
        self.new_ep = True
        ################################

        # Deep Value Network is just a ConvNet
        self.model = ConvNet(use_unary, use_features, label_dim, num_hidden,
                             num_pairwise, add_second_layer).to(self.device)

        if self.use_bce:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # Paper use SGD for convnet with learning rate = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
        #                                  momentum=momentum, weight_decay=weight_decay)

        self.shuffle_n_size = shuffle_n_size

        # for inference, make sure gradients of convnet don't get accumulated
        self.training = False

        if self.use_f1_score:
            self.get_oracle_value = lambda x, y: self.get_f1_score(x, y)
        else:
            self.get_oracle_value = lambda x, y: self.get_scaled_hamming_loss(x, y)

    def generate_output(self, x, gt_labels=None, ep=0):
        """
        Generate an output y to compute
        the loss v(y, y*) --> we can use different
        techniques to generate the output
        1) Gradient based inference
        2) Simply add the ground truth outputs
        2) Generating adversarial tuples
        3) TODO: Stratified Sampling: Random samples from Y, biased towards y*
        """

        using_inference = False
        if self.mode_sampling == Sampling.ADV and self.training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            # n_steps = random.randint(1, self.n_steps_adv)
            pred_labels = self.inference(x, init_labels, self.n_steps_adv, gt_labels=gt_labels, ep=ep)
        elif self.mode_sampling == Sampling.GT and self.training and np.random.rand() >= 0.5:
            # In training: If add_ground_truth=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            using_inference = True
            init_labels = self.get_ini_labels(x)
            if self.training and self.shuffle_n_size:
                n_steps = random.randint(1, self.n_steps_inf)
            else:
                n_steps = self.n_steps_inf

            pred_labels = self.inference(x, init_labels, n_steps, ep=ep)

        return pred_labels.detach().clone(), using_inference

    def inference(self, x, y_old, num_iterations, gt_labels=None, ep=0):

        if self.training:
            self.model.eval()

        y = y_old.detach().clone()
        y.requires_grad = True
        optim_inf = SGD(y, lr=self.inf_lr)

        with torch.enable_grad():
            for i in range(num_iterations):

                if gt_labels is not None:  # Adversarial
                    output = self.model(x, y)
                    oracle = self.get_oracle_value(y, gt_labels)
                    value = self.loss_fn(output, oracle)
                else:
                    output = self.model(x, y)
                    value = torch.sigmoid(output) if self.use_bce else output

                grad = torch.autograd.grad(value, y, grad_outputs=torch.ones_like(value), only_inputs=True)

                y_grad = grad[0].detach()

                if gt_labels is not None:
                    y = y + optim_inf.update(y_grad)
                else:
                    if self.use_f1_score:
                        y = y + optim_inf.update(y_grad)
                    else:  # We want to reduce !! the Hamming loss in this case
                        y = y - optim_inf.update(y_grad)

                # Project back to the valid range
                y = torch.clamp(y, 0, 1)

                # if (ep == 5 or ep == 10) and self.new_ep and (i == 10 or i == 15):
                #    pdb.set_trace()

        if self.training:
            self.model.train()

        return y

    def train(self, loader, ep):

        self.model.train()
        self.training = True
        self.new_ep = True
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, hamming_loss, t_size, inf_size = 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels, using_inference = self.generate_output(inputs, targets, ep)
            output = self.model(inputs, pred_labels)
            oracle = self.get_oracle_value(pred_labels, targets)
            loss = self.loss_fn(output, oracle)

            if using_inference:
                inf_size += len(inputs)
                t_loss += loss.detach().clone().item()
                # use detach.clone() to avoid pytorch storing the variables in computational graph
                if self.use_f1_score:
                    hamming_loss += calculate_hamming_loss(pred_labels.detach().clone(), targets.detach().clone())
                else:
                    if self.use_bce:
                        hamming_loss += 24. * oracle.detach().clone().sum()
                    else:
                        hamming_loss += oracle.detach().clone().sum()

            if torch.isnan(loss):
                print('Loss has NaN! Loss={:.5f}'.format(loss.item()))
                raise ValueError('Loss has Nan')

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0 and inf_size > 0:
                print_output = torch.sigmoid(output) if self.use_bce else output

                if self.use_f1_score:
                    print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                          'Avg_Loss = {:.5f}; Hamming_Loss = {:.4f}; Pred_F1 = {:.2f}%; Real_F1 = {:.2f}%'
                          ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                    (n_train / t_size) * (time.time() - time_start), t_loss / inf_size,
                                    hamming_loss / inf_size, 100 * print_output.mean(), 100 * oracle.mean()),
                          end='')
                else:
                    scaled_out = 24 * print_output if self.use_bce else print_output
                    scale_oracle = 24 * oracle if self.use_bce else oracle
                    print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                          'Avg_Loss = {:.5f}; Avg_H_Loss = {:.4f}; Pred_H = {:.2f}; Real_H = {:.2f}'
                          ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                    (n_train / t_size) * (time.time() - time_start), t_loss / inf_size,
                                    hamming_loss / inf_size, scaled_out.mean(), scale_oracle.mean()),
                          end='')

            self.new_ep = False

        hamming_loss /= inf_size
        t_loss /= inf_size
        self.training = False
        print('')
        return t_loss, hamming_loss

    def valid(self, loader, ep):

        self.model.eval()
        self.training = False
        self.new_ep = True
        loss, hamming_loss, t_size = 0, 0, 0
        mean_f1 = []
        int_show = random.randint(0, 20)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels, _ = self.generate_output(inputs, gt_labels=None, ep=ep)
                output = self.model(inputs, pred_labels)

                oracle = self.get_oracle_value(pred_labels, targets)

                if self.use_f1_score:
                    hamming_loss += calculate_hamming_loss(pred_labels, targets)
                else:
                    if self.use_bce:
                        hamming_loss += 24. * oracle.sum()
                    else:
                        hamming_loss += oracle.sum()

                loss += self.loss_fn(output, oracle)

                if self.use_f1_score:
                    for o in oracle:
                        mean_f1.append(o)
                else:
                    f1 = self.get_f1_score(pred_labels, targets)
                    for f in f1:
                        mean_f1.append(f)
                self.new_ep = False

                if batch_idx == int_show:
                    idx = np.random.choice(np.arange(len(inputs)), 3, replace=False)
                    if not self.use_features and not self.use_unary:
                        inputs_unnormalized = inputs[idx].cpu()
                        inputs_unnormalized = [inv_normalize(i) for i in inputs_unnormalized]
                        show_grid_imgs(inputs_unnormalized, black_and_white=False)
                    for i, j in enumerate(idx):
                        print('({}) pred labels: '.format(i), end='')
                        show_pred_labels(pred_labels[j], False)
                        print('({}) true labels: '.format(i), end='')
                        show_pred_labels(targets[j], True)
                        print('------------------------------------')

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        loss /= t_size
        hamming_loss /= t_size
        print_output = torch.sigmoid(output) if self.use_bce else output
        if self.use_f1_score:
            print('Validation: Loss = {:.5f}; Hamming_Loss = {:.4f}; Pred_F1 = {:.2f}%, Real_F1 = {:.2f}%'
                  ''.format(loss.item(), hamming_loss, 100 * print_output.mean(), 100 * mean_f1))
        else:
            print('Validation: Loss = {:.5f}; Hamming_Loss = {:.4f}; Real_F1 = {:.2f}%'
                  ''.format(loss.item(), hamming_loss, 100 * mean_f1))

        return loss.item(), hamming_loss, mean_f1


class SPEN:

    def __init__(self, use_top_layer, use_bce, use_unary, use_features, use_f1_score, use_cuda,
                 add_second_layer=False, learning_rate=0.01, momentum=0, weight_decay=1e-4,
                 inf_lr=0.50, momentum_inf=0, num_hidden=48, num_pairwise=32, label_dim=24, n_steps_inf=30):
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
        """

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.label_dim = label_dim
        self.use_features = use_features
        self.use_unary = use_unary
        self.use_f1_score = use_f1_score
        self.use_bce = use_bce
        self.use_top_layer = use_top_layer

        # Inference hyperparameters
        self.n_steps_inf = n_steps_inf
        self.inf_lr = inf_lr
        self.momentum_inf = momentum_inf
        self.new_ep = True
        ################################

        # Deep Value Network is just a ConvNet
        if self.use_top_layer:
            self.model = TopLayer(label_dim).to(self.device)
        else:
            self.model = ConvNet(use_unary, use_features, label_dim, num_hidden,
                                 num_pairwise, add_second_layer).to(self.device)

        if self.use_bce:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # Paper use SGD for convnet with learning rate = 0.01
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # for inference, make sure gradients of convnet don't get accumulated
        self.training = False

    def get_ini_labels(self, x, gt_labels=None):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = torch.zeros(x.size()[0], self.label_dim, dtype=torch.float32, device=self.device)

        # Set requires_grad=True after in_place operation (changing the indices)
        y.requires_grad = True
        return y

    def inference(self, x, gt_labels, num_iterations):

        if self.training:
            self.model.eval()

        if self.use_top_layer:
            y_pred = x
        else:
            y_pred = self.get_ini_labels(x)

        y_pred.requires_grad = True

        optim_inf = SGD(y_pred, lr=self.inf_lr, momentum=self.momentum_inf)

        with torch.enable_grad():
            for i in range(num_iterations):

                if self.use_top_layer:
                    pred_energy = self.model(y_pred)
                else:
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

    def train(self, loader, ep):

        self.model.train()
        self.training = True
        self.new_ep = True
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, hamming_loss, t_size = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels = self.inference(inputs, targets, self.n_steps_inf)
            if self.use_top_layer:
                pred_energy = self.model(inputs)
            else:
                pred_energy = self.model(inputs, pred_labels)

            # Energy ground truth
            if self.use_top_layer:
                gt_energy = self.model(targets)
            else:
                gt_energy = self.model(inputs, targets)

            # print('pred_energy', pred_energy.mean(),'gt_energy', gt_energy.mean())

            # Max-margin Loss
            pre_loss = self.loss_fn(pred_labels, targets) - pred_energy + gt_energy
            loss = torch.max(pre_loss, torch.zeros(pre_loss.size()).to(self.device))

            # Take the mean over all losses of the mini batch
            loss = torch.mean(loss)

            t_loss += loss.detach().clone().item()
            hamming_loss += calculate_hamming_loss(pred_labels.detach().clone(), targets.detach().clone())

            if torch.isnan(loss):
                print('Loss has NaN! Loss={:.5f}'.format(loss.item()))
                raise ValueError('Loss has Nan')

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                      'Avg_Loss = {:.5f}; Hamming_Loss = {:.4f}'
                      ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size,
                                hamming_loss / t_size),
                      end='')

            self.new_ep = False

        hamming_loss /= t_size
        t_loss /= t_size
        self.training = False
        print('')
        return t_loss, hamming_loss

    def valid(self, loader, ep):

        self.model.eval()
        self.training = False
        self.new_ep = True
        t_loss, hamming_loss, t_size = 0, 0, 0
        mean_f1 = []
        int_show = random.randint(0, 20)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels = self.inference(inputs, targets, self.n_steps_inf)

                if self.use_top_layer:
                    pred_energy = self.model(inputs)
                else:
                    pred_energy = self.model(inputs, pred_labels)

                # Energy ground truth
                if self.use_top_layer:
                    gt_energy = self.model(targets)
                else:
                    gt_energy = self.model(inputs, targets)

                # Max-margin Loss
                pre_loss = self.loss_fn(pred_labels, targets) - pred_energy + gt_energy
                # Take the mean over all losses of the mini batch
                # loss = torch.max(pre_loss, torch.zeros(pre_loss.size()).to(self.device))
                t_loss += torch.mean(pre_loss).item()

                # round prediction to binary 0/1
                pred_labels = pred_labels.round().float()

                hamming_loss += calculate_hamming_loss(pred_labels, targets)

                f1 = self.get_f1_score(targets, pred_labels)
                for f in f1:
                    mean_f1.append(f)

                self.new_ep = False

                if batch_idx == int_show:
                    idx = np.random.choice(np.arange(len(inputs)), 3, replace=False)
                    if not self.use_features and not self.use_unary:
                        inputs_unnormalized = inputs[idx].cpu()
                        inputs_unnormalized = [inv_normalize(i) for i in inputs_unnormalized]
                        show_grid_imgs(inputs_unnormalized, black_and_white=False)
                    for i, j in enumerate(idx):
                        print('({}) pred labels: '.format(i), end='')
                        show_pred_labels(pred_labels[j], False)
                        print('({}) true labels: '.format(i), end='')
                        show_pred_labels(targets[j], True)
                        print('------------------------------------')

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        t_loss /= t_size
        hamming_loss /= t_size
        print('Validation: Loss = {:.5f}; Hamming_Loss = {:.4f}; Real_F1 = {:.2f}%'
              ''.format(t_loss, hamming_loss, 100 * mean_f1))

        return t_loss, hamming_loss, mean_f1


def run_the_model(use_unary, use_features, train_loader, valid_loader, dir_path, use_cuda):
    mode_sampling = Sampling.ADV

    use_f1_score = False
    if use_f1_score:
        print('Using F1 Score as Oracle Value !')
    else:
        print('Using Hamming Loss as Oracle Value !')

    use_bce = False
    if use_bce:
        print('Using Binary Cross Entropy with Logits Loss')
    else:
        print('Using Mean Squared Error Loss')

    use_top_layer = False
    if use_top_layer:
        print('Using Top Layer Model of NIPS paper')

    use_dvn = False
    if use_dvn:
        energy_network = DeepValueNetwork(use_top_layer, use_bce, use_unary, use_features, use_f1_score, use_cuda,
                                          mode_sampling, add_second_layer=False, shuffle_n_size=False,
                                          learning_rate=1e-5, momentum=0, weight_decay=0, inf_lr=10.,
                                          num_hidden=12, num_pairwise=32, n_steps_inf=20, n_steps_adv=1)
        str_res = 'DVN_Ground_Truth' if mode_sampling == Sampling.GT else 'DVN_Adversarial'
    else:
        energy_network = SPEN(use_top_layer, use_bce, use_unary, use_features, use_f1_score, use_cuda,
                              add_second_layer=False, learning_rate=3e-2, momentum=0, weight_decay=1e-4,
                              inf_lr=0.5, momentum_inf=0, num_hidden=1152, num_pairwise=16, label_dim=24,
                              n_steps_inf=30)
        str_res = 'SPEN'

    print('Using {}'.format(str_res))

    dir_results = os.path.join(dir_path, "results")
    path_model = create_path_that_doesnt_exist(dir_results, "model", ".pth")
    path_results = create_path_that_doesnt_exist(dir_results, "results", ".pkl")

    results = {'name': str_res, 'loss_train': [],
               'hamming_loss_train': [], 'hamming_loss_valid': [],
               'use_features': use_features, 'loss_valid': [], 'f1_valid': []}

    best_val_valid = 100
    save_model = False

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(energy_network.optimizer, step_size=40, gamma=0.25)

    for epoch in range(100):
        loss_train, h_loss_train = energy_network.train(train_loader, epoch)
        loss_valid, h_loss_valid, f1_valid = energy_network.valid(valid_loader, epoch)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)
        results['hamming_loss_train'].append(h_loss_train)
        results['hamming_loss_valid'].append(h_loss_valid)

        with open(path_results, 'wb') as fout:
            pickle.dump(results, fout)

        if epoch > 10 and save_model and h_loss_valid < best_val_valid:
            best_val_valid = h_loss_valid
            print('--- Saving model at Hamming = {:.4f} ---'.format(h_loss_valid))
            torch.save(energy_network.model.state_dict(), path_model)

    plot_results(results, iou=False)
    plot_hamming_loss(results)


def start(dir_path):
    type_dataset = 'full'
    img_dir = None  # 'mirflickr/'
    label_dir = None  # dir_path + '/annotations/'

    # Set ON/OFF !!
    use_features = True
    if use_features:
        print('Using Precomputed Unary Features !')

    use_unary = False
    if use_unary:
        print('Using Precomputed Unary Predictions !')

    if use_unary and use_features:
        raise ValueError('Both using features and unary is impossible')

    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    train_label_file = dir_path + '/preprocessed/train_labels_1k.pt'
    val_label_file = dir_path + '/preprocessed/val_labels.pt'
    train_save_img_file = dir_path + '/preprocessed/train_imgs_1k.pt'
    val_save_img_file = dir_path + '/preprocessed/val_imgs.pt'

    train_feature_file = dir_path + '/preprocessed/train_1k_features_20_epochs.pt'
    val_feature_file = dir_path + '/preprocessed/val_features_20_epochs.pt'

    train_unary_file = dir_path + '/preprocessed/train_unary_20_epochs.pt'
    val_unary_file = dir_path + '/preprocessed/val_unary_20_epochs.pt'

    if use_features:
        load = True if train_feature_file is not None else False
    elif use_unary:
        load = True if train_unary_file is not None else False
    else:
        load = True if train_save_img_file is not None else False

    print('Loading training set....')
    if use_features:
        train_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=train_feature_file,
                                                 annotations_folder=label_dir, save_label_file=train_label_file,
                                                 mode='train', load=load)
    elif use_unary:
        train_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=train_unary_file,
                                                 annotations_folder=label_dir, save_label_file=train_label_file,
                                                 mode='train', load=load)
    else:
        train_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=train_save_img_file,
                                         annotations_folder=label_dir, save_label_file=train_label_file,
                                         mode='train', load=load)

    print('Loading validation set....')
    if use_features:
        valid_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=val_feature_file,
                                                 annotations_folder=label_dir, save_label_file=val_label_file,
                                                 mode='val', load=load)
    elif use_unary:
        valid_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=val_unary_file,
                                                 annotations_folder=label_dir, save_label_file=val_label_file,
                                                 mode='val', load=load)
    else:
        valid_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=val_save_img_file,
                                         annotations_folder=label_dir, save_label_file=val_label_file,
                                         mode='val', load=load)

    print('Using a {} train {} validation split'.format(len(train_set), len(valid_set)))

    batch_size = 32
    batch_size_eval = 512

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=use_cuda
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_eval,
        pin_memory=use_cuda
    )

    # Start training
    run_the_model(use_unary, use_features, train_loader, valid_loader, dir_path, use_cuda)


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    start(dir_path)


if __name__ == "__main__":
    main()
