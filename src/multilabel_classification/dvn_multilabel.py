import numpy as np
import os
import torch
import torch.nn as nn
import pickle

from src.model.deep_value_network import DeepValueNetwork
from src.visualization_utils import plot_results
from src.utils import Sampling, SGD
from src.multilabel_classification.utils import (
   load_training_set_bibtex, load_test_set_bibtex, PATH_BIBTEX, PATH_MODELS_ML_BIB
)

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


class DVNMultiLabel(DeepValueNetwork):
    def __init__(self, use_cuda, metric_optimize: str, n_steps_inf: int, loss_fn: str, mode_sampling=Sampling.GT,
                 optim='sgd', learning_rate=1e-1, weight_decay=1e-3, inf_lr=0.5, num_hidden=None,
                 add_second_layer=False, feature_dim=1836, label_dim=159, num_pairwise=16, non_linearity=nn.Softplus()):
        """
        Parameters
        ----------
        use_cuda: boolean
            true if we are using gpu, false if using cpu
        learning_rate : float
            learning rate for updating the value network parameters
        inf_lr : float
            learning rate for the inference procedure
        mode_sampling: str
            Sampling.ADV:
                Generate adversarial tuples while training.
                (Usually outperforms stratified sampling and adding ground truth)
            Sampling.STRAT: Not yet implemented)
                Sample y proportional to its exponential oracle value.
                Sample from the exponentiated value distribution using stratified sampling.
            Sampling.GT:
                Simply add the ground truth outputs y* with some probably p while training.
        """
        if num_hidden:
            num_hidden = num_hidden
        else:
            # SPEN uses 150 see https://github.com/davidBelanger/SPEN/blob/master/mlc_cmd.sh (feature_hid_size)
            num_hidden = 200

        # Deep Value Network is just a SPEN
        model = EnergyNetwork(feature_dim, label_dim, num_hidden,
                              num_pairwise, add_second_layer, non_linearity)

        super().__init__(model, metric_optimize, use_cuda, mode_sampling, optim, learning_rate,
                         weight_decay, inf_lr, n_steps_inf, label_dim, loss_fn)

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
            pred_labels = self.inference(x, init_labels, gt_labels=gt_labels, n_steps=1)
        elif self.mode_sampling == Sampling.GT and self.training and np.random.rand() >= 0.5:
            # In training: If gt_sampling=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            pred_labels = self.inference(x, init_labels)

        return pred_labels

    def inference(self, x, y, gt_labels=None, n_steps=20) -> torch.Tensor:
        if self.training:
            self.model.eval()

        optim_inf = SGD(y, lr=self.inf_lr, momentum=0)

        with torch.enable_grad():
            for i in range(n_steps):
                y = self._loop_inference(gt_labels, x, y, optim_inf)

        if self.training:
            self.model.train()

        return y

    def test(self, loader):
        return self.valid(loader)


def run_test_set(path_data: str, path_save: str, mode_sampling: str):
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    test_loader = load_test_set_bibtex(path_data, path_save, use_cuda)

    if mode_sampling == Sampling.GT:
        params = optim_params_gt
    else:
        params = optim_params_adv

    dvn = DVNMultiLabel(use_cuda, optim=params['optim'], inf_lr=params['inf_lr'], learning_rate=params['lr'],
                        weight_decay=params['weight_decay'], num_hidden=params['num_hidden'],
                        add_second_layer=params['add_second_layer'], loss_fn="bce",
                        metric_optimize="f1", n_steps_inf=20)

    if mode_sampling == Sampling.GT:
        dvn.model.load_state_dict(torch.load('DVN_Inference_and_Ground_Truth.pth'))
    else:
        dvn.model.load_state_dict(torch.load('DVN_Inference_and_Adversarial.pth'))

    print('Computing the F1 Score on the test set...')
    dvn.valid(test_loader)


def run_the_model(path_data: str, path_save: str):
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    train_loader, valid_loader = load_training_set_bibtex(path_data, path_save, use_cuda, batch_size=32)

    # Choose ground truth sampling or adversarial sampling
    mode_sampling = Sampling.ADV

    if mode_sampling == Sampling.GT:
        params = optim_params_gt
    else:
        params = optim_params_adv

    dvn = DVNMultiLabel(use_cuda, mode_sampling=mode_sampling, optim=params['optim'],
                        inf_lr=params['inf_lr'], learning_rate=params['lr'], weight_decay=params['weight_decay'],
                        num_hidden=params['num_hidden'], add_second_layer=params['add_second_layer'],
                        loss_fn="bce", metric_optimize="f1", n_steps_inf=10)

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(dvn.optimizer, step_size=params['n_epochs_reduce'], gamma=0.1)
    total_epochs = int(params['n_epochs_reduce'] * 2.5)

    results = {'name': 'dvn_' + mode_sampling, 'loss_train': [],
               'loss_valid': [], 'f1_valid': []}

    save_results_file = os.path.join(path_save, results['name'] + '.pkl')
    save_model_file = os.path.join(path_save, results['name'] + '.pth')

    for epoch in range(total_epochs):
        print(f"Epoch {epoch}")
        loss_train = dvn.train(train_loader)
        loss_valid, f1_valid = dvn.valid(valid_loader)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)

        with open(save_results_file, 'wb') as fout:
            pickle.dump(results, fout)

    plot_results(results, iou=False)
    torch.save(dvn.model.state_dict(), save_model_file)


def main():
    run_the_model(PATH_BIBTEX, PATH_MODELS_ML_BIB)
    # Test the pretrained models (Ground Truth and Adversarial)
    # run_test_set(mode_sampling=Sampling.ADV)


if __name__ == "__main__":
    main()
