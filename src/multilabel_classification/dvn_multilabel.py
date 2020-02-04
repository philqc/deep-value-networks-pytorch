import numpy as np
import os
import torch
import torch.nn as nn
from typing import Optional

from src.multilabel_classification.model.energy_network_dvn import EnergyNetwork
from src.model.deep_value_network import DeepValueNetwork
from src.multilabel_classification.utils import (
    load_training_set_bibtex, load_test_set_bibtex, PATH_BIBTEX, PATH_MODELS_ML_BIB, train_for_num_epochs
)

# Optimal parameters for Adversarial sampling from Gygli
# at https://github.com/gyglim/dvn/blob/master/reproduce_results.py
# Optimal parameters for Ground truth sampling found by us by a small
# hyperparameter search (Second layer is B matrix in SPEN)
optim_params_gt = {'lr': 1e-3, 'optim': 'adam', 'inf_lr': 0.5, 'add_second_layer': True,
                   'n_epochs_reduce': 150, 'weight_decay': 1e-4, 'num_hidden': 150}

optim_params_adv = {'lr': 0.1, 'optim': 'sgd', 'inf_lr': 0.5, 'add_second_layer': False,
                    'n_epochs_reduce': 150, 'weight_decay': 1e-3, 'num_hidden': 150}


class DVNMultiLabel(DeepValueNetwork):
    def __init__(self, metric_optimize: str, n_steps_inf: int, loss_fn: str,
                 mode_sampling: str, optim='sgd', learning_rate=1e-1, weight_decay=1e-3,
                 inf_lr=0.5, num_hidden=None, add_second_layer=False, feature_dim=1836, label_dim=159,
                 num_pairwise=16, non_linearity=nn.Softplus()):
        """
        Parameters
        ----------
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

        super().__init__(model, metric_optimize, mode_sampling, optim, learning_rate,
                         weight_decay, inf_lr, n_steps_inf, label_dim, loss_fn)

    def generate_output(self, x, training: bool, gt_labels: Optional[torch.Tensor] = None):
        """
        Generate an output y to compute
        the loss v(y, y*) --> we can use different
        techniques to generate the output
        1) Gradient based inference
        2) Simply add the ground truth outputs
        2) Generating adversarial tuples
        3) TODO: Stratified Sampling: Random samples from Y, biased towards y*
        """
        if self.using_adv_sampling() and training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            pred_labels = self.inference(x, init_labels, training, gt_labels=gt_labels, n_steps=1)
        elif self.using_gt_sampling() and training and np.random.rand() >= 0.5:
            # In training: If gt_sampling=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            pred_labels = self.inference(x, init_labels, training)

        return pred_labels


def run_test_set(path_data: str, path_save: str, mode_sampling: str):
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    test_loader = load_test_set_bibtex(path_data, path_save, use_cuda)

    if mode_sampling == DeepValueNetwork.Sampling_GT:
        params = optim_params_gt
    else:
        params = optim_params_adv

    dvn = DVNMultiLabel(mode_sampling=mode_sampling, metric_optimize="f1", loss_fn="bce",
                        optim=params['optim'], inf_lr=params['inf_lr'], learning_rate=params['lr'],
                        weight_decay=params['weight_decay'], num_hidden=params['num_hidden'],
                        add_second_layer=params['add_second_layer'], n_steps_inf=20)

    dvn.model.load_state_dict(torch.load(os.path.join(path_save, 'dvn_' + mode_sampling + ".pth")))

    print('Computing the F1 Score on the test set...')
    dvn.test(test_loader)


def run_the_model(path_data: str, path_save: str, mode_sampling: str):
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    train_loader, valid_loader = load_training_set_bibtex(path_data, path_save, use_cuda, batch_size=32)

    if mode_sampling == DeepValueNetwork.Sampling_GT:
        params = optim_params_gt
    else:
        params = optim_params_adv

    dvn = DVNMultiLabel(mode_sampling=mode_sampling, metric_optimize="f1", loss_fn="bce",
                        optim=params['optim'], inf_lr=params['inf_lr'], learning_rate=params['lr'],
                        weight_decay=params['weight_decay'], num_hidden=params['num_hidden'],
                        add_second_layer=params['add_second_layer'], n_steps_inf=10)

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(dvn.optimizer, step_size=params['n_epochs_reduce'], gamma=0.1)
    total_epochs = int(params['n_epochs_reduce'] * 2.5)

    name = 'dvn_' + mode_sampling
    path_save_model = os.path.join(path_save, name + '.pth')

    train_for_num_epochs(
        dvn,
        train_loader,
        valid_loader,
        path_save_model,
        total_epochs,
        scheduler
    )


def main():
    mode_sampling = DeepValueNetwork.Sampling_Adv
    # run_the_model(PATH_BIBTEX, PATH_MODELS_ML_BIB, mode_sampling)
    # Test the pretrained models
    run_test_set(PATH_BIBTEX, PATH_MODELS_ML_BIB, mode_sampling)


if __name__ == "__main__":
    main()
