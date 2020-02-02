import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import pickle

from src.utils import Sampling, SGD, create_path_that_doesnt_exist
from src.visualization_utils import plot_results
from src.model.deep_value_network import DeepValueNetwork
from .utils import (
    average_over_crops, PATH_DATA_WEIZMANN, PATH_SAVE_HORSE, show_preds_test_time,
    load_test_set_horse, load_train_set_horse
)
from .model.conv_net import ConvNet


__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"


class DVNHorse(DeepValueNetwork):

    def __init__(self, use_cuda, metric_optimize: str, optim: str, loss_fn: str,
                 mode_sampling=Sampling.GT, learning_rate=0.01, weight_decay=1e-3, shuffle_n_size=False, inf_lr=50,
                 momentum_inf=0, label_dim=(24, 24), n_steps_inf=30, n_steps_adv=1):
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
        # Deep Value Network is just a ConvNet
        model = ConvNet()

        super().__init__(model, metric_optimize, use_cuda, mode_sampling, optim, learning_rate, weight_decay,
                         inf_lr, n_steps_inf, label_dim, loss_fn)

        # Inference hyperparameters
        self.n_steps_adv = n_steps_adv
        self.momentum_inf = momentum_inf

        self.shuffle_n_size = shuffle_n_size

    def generate_output(self, x, gt_labels=None):
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
            # n_steps = random.randint(1, self.n_steps_adv)
            pred_labels = self.inference(x, init_labels, n_steps=self.n_steps_adv, gt_labels=gt_labels)
        elif self.mode_sampling == Sampling.GT and self.training and np.random.rand() >= 0.5:
            # In training: If add_ground_truth=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            if self.training and self.shuffle_n_size:
                n_steps = random.randint(1, self.n_steps_inf)
            else:
                n_steps = self.n_steps_inf

            pred_labels = self.inference(x, init_labels, n_steps=n_steps)

        return pred_labels.detach().clone()

    def inference(self, x, y, gt_labels=None, n_steps=20):

        if self.training:
            self.model.eval()

        optim_inf = SGD(y, lr=self.inf_lr, momentum=self.momentum_inf)

        with torch.enable_grad():
            for i in range(n_steps):
                y = self._loop_inference(gt_labels, x, y, optim_inf)

        if self.training:
            self.model.train()

        return y

    def test(self, loader):
        """
        At Test time, we are averaging our predictions
        over 36 crops of 24x24 mask to predict a 32x32 mask
        """
        self.model.eval()
        self.training = False
        self.new_ep = False
        mean_iou = []

        with torch.no_grad():
            for batch_idx, (raw_inputs, inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()

                # For test: inputs is a 5d tensor
                bs, ncrops, channels, h, w = inputs.size()

                pred_labels = self.generate_output(inputs.view(-1, channels, h, w), gt_labels=None)
                # fuse batch size and ncrops to know our estimated IOU
                # output = self.model(inputs.view(-1, channels, h, w), pred_labels)
                # go back to normal shape and take the mean in the 1 dim
                # output = output.view(bs, ncrops, 1).mean(1)

                pred_labels = pred_labels.view(bs, ncrops, h, w)

                final_pred = average_over_crops(pred_labels, self.device)
                oracle = self.oracle_value(final_pred, targets)
                for o in oracle:
                    mean_iou.append(o)

                # Visualization
                show_preds_test_time(raw_inputs, final_pred, oracle)

        mean_iou = torch.stack(mean_iou)
        mean_iou = torch.mean(mean_iou)
        mean_iou = mean_iou.cpu().numpy()

        print('Test set: IOU = {:.2f}%'.format(100 * mean_iou))

        return mean_iou


def run_the_model(train_loader: DataLoader, valid_loader: DataLoader, path_save: str, use_cuda: bool,
                  save_model: bool, n_epochs: int, mode_sampling=Sampling.GT, shuffle_n_size=False,
                  learning_rate=0.01, weight_decay=1e-3, inf_lr=50., momentum_inf=0, n_steps_inf=30, n_steps_adv=1,
                  step_size_scheduler_main=300, gamma_scheduler_main=1.):

    dvn = DVNHorse(use_cuda, metric_optimize="iou", mode_sampling=mode_sampling, shuffle_n_size=shuffle_n_size,
                   learning_rate=learning_rate, weight_decay=weight_decay, inf_lr=inf_lr,
                   momentum_inf=momentum_inf, n_steps_inf=n_steps_inf, n_steps_adv=n_steps_adv,
                   optim="adam", loss_fn="bce")

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(dvn.optimizer, step_size=step_size_scheduler_main,
                                                gamma=gamma_scheduler_main)

    results = {'name': 'dvn_whorse', 'loss_train': [],
               'loss_valid': [], 'IOU_valid': [], 'batch_size': train_loader.batch_size,
               'shuffle_n_size': shuffle_n_size, 'learning_rate': learning_rate,
               'weight_decay': weight_decay, 'batch_size_eval': valid_loader.batch_size,
               'mode_sampling': mode_sampling, 'inf_lr': inf_lr, 'n_steps_inf': n_steps_inf,
               'momentum_inf': momentum_inf,
               'step_size_scheduler_main': step_size_scheduler_main,
               'gamma_scheduler_main': gamma_scheduler_main, 'n_steps_adv': n_steps_adv}

    path_results = create_path_that_doesnt_exist(path_save, "results_" + mode_sampling, ".pkl")
    path_model_save = create_path_that_doesnt_exist(path_save, "model_" + mode_sampling, ".pth")

    best_iou_valid = 0

    for epoch in range(n_epochs):
        loss_train = dvn.train(train_loader)
        loss_valid, iou_valid = dvn.valid(valid_loader)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['IOU_valid'].append(iou_valid)

        with open(path_results, 'wb') as fout:
            pickle.dump(results, fout)

        if epoch > 20 and save_model and iou_valid > best_iou_valid:
            best_iou_valid = iou_valid
            print('--- Saving model at IOU_{:.2f} ---'.format(100 * best_iou_valid))
            torch.save(dvn.model.state_dict(), path_model_save)

    plot_results(results, iou=True)


def start():
    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    train_loader, valid_loader = load_train_set_horse(PATH_DATA_WEIZMANN, use_cuda,
                                                      batch_size=1, batch_size_valid=20)
    mode_sampling = Sampling.ADV

    # Run the model
    run_the_model(train_loader, valid_loader, PATH_SAVE_HORSE, use_cuda, save_model=True,
                  n_epochs=1000, mode_sampling=mode_sampling, shuffle_n_size=False, learning_rate=1e-4,
                  weight_decay=1e-5, inf_lr=5e3, momentum_inf=0, n_steps_inf=30,
                  n_steps_adv=3, step_size_scheduler_main=800, gamma_scheduler_main=0.10)

    # plot_results(results, iou=True)


def run_test_set(path_best_model: str):
    """ Compute IOU on test set using 36 crops averaging """
    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dvn = DVNHorse(use_cuda, mode_sampling=Sampling.ADV, learning_rate=1e-4, weight_decay=1e-3,
                   inf_lr=5e2, n_steps_inf=30, n_steps_adv=3, metric_optimize="iou", optim="adam", loss_fn="bce")

    dvn.model = ConvNet().to(device)
    dvn.model.load_state_dict(torch.load(path_best_model))
    dvn.model.eval()

    # Compute IOU single prediction on 24x24 crops and 36 crops averaging on 32x32
    for i in range(2):
        thirtysix_crops = False if i == 0 else True

        test_loader = load_test_set_horse(PATH_DATA_WEIZMANN, use_cuda,
                                          batch_size=8, thirtysix_crops=thirtysix_crops)

        print('-------------------------------------------')
        if i == 0:
            print('Single crop IOU prediction')
            dvn.valid(test_loader)
        else:
            print('36 Crops IOU prediction')
            dvn.test(test_loader)


def main():
    start()
    # run_test_set()


if __name__ == "__main__":
    main()
