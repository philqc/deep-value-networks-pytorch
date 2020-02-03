import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import os
import pickle

from src.model.deep_value_network import DeepValueNetwork
from src.model.spen import SPEN
from src.visualization_utils import plot_results
from src.utils import SGD, create_path_that_doesnt_exist, project_root
from .utils import calculate_hamming_loss, plot_hamming_loss, load_train_dataset_flickr
from .load_flickr import visualize_predictions
from .model.top_layer import TopLayer
from .model.conv_net import ConvNet

PATH_FLICKR = os.path.join(project_root(), "data", "flickr")


class DVNImgTagging(DeepValueNetwork):

    def __init__(self, use_top_layer, use_unary, use_features,
                 metric_optimize: str, optim: str, loss_fn: str, mode_sampling=DeepValueNetwork.Sampling_GT,
                 add_second_layer=False, learning_rate=0.01, momentum=0, weight_decay=1e-4, shuffle_n_size=False,
                 inf_lr=0.50, num_hidden=48, num_pairwise=32, label_dim=24, n_steps_inf=30, n_steps_adv=1):
        """
        Parameters
        ----------
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
        model = ConvNet(use_unary, use_features, label_dim, num_hidden,
                        num_pairwise, add_second_layer)

        super().__init__(model, metric_optimize, mode_sampling, optim, learning_rate, weight_decay,
                         inf_lr, n_steps_inf, label_dim, loss_fn, momentum=momentum)

        self.use_features = use_features
        self.use_unary = use_unary
        self.use_top_layer = use_top_layer

        # Inference hyperparameters
        self.n_steps_adv = n_steps_adv

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
        using_inference = False
        if self.using_adv_sampling() and self.training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            # n_steps = random.randint(1, self.n_steps_adv)
            pred_labels = self.inference(x, init_labels, n_steps=self.n_steps_adv, gt_labels=gt_labels)
        elif self.using_gt_sampling() and self.training and np.random.rand() >= 0.5:
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

            pred_labels = self.inference(x, init_labels, n_steps=n_steps)

        return pred_labels.detach().clone(), using_inference

    def inference(self, x, y, gt_labels=None, n_steps=20):

        if self.training:
            self.model.eval()

        y = y.detach().clone()
        y.requires_grad = True
        optim_inf = SGD(y, lr=self.inf_lr)

        with torch.enable_grad():
            for i in range(n_steps):
                self._loop_inference(gt_labels, x, y, optim_inf)

        if self.training:
            self.model.train()

        return y

    def train(self, loader):

        self.model.train()
        self.training = True
        self.new_ep = True
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, t_hamming_loss, t_size, inf_size = 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels, using_inference = self.generate_output(inputs, targets)
            output = self.model(inputs, pred_labels)
            oracle = self.oracle_value(pred_labels, targets)
            loss = self.loss_fn(output, oracle)

            if using_inference:
                inf_size += len(inputs)
                t_loss += loss.detach().clone().item()
                # use detach.clone() to avoid pytorch storing the variables in computational graph
                if self.use_hamming_metric:
                    if self.use_bce:
                        t_hamming_loss += 24. * oracle.detach().clone().sum()
                    else:
                        t_hamming_loss += oracle.detach().clone().sum()
                else:
                    t_hamming_loss += calculate_hamming_loss(pred_labels.detach().clone(), targets.detach().clone())

            if torch.isnan(loss):
                print('Loss has NaN! Loss={:.5f}'.format(loss.item()))
                raise ValueError('Loss has Nan')

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0 and inf_size > 0:
                print_output = torch.sigmoid(output) if self.use_bce else output

                if self.use_hamming_metric:
                    scaled_out = 24 * print_output if self.use_bce else print_output
                    scale_oracle = 24 * oracle if self.use_bce else oracle
                    print('\rTraining: [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                          'Avg_Loss = {:.5f}; Avg_H_Loss = {:.4f}; Pred_H = {:.2f}; Real_H = {:.2f}'
                          ''.format(t_size, n_train, 100 * t_size / n_train,
                                    (n_train / t_size) * (time.time() - time_start), t_loss / inf_size,
                                    t_hamming_loss / inf_size, scaled_out.mean(), scale_oracle.mean()),
                          end='')
                else:
                    print('\rTraining: [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                          'Avg_Loss = {:.5f}; Hamming_Loss = {:.4f}; Pred_F1 = {:.2f}%; Real_F1 = {:.2f}%'
                          ''.format(t_size, n_train, 100 * t_size / n_train,
                                    (n_train / t_size) * (time.time() - time_start), t_loss / inf_size,
                                    t_hamming_loss / inf_size, 100 * print_output.mean(), 100 * oracle.mean()),
                          end='')

            self.new_ep = False

        t_hamming_loss /= inf_size
        t_loss /= inf_size
        self.training = False
        print('')
        return t_loss, t_hamming_loss

    def valid(self, loader):

        self.model.eval()
        self.training = False
        self.new_ep = True
        t_loss, t_hamming_loss, t_size = 0, 0, 0
        mean_f1 = []
        int_show = random.randint(0, 20)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels, _ = self.generate_output(inputs, gt_labels=None)
                output = self.model(inputs, pred_labels)

                oracle = self.oracle_value(pred_labels, targets)

                if self.use_hamming_metric:
                    if self.use_bce:
                        t_hamming_loss += 24. * oracle.sum()
                    else:
                        t_hamming_loss += oracle.sum()
                else:
                    t_hamming_loss += calculate_hamming_loss(pred_labels, targets)

                t_loss += self.loss_fn(output, oracle)
                if self.use_hamming_metric:
                    f1 = self._f1_score(pred_labels, targets)
                    for f in f1:
                        mean_f1.append(f)
                else:
                    for o in oracle:
                        mean_f1.append(o)

                self.new_ep = False

                if batch_idx == int_show:
                    visualize_predictions(inputs, pred_labels, targets,
                                          self.use_features, self.use_unary)

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        t_loss /= t_size
        t_hamming_loss /= t_size
        print_output = torch.sigmoid(output) if self.use_bce else output
        if self.use_hamming_metric:
            print('Validation: Loss = {:.5f}; Hamming_Loss = {:.4f}; Real_F1 = {:.2f}%'
                  ''.format(t_loss.item(), t_hamming_loss, 100 * mean_f1))
        else:
            print('Validation: Loss = {:.5f}; Hamming_Loss = {:.4f}; Pred_F1 = {:.2f}%, Real_F1 = {:.2f}%'
                  ''.format(t_loss.item(), t_hamming_loss, 100 * print_output.mean(), 100 * mean_f1))

        return t_loss.item(), t_hamming_loss, mean_f1


class SPENImgTagging(SPEN):

    def __init__(self, use_top_layer: bool, use_unary: bool, use_features: bool,
                 optim: str, loss_fn: str, add_second_layer=False, learning_rate=0.01,
                 momentum=0, weight_decay=1e-4, inf_lr=0.50, momentum_inf=0, num_hidden=48, num_pairwise=32,
                 label_dim=24, n_steps_inf=30):
        """
        Parameters
        ----------
        learning_rate : float
            learning rate for updating the value network parameters
            default: 0.01 in DVN paper
        inf_lr : float
            learning rate for the inference procedure
        """
        # Deep Value Network is just a ConvNet
        if self.use_top_layer:
            model = TopLayer(label_dim).to(self.device)
        else:
            model = ConvNet(use_unary, use_features, label_dim, num_hidden,
                            num_pairwise, add_second_layer).to(self.device)

        super().__init__(model, use_cuda, optim, learning_rate, weight_decay, inf_lr, n_steps_inf, label_dim,
                         loss_fn, momentum, momentum_inf)

        self.use_features = use_features
        self.use_unary = use_unary
        self.use_top_layer = use_top_layer

    def inference(self, x, n_steps: int):

        if self.training:
            self.model.eval()

        y_pred = self.get_ini_labels(x)
        optim_inf = SGD(y_pred, lr=self.inf_lr, momentum=self.momentum_inf)

        with torch.enable_grad():
            for i in range(n_steps):

                if self.use_top_layer:
                    pred_energy = self.model(y_pred)
                else:
                    pred_energy = self.model(x, y_pred)

                # Update y_pred
                y_pred = self._loop_inference(pred_energy, y_pred, optim_inf)

        if self.training:
            self.model.train()

        return y_pred

    def _compute_loss(self, inputs, targets):
        pred_labels = self.inference(inputs, self.n_steps_inf)

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
        loss = torch.max(pre_loss, torch.zeros(pre_loss.size()).to(self.device))

        # Take the mean over all losses of the mini batch
        loss = torch.mean(loss)

        # round prediction to binary 0/1
        pred_labels = pred_labels.round().float()

        hamming_loss = calculate_hamming_loss(pred_labels, targets)

        f1 = self._f1_score(targets, pred_labels)
        return pred_labels, loss, hamming_loss, f1

    def train(self, loader):

        self.model.train()
        self.training = True
        self.new_ep = True
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, t_hamming_loss, t_size = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels, loss, hamming_loss, f1 = self._compute_loss(inputs, targets)

            t_loss += loss.detach().clone().item()
            t_hamming_loss += hamming_loss

            if torch.isnan(loss):
                print('Loss has NaN! Loss={:.5f}'.format(loss.item()))
                raise ValueError('Loss has Nan')

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining: [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                      'Avg_Loss = {:.5f}; Hamming_Loss = {:.4f}'
                      ''.format(t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size,
                                t_hamming_loss / t_size),
                      end='')

            self.new_ep = False

        t_hamming_loss /= t_size
        t_loss /= t_size
        self.training = False
        print('')
        return t_loss, t_hamming_loss

    def valid(self, loader):

        self.model.eval()
        self.training = False
        self.new_ep = True
        t_loss, t_hamming_loss, t_size = 0, 0, 0
        mean_f1 = []
        int_show = random.randint(0, 20)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels, loss, hamming_loss, f1 = self._compute_loss(inputs, targets)

                t_loss += loss.detach().clone().item()
                t_hamming_loss += hamming_loss

                for f in f1:
                    mean_f1.append(f)

                self.new_ep = False

                if batch_idx == int_show:
                    visualize_predictions(inputs, pred_labels, targets,
                                          self.use_features, self.use_unary)

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        t_loss /= t_size
        t_hamming_loss /= t_size
        print('Validation: Loss = {:.5f}; Hamming_Loss = {:.4f}; Real_F1 = {:.2f}%'
              ''.format(t_loss, t_hamming_loss, 100 * mean_f1))

        return t_loss, t_hamming_loss, mean_f1


def run_the_model(use_unary: bool, use_features: bool, train_loader: DataLoader,
                  valid_loader: DataLoader, path_save: str, use_cuda):
    mode_sampling = DeepValueNetwork.Sampling_Adv

    use_top_layer = False
    if use_top_layer:
        print('Using Top Layer Model of NIPS paper')

    use_dvn = False
    if use_dvn:
        energy_network = DVNImgTagging(use_top_layer, use_unary, use_features,
                                       add_second_layer=False, shuffle_n_size=False,
                                       learning_rate=1e-5, momentum=0, weight_decay=0, inf_lr=10.,
                                       num_hidden=12, num_pairwise=32, n_steps_inf=20, n_steps_adv=1,
                                       mode_sampling=mode_sampling, loss_fn="mse", metric_optimize="hamming",
                                       optim="adam")
        str_res = "dvn_" + mode_sampling
    else:
        energy_network = SPENImgTagging(use_top_layer, use_unary, use_features,
                                        add_second_layer=False, learning_rate=3e-2, momentum=0, weight_decay=1e-4,
                                        inf_lr=0.5, momentum_inf=0, num_hidden=1152, num_pairwise=16, label_dim=24,
                                        n_steps_inf=30, loss_fn="mse", optim="adam")
        str_res = 'spen'

    print('Using {}'.format(str_res))

    path_model = create_path_that_doesnt_exist(path_save, "model", ".pth")
    path_results = create_path_that_doesnt_exist(path_save, "results", ".pkl")

    results = {'name': str_res, 'loss_train': [],
               'hamming_loss_train': [], 'hamming_loss_valid': [],
               'use_features': use_features, 'loss_valid': [], 'f1_valid': []}

    best_val_valid = 100
    save_model = False

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(energy_network.optimizer, step_size=40, gamma=0.25)

    for epoch in range(100):
        loss_train, h_loss_train = energy_network.train(train_loader)
        loss_valid, h_loss_valid, f1_valid = energy_network.valid(valid_loader)
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


def main():
    use_features = False
    use_unary = True

    use_cuda = torch.cuda.is_available()
    train_loader, valid_loader = load_train_dataset_flickr(
        PATH_FLICKR,
        use_features=use_features,
        use_unary=use_features,
        use_cuda=use_cuda,
        batch_size=1,
        batch_size_eval=1
    )

    # Start training
    run_the_model(use_unary, use_features, train_loader,
                  valid_loader, PATH_FLICKR, use_cuda)


if __name__ == "__main__":
    main()
