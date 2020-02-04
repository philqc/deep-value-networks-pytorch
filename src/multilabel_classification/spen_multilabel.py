import os
import torch
import torch.nn as nn
import time

from src.multilabel_classification.model.energy_network_spen import EnergyNetwork
from src.multilabel_classification.feature_network import FeatureMLP
from src.utils import SGD
from src.multilabel_classification.utils import (
    PATH_MODELS_ML_BIB, PATH_BIBTEX, load_training_set_bibtex, load_test_set_bibtex, train_for_num_epochs
)
from src.model.spen import SPEN
from src.multilabel_classification.feature_network import FILE_FEATURE_NETWORK


class SPENClassification(SPEN):
    def __init__(self, path_feature_extractor: str, loss_fn: str, optim: str = "adam", momentum=0.90,
                 inf_lr=0.1, momentum_inf=0.95, n_steps_inf=30, input_dim=1836, label_dim=159, num_pairwise=16,
                 learning_rate=1e-5, weight_decay=1e-4, non_linearity=nn.Softplus()):
        """
        From SPEN paper, for training we used SGD + momentum
        with momentum = 0.9, and learning rate + weight decay
        are decided using the validation set
        Parameters
        ----------
        path_feature_extractor
        optim
        inf_lr: From SPEN paper,  inf_lr = 0.1
        n_steps_inf: From SPEN paper, n_steps_inf = 30
        momentum_inf: From SPEN paper, momentum_inf = 0.95
        loss_fn
        momentum
        input_dim
        label_dim
        num_pairwise
        learning_rate
        weight_decay
        non_linearity
        """
        # F(x) model that makes a feature extraction of the inputs
        self.feature_extractor = FeatureMLP(label_dim, input_dim, only_feature_extraction=True)
        self.feature_extractor.load_state_dict(torch.load(path_feature_extractor))
        self.feature_extractor.eval()

        model = EnergyNetwork(self.feature_extractor.fc3.weight, input_dim, label_dim,
                              num_pairwise, non_linearity)

        super().__init__(model, optim, learning_rate, weight_decay, inf_lr, n_steps_inf, label_dim,
                         loss_fn, momentum, momentum_inf)

    def inference(self, x, training: bool, n_steps: int):

        if training:
            self.model.eval()

        y_pred = self.get_ini_labels(x)
        optim_inf = SGD(y_pred, lr=self.inf_lr, momentum=self.momentum_inf)

        with torch.enable_grad():
            for i in range(n_steps):
                pred_energy = self.model(x, y_pred)
                # Update y_pred
                y_pred = self._loop_inference(pred_energy, y_pred, optim_inf)

        if training:
            self.model.train()

        return y_pred

    def _compute_loss(self, inputs, targets, training: bool):
        f_x = self.feature_extractor(inputs)

        pred_labels = self.inference(f_x, training, self.n_steps_inf)
        pred_energy = self.model(f_x, pred_labels)
        # Energy ground truth
        gt_energy = self.model(f_x, targets)

        # Max-margin Loss
        pre_loss = self.loss_fn(pred_labels, targets) - pred_energy + gt_energy
        loss = torch.max(pre_loss, torch.zeros(pre_loss.size()))
        # Take the mean over all losses of the mini batch
        loss = torch.mean(loss)
        return pred_labels, loss

    def train(self, loader):

        self.model.train()
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels, loss = self._compute_loss(inputs, targets, True)
            t_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining: [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; Avg_Loss = {:.5f}'
                      ''.format(t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size),
                      end='')

        t_loss /= t_size
        print('')
        return t_loss

    def valid(self, loader):

        self.model.eval()
        t_loss, t_size = 0, 0
        mean_f1 = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels, loss = self._compute_loss(inputs, targets, False)
                t_loss += loss

                f1_score = self._f1_score(pred_labels, targets, False)
                for f in f1_score:
                    mean_f1.append(f)

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        loss /= t_size
        print('Avg_Loss = {:.2f}; F1_Score = {:.2f}%'.format(loss.item(), 100 * mean_f1))

        return loss.item(), mean_f1


def run_test_set(path_data: str, path_save: str, path_feature_extractor: str):
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    test_loader = load_test_set_bibtex(path_data, path_save, use_cuda)

    spen = SPENClassification(path_feature_extractor, loss_fn="mse")

    spen.model.load_state_dict(torch.load('Spen_bibtex.pth'))

    print('Computing the F1 Score on the test set...')
    spen.valid(test_loader)


def run_the_model(path_data: str, path_save: str, path_feature_extractor: str):
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    train_loader, valid_loader = load_training_set_bibtex(path_data, path_save, use_cuda,
                                                          batch_size=32, shuffle=False)

    spen = SPENClassification(path_feature_extractor, loss_fn="mse")

    name = 'spen_bibtex'
    path_save_model = os.path.join(path_save, name + '.pth')

    scheduler = torch.optim.lr_scheduler.StepLR(spen.optimizer, step_size=10, gamma=0.1)
    n_epochs = 10

    train_for_num_epochs(
        spen,
        train_loader,
        valid_loader,
        path_save_model,
        n_epochs,
        scheduler
    )


def main():
    path_feature_extractor = os.path.join(PATH_MODELS_ML_BIB, FILE_FEATURE_NETWORK)

    run_the_model(PATH_BIBTEX, PATH_MODELS_ML_BIB, path_feature_extractor)

    # run_test_set(dir_path, path_feature_extractor)


if __name__ == "__main__":
    main()
