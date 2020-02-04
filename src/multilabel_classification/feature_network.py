import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from src.multilabel_classification.utils import (
    compute_f1_score, PATH_MODELS_ML_BIB, PATH_BIBTEX,
    load_training_set_bibtex, load_test_set_bibtex, train_for_num_epochs
)
from src.model.base_model import BaseModel
from src.multilabel_classification.model.feature_mlp import FeatureMLP


# Parameters to reproduce the baseline results of the SPEN paper
params_baseline = {'epochs': 20, 'optim': 'adam', 'lr': 1e-3, "batch_size": 32,
                   'momentum': 0, 'scheduler': 15, 'weight_decay': 1e-5}
# Parameters to do feature extraction for pretraining of SPEN
params_feature_extraction = {'epochs': 10, 'optim': 'adam', 'lr': 1e-3, "batch_size": 32,
                             'momentum': 0, 'scheduler': 20, 'weight_decay': 0}

FILE_FEATURE_NETWORK = "feature_network.pth"
PATH_FEATURE_NETWORK = os.path.join(PATH_MODELS_ML_BIB, FILE_FEATURE_NETWORK)


class FeatureNetwork(BaseModel):

    def __init__(self, lr=1e-3, momentum=0, optimizer='adam',
                 weight_decay=0, input_dim=1836, label_dim=159):
        """
        Model to make a word embedding
        from x --> F(x) for SPEN/INFNET models
        It can also be used to show the decent results
        obtained by a vanilla MLP using independent-label
        cross entropy
        """
        super().__init__(FeatureMLP(label_dim, input_dim))

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Binary Cross entropy loss
        # Computes independent loss for each label in the vector
        # Our final loss is the sum over all our losses
        self.loss_fn = nn.BCELoss(reduction='sum')

    def train(self, loader):

        self.model.train()

        n_train = len(loader.dataset)
        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = inputs.float()
            t_size += len(inputs)

            self.model.zero_grad()

            output = self.model(inputs)
            loss = self.loss_fn(output.float(), targets.float())
            t_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining: [{} / {} ({:.0f}%)]: Avg_Loss = {:.5f}'
                      ''.format(t_size, n_train, 100 * t_size / n_train, t_loss / t_size),
                      end='')

        t_loss /= t_size
        print('')
        return t_loss

    def valid(self, loader):
        """
        Compute the loss and the F1 Score
        """
        self.model.eval()

        loss, t_size = 0, 0
        mean_f1 = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.float()
                t_size += len(inputs)

                output = self.model(inputs)

                loss += self.loss_fn(output.float(), targets.float())

                # round output to 0/1
                output_in_0_1 = output.round().int()

                f1 = compute_f1_score(targets, output_in_0_1)
                for f in f1:
                    mean_f1.append(f)

        mean_f1 = np.mean(mean_f1)
        loss /= t_size
        print('Avg_Loss = {:.2f}; F1_Score = {:.2f}'.format(loss.item(), 100 * mean_f1))
        return loss.item(), mean_f1

    def test(self, loader):
        return self.valid(loader)


def run_the_model(f_net: FeatureNetwork, path_save: str, use_cuda: bool,
                  batch_size: int, n_epochs: int, step_size_scheduler: int):

    train_loader, valid_loader = load_training_set_bibtex(
        PATH_BIBTEX, path_save, use_cuda, batch_size=batch_size, shuffle=False
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        f_net.optimizer, step_size=step_size_scheduler, gamma=0.1
    )

    train_for_num_epochs(
        f_net,
        train_loader,
        valid_loader,
        os.path.join(path_save, FILE_FEATURE_NETWORK),
        n_epochs,
        scheduler
    )


def run_test_set(path_data: str, path_save: str, use_cuda: bool):
    f_net = FeatureNetwork(use_cuda)
    f_net.model.load_state_dict(torch.load(os.path.join(path_save, FILE_FEATURE_NETWORK)))
    test_loader = load_test_set_bibtex(path_data, path_save, use_cuda)
    print(f"Computing F1 Score on Test set...")
    f_net.valid(test_loader)


def main():
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    parser = argparse.ArgumentParser(description='DVN Img Segmentation')

    parser.add_argument('--train', type=bool, default=True,
                        help='If set to true train model, else show predictions/results on test set')

    parser.add_argument('--path_save', type=str, default=PATH_MODELS_ML_BIB,
                        help='path where to save the models')

    parser.add_argument('--feature_extraction', type=bool, default=True,
                        help='If set to true, use the hyperparameters to reproduce feature extraction of'
                             'SPEN paper, else use best hyparameters for training')

    args = parser.parse_args()

    if args.feature_extraction:
        print("Running Hyperparameters of MLP for Feature Extraction")
        params = params_feature_extraction
    else:
        print("Running Hyperparameters of MLP for Baseline")
        params = params_baseline

    f_net = FeatureNetwork(
        lr=params['lr'], momentum=params['momentum'],
        optimizer=params['optim'], weight_decay=params['weight_decay']
    )

    if args.train:
        run_the_model(f_net, PATH_MODELS_ML_BIB, use_cuda, params['batch_size'],
                      params['epochs'], params['scheduler'])
    else:
        run_test_set(PATH_BIBTEX, PATH_MODELS_ML_BIB, use_cuda)


if __name__ == "__main__":
    main()
