import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
from src.multilabel_classification.utils import (
    normalize_inputs, print_a_sentence_bibtex, compute_f1_score, get_bibtex
)
from src.utils import MyDataset
from src.visualization_utils import plot_results

# Parameters to reproduce the baseline results of the SPEN paper
params_baseline = {'epochs': 20, 'optim': 'adam', 'lr': 1e-3,
                   'momentum': 0, 'scheduler': 15, 'weight_decay': 1e-5}
# Parameters to do feature extraction for pretraining of SPEN
params_feature_extraction = {'epochs': 10, 'optim': 'adam', 'lr': 1e-3,
                             'momentum': 0, 'scheduler': 20, 'weight_decay': 0}


class FeatureMLP(nn.Module):

    def __init__(self, label_dim, input_dim, only_feature_extraction=False, n_hidden_units=150):
        """
        MLP to make a mapping from x -> F(x)
        where F(x) is a feature representation of the inputs
        2 layer network with sigmoid ending to predict
        independently for each x_i its label y_i
        n_hidden_units=150 in SPEN/INFNET papers for bibtex/Bookmarks
        n_hidden_units=250 in SPEN/INFNET papers for Delicious
        using Adam with lr=0.001 as the INFNET paper

        Parameters:
        ---------------
        only_feature_extraction: bool
            once the network is trained, we just use it until the second layer
            for feature extraction of the inputs.
        """
        super().__init__()

        self.only_feature_extraction = only_feature_extraction
        self.n_hidden_units = n_hidden_units

        self.fc1 = nn.Linear(input_dim, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, label_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if not self.only_feature_extraction:
            x = torch.sigmoid(self.fc3(x))
        return x


class FeatureNetwork:

    def __init__(self, use_cuda, lr=1e-3, momentum=0, optimizer='adam', weight_decay=0, input_dim=1836, label_dim=159):
        """
        Model to make a word embedding
        from x --> F(x) for SPEN/INFNET models
        It can also be used to show the decent results
        obtained by a vanilla MLP using independent-label
        cross entropy
        """
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = FeatureMLP(label_dim, input_dim).to(self.device)
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Binary Cross entropy loss
        # Computes independent loss for each label in the vector
        # Our final loss is the sum over all our losses
        self.loss_fn = nn.BCELoss(reduction='sum')

    def train(self, loader, ep):

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
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Avg_Loss = {:.5f}'
                      ''.format(ep, t_size, n_train, 100 * t_size / n_train, t_loss / t_size),
                      end='')

        t_loss /= t_size
        print('')
        return t_loss

    def valid(self, loader):
        """
        Compute the loss and the F1 Score
        on the validation set
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
        print('Validation set: Avg_Loss = {:.2f}; F1_Score = {:.2f}'.format(loss.item(), 100 * mean_f1))

        return loss.item(), mean_f1

    def test(self, loader, test_labels):

        self.model.eval()
        outputs = []
        loss, t_size = 0, 0

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.float()
                t_size += len(inputs)

                output = self.model(inputs)
                loss += self.loss_fn(output.float(), targets.float())

                output_in_0_1 = output.round().int()
                outputs.append(output_in_0_1)

        loss /= t_size

        # convert list of tensors to tensor
        b = torch.Tensor(test_labels.shape).int()
        outputs = torch.cat(outputs, out=b)
        test_labels = torch.from_numpy(test_labels)

        f1 = np.mean(compute_f1_score(test_labels, outputs))

        print('Test set : Avg_Loss = {:.2f}; F1 score = {:.2f}%'.format(loss, 100 * f1))

        return loss, f1


def run_the_model(do_feature_extraction, dir_path, use_cuda):

    print('Loading the training set...')
    train_labels, train_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'train')
    train_inputs = normalize_inputs(train_inputs, dir_path, load=False)

    n_train = int(len(train_inputs) * 0.95)
    indices = list(range(len(train_inputs)))
    # don't shuffle here because we want to use same train/valid split for SPEN

    print_a_sentence_bibtex(train_inputs[1], train_labels[1], txt_inputs, txt_labels)

    train_data = MyDataset(train_inputs, train_labels)
    batch_size, batch_size_eval = 32, 32

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[:n_train]),
        pin_memory=use_cuda
    )

    valid_loader = DataLoader(
        train_data,
        batch_size=batch_size_eval,
        sampler=SubsetRandomSampler(indices[n_train:]),
        pin_memory=use_cuda
    )

    if do_feature_extraction:
        params = params_feature_extraction
    else:
        params = params_baseline

    F_Net = FeatureNetwork(use_cuda, lr=params['lr'], momentum=params['momentum'],
                           optimizer=params['optim'], weight_decay=params['weight_decay'])

    print('train_labels.shape =', train_labels.shape,
          'train_inputs.shape =', train_inputs.shape,
          'length_txt_labels =', len(txt_labels))

    results = {'name': 'MLP_Baseline', 'loss_train': [], 'loss_valid': [], 'f1_valid': []}

    scheduler = torch.optim.lr_scheduler.StepLR(F_Net.optimizer, step_size=params['scheduler'], gamma=0.1)

    best_val_found = 0
    # Train for 10 epochs as the INFNET paper
    for epoch in range(params['epochs']):
        loss_train = F_Net.train(train_loader, epoch)
        loss_valid, f1_valid = F_Net.valid(valid_loader)
        scheduler.step()

        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)
        if f1_valid > best_val_found:
            best_val_found = f1_valid
            print('--- Saving model at F1 = {:.2f} ---'.format(100 * best_val_found))
            torch.save(F_Net.model.state_dict(), dir_path + '/bibtex_feature_network.pth')

    # Plot results and save the model
    plot_results(results, iou=False)


def test_the_model(dir_path, use_cuda, model_path):

    F_Net = FeatureNetwork(use_cuda)
    F_Net.model.load_state_dict(torch.load(dir_path + model_path))

    # Testing phase
    print('Loading Test set...')
    test_labels, test_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'test')
    test_inputs = normalize_inputs(test_inputs, dir_path, load=True)
    test_data = MyDataset(test_inputs, test_labels)
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        pin_memory=use_cuda
    )
    print('Computing the F1 Score on the test set...')
    loss_test, f1_test = F_Net.test(test_loader, test_labels)


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    feature_extraction = True

    run_the_model(feature_extraction, dir_path, use_cuda)

    test_the_model(dir_path, use_cuda, 'bibtex_feature_network.pth')





