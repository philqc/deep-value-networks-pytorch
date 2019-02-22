import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_bibtex import get_bibtex
from auxiliary_functions import *


class FeatureMLP(nn.Module):
    """ 
    MLP to make a mapping from x -> F(x) 
    where F(x) is a feature representation of the inputs
    2 layer network with sigmoid ending to predict
    independently for each x_i its label y_i
    n_hidden_units=150 in SPEN/INFNET papers for bibtex/Bookmarks
    n_hidden_units=250 in SPEN/INFNET papers for Delicious
    using Adam with lr=0.001 as the INFNET paper
    """
    def __init__(self, n_labels, dim_input, n_hidden_units=150):
        super().__init__()

        self.fc1 = nn.Linear(dim_input, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_labels)

        # Binary Cross entropy loss
        # Computes independent loss for each label in the vector
        # Our final loss is the sum over all our losses
        self.loss_fn = nn.BCELoss(reduction='sum')

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class FeatureNetwork:

    def __init__(self, inputs, labels, use_cuda):
        """
        Model to make a word embedding
        from x --> F(x) for SPEN/INFNET models
        It can also be used to show the decent results
        obtained by a vanilla MLP using independent-label
        cross entropy
        """

        if use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.dim_input = inputs.shape[1]
        self.n_labels = labels.shape[1]

        self.model = FeatureMLP(self.n_labels, self.dim_input).to(self.device)

        self.batch_size = 64
        self.batch_size_eval = 64

        self.n_train = int(len(inputs) * 0.90)

        indices = list(range(len(inputs)))
        random.shuffle(indices)

        train_data = MyDataset(inputs, labels)

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(indices[:self.n_train]),
            pin_memory=use_cuda
        )

        self.valid_loader = DataLoader(
            train_data,
            batch_size=self.batch_size_eval,
            sampler=SubsetRandomSampler(indices[self.n_train:]),
            pin_memory=use_cuda
        )

    def train(self, ep):

        self.model.train()

        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = inputs.float()

            t_size += len(inputs)

            self.model.zero_grad()

            output = self.model(inputs)
            loss = self.model.loss_fn(output.float(), targets.float())
            t_loss += loss.item()

            loss.backward()
            self.model.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Avg_Loss = {:.5f}'
                      ''.format(ep, t_size, self.n_train, 100 * t_size / self.n_train, t_loss / t_size),
                      end='')

        t_loss /= t_size
        print('')
        return t_loss

    def valid(self):
        """
        Compute the loss and the F1 Score
        on the validation set
        """
        self.model.eval()

        loss, t_size = 0, 0
        mean_f1 = []

        with torch.no_grad():
            for (inputs, targets) in self.valid_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.float()
                t_size += len(inputs)

                output = self.model(inputs)

                loss += self.model.loss_fn(output.float(), targets.float())

                # round output to 0/1
                output_in_0_1 = output.round().int()

                f1 = compute_f1_score(targets, output_in_0_1)
                mean_f1.append(f1)

        mean_f1 = np.mean(mean_f1)
        loss /= t_size
        print('Validation set: Avg_Loss = {:.2f}; F1_Score = {:.2f}'
              ''.format(loss.item(), 100 * mean_f1))

        return loss.item(), mean_f1

    def test(self, test_loader, test_labels):

        self.model.eval()
        outputs = []
        loss, t_size = 0, 0

        for (inputs, targets) in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = inputs.float()
            t_size += len(inputs)

            output = self.model(inputs)
            loss += self.model.loss_fn(output.float(), targets.float())

            output_in_0_1 = output.round().int()
            outputs.append(output_in_0_1)

        loss /= t_size

        # convert list of tensors to tensor
        b = torch.Tensor(test_labels.shape).int()
        outputs = torch.cat(outputs, out=b)
        test_labels = torch.from_numpy(test_labels)

        f1 = compute_f1_score(test_labels, outputs)

        print('Test set : Avg_Loss = {:.2f}; F1 score = {:.2f}'
              ''.format(loss, 100 * f1))

        return loss, f1


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # If a GPU is available, use it
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    train_labels, train_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'train')

    F_Net = FeatureNetwork(train_inputs, train_labels, use_cuda)

    print('train_labels.shape', train_labels.shape,
          'train_inputs.shape=', train_inputs.shape,
          'length_txt_labels', len(txt_labels))

    results = {'loss_train': [], 'loss_valid': [], 'f1_valid': []}

    # Train foo 10 epochs as the INFNET paper
    for epoch in range(10):
        loss_train = F_Net.train(epoch)
        loss_valid, mean_f1 = F_Net.valid()

        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(mean_f1)

    # Testing phase
    test_labels, test_inputs, txt_labels, txt_inputs = get_bibtex(dir_path, 'test')
    test_data = MyDataset(test_inputs, test_labels)
    test_loader = DataLoader(
        test_data,
        batch_size=F_Net.batch_size_eval,
        pin_memory=use_cuda
    )

    loss_test, f1_test = F_Net.test(test_loader, test_labels)
    plot_results(results)

    torch.save(F_Net.model.state_dict(), dir_path + '/model.pth')


