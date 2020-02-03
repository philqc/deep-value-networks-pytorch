import torch.nn as nn
import torch
import numpy as np


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
