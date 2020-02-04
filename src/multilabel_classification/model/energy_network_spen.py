import torch.nn as nn
import torch
import numpy as np


class EnergyNetwork(nn.Module):

    def __init__(self, weights_last_layer_mlp, feature_dim=150, label_dim=159,
                 num_pairwise=16, non_linearity=nn.Softplus()):
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
        num_pairwise : int
            number of pairwise units for the global (label interactions) part
        non_linearity: pytorch nn. functions
            type of non-linearity to apply
        weights_last_layer_mlp:
            weights from the third layer of the Feature extraction MLP
             which will serve to initialize the B matrix
        """
        super().__init__()

        self.non_linearity = non_linearity

        self.B = torch.nn.Parameter(torch.transpose(-weights_last_layer_mlp, 0, 1))

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

    def forward(self, x, y):
        # Local energy
        e_local = torch.mm(x, self.B)
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        e_local = e_local.view(e_local.size()[0], 1)

        # Label energy
        e_label = self.non_linearity(torch.mm(y, self.C1))
        e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label, e_local)

        return e_global
