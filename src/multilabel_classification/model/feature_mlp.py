import torch.nn as nn
import torch.nn.functional as F
import torch


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
