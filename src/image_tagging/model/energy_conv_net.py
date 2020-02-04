import torch.nn as nn
import torch
import torchvision.models
import numpy as np

from .top_layer import TopLayer


class EnergyConvNet(nn.Module):
    def __init__(self, use_unary, use_features, label_dim, num_hidden, num_pairwise,
                 add_second_layer, non_linearity=nn.Hardtanh()):
        super().__init__()

        self.non_linearity = non_linearity
        self.top = TopLayer(label_dim)
        if use_features:
            if add_second_layer:
                self.unary_model = nn.Linear(4096, num_hidden)
                self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
                # using same initialization as DVN paper
                # torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
            else:
                self.unary_model = nn.Linear(4096, label_dim)
                self.B = None
        elif use_unary:
            if add_second_layer:
                self.unary_model = nn.Linear(label_dim, num_hidden)
                self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
                # using same initialization as DVN paper
                torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
            else:
                self.unary_model = None
                self.B = None
        else:
            # Load pretrained AlexNet on ImageNet
            self.unary_model = torchvision.models.alexnet(pretrained=True)

            # Replace the last FC layer
            tmp = list(self.unary_model.classifier)

            if add_second_layer:
                tmp[-1] = nn.Linear(4096, num_hidden)

                self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
                # using same initialization as DVN paper
                torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
            else:
                tmp[-1] = nn.Linear(4096, label_dim)
                self.B = None
            self.unary_model.classifier = nn.Sequential(*tmp)

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

    def forward(self, x, y):

        # First, send image through AlexNet if not using features
        # else use directly 4096 features computed on Unary model
        if self.unary_model is not None:
            x = self.unary_model(x)

        # Local energy
        if self.B is not None:
            e_local = self.non_linearity(x)
            e_local = torch.mm(e_local, self.B)
        else:
            e_local = x
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        e_local = e_local.view(e_local.size()[0], 1)

        # Label energy
        e_label = self.top(y)
        # e_label = self.non_linearity(torch.mm(y, self.C1))
        # e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label, e_local)
        return e_global

