import torch
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import Union, Tuple
from .base_model import BaseModel


class EnergyModel(BaseModel):

    def __init__(self, torch_model, optim: str, learning_rate: float, weight_decay: float,
                 inf_lr: float, n_steps_inf: int, label_dim: Union[int, Tuple[int]], loss_fn: str, momentum: float,
                 momentum_inf: float = 0.0):

        super().__init__(torch_model)

        if loss_fn.lower() == "bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.use_bce = True
            print('Using Binary Cross Entropy with Logits Loss')
        elif loss_fn.lower() == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
            self.use_bce = False
            print('Using Mean Squared Error Loss')
        else:
            raise ValueError(f"Invalid loss_fn provided = {loss_fn}")

        # Inference hyperparameters
        self.inf_lr = inf_lr
        self.n_steps_inf = n_steps_inf
        self.momentum_inf = momentum_inf
        ############################

        self.label_dim = label_dim

        if optim.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                             momentum=momentum, weight_decay=weight_decay)
        elif optim.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Invalid optimizer provided: {optim}')

    def train(self, loader: DataLoader):
        pass

    def valid(self, loader: DataLoader):
        pass

    def test(self, loader: DataLoader):
        return self.valid(loader)

    def _get_tensor_init_labels(self, x):
        if isinstance(self.label_dim, Tuple):
            # image or other 2D input
            return torch.zeros(x.size()[0], 1, self.label_dim[0], self.label_dim[1],
                               dtype=torch.float32, device=self.device)
        else:
            # 1D input
            return torch.zeros(x.size()[0], self.label_dim, dtype=torch.float32, device=self.device)

    @abstractmethod
    def get_ini_labels(self, x):
        pass

    def _scaled_hamming_loss(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor, training: bool):
        """ Scaled Hamming Loss """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels, training)
        # Hamming Loss in 0-1
        loss = torch.sum(torch.abs(gt_labels - pred_labels), dim=1)
        if self.use_bce:
            loss /= self.label_dim
        loss = loss.view(-1, 1)
        return loss
