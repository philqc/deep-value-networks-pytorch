import torch
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import Union, Tuple
from .energy_model import EnergyModel


class SPEN(EnergyModel):

    def __init__(self, torch_model, optim: str, learning_rate: float, weight_decay: float,
                 inf_lr: float, n_steps_inf: int, label_dim: Union[int, Tuple[int]], loss_fn: str,
                 momentum: float, momentum_inf: float = 0.0):

        super().__init__(torch_model, optim, learning_rate, weight_decay, inf_lr,
                         n_steps_inf, label_dim, loss_fn, momentum, momentum_inf)

    def get_ini_labels(self, x: torch.Tensor):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = self._get_tensor_init_labels(x)
        y.requires_grad = True
        return y

    @staticmethod
    def _loop_inference(pred_energy, y_pred, optim_inf) -> torch.Tensor:
        # Max-margin surrogate objective (with E(ground_truth) missing)
        loss = pred_energy

        grad = torch.autograd.grad(loss, y_pred, grad_outputs=torch.ones_like(loss),
                                   only_inputs=True)
        y_grad = grad[0].detach()
        # Gradient descent to find the y_prediction that minimizes the energy
        y_pred = y_pred - optim_inf.update(y_grad)

        # Project back to the valid range
        y_pred = torch.clamp(y_pred, 0, 1)
        return y_pred

    @abstractmethod
    def inference(self, x, training: bool, n_steps: int):
        pass

    def train(self, loader: DataLoader):
        pass

    def valid(self, loader: DataLoader):
        pass

    def test(self, loader: DataLoader):
        return self.valid(loader)
