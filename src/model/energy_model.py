import torch
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

        self.new_ep = True

        self.label_dim = label_dim
        self.training = False

        if optim.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                             momentum=momentum, weight_decay=weight_decay)
        elif optim.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Invalid optimizer provided: {optim}')

    def train(self, loader):
        pass

    def valid(self, loader):
        pass

    def test(self, loader):
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

    def _adjust_labels(self, pred_labels, gt_labels):
        if pred_labels.shape != gt_labels.shape:
            raise ValueError('Invalid labels shape: gt = ', gt_labels.shape, 'pred = ', pred_labels.shape)

        if not self.training:
            # No relaxation, 0-1 only
            pred_labels = torch.where(pred_labels >= 0.5,
                                      torch.ones(1).to(self.device),
                                      torch.zeros(1).to(self.device))
            pred_labels = pred_labels.float()
        return pred_labels, gt_labels

    def _f1_score(self, pred_labels, gt_labels):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the F1 Score when training.
        and the discrete F1 when validating/testing
        """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels)

        intersect = torch.sum(torch.min(pred_labels, gt_labels), dim=1)
        union = torch.sum(torch.max(pred_labels, gt_labels), dim=1)

        # for numerical stability
        epsilon = torch.full(union.size(), 10 ** -8).to(self.device)

        # Add epsilon also in numerator since some images have 0 tags
        # and then we get 0/0 --> = nan instead of f1=1
        f1 = (2 * intersect + epsilon) / (intersect + torch.max(epsilon, union))
        # we want a (Batch_size x 1) tensor
        f1 = f1.view(-1, 1)
        return f1

    def _iou_score(self, pred_labels, gt_labels):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the IOU (intersection
        over union) when training, and the discrete IOU
        when validating/testing
        """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels)

        pred_labels = torch.flatten(pred_labels).reshape(pred_labels.size()[0], -1)
        gt_labels = torch.flatten(gt_labels).reshape(gt_labels.size()[0], -1)

        intersect = torch.sum(torch.min(pred_labels, gt_labels), dim=1)
        union = torch.sum(torch.max(pred_labels, gt_labels), dim=1)

        # for numerical stability
        epsilon = torch.full(union.size(), 10 ** -8).to(self.device)

        iou = intersect / torch.max(epsilon, union)
        # we want a (Batch_size x 1) tensor
        iou = iou.view(-1, 1)
        return iou

    def _scaled_hamming_loss(self, pred_labels, gt_labels):
        """ Scaled Hamming Loss """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels)
        # Hamming Loss in 0-1
        loss = torch.sum(torch.abs(gt_labels - pred_labels), dim=1)
        if self.use_bce:
            # TODO: Change this constant
            loss /= 24.
        loss = loss.view(-1, 1)
        return loss
