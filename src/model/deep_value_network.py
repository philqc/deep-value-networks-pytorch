import torch
from abc import ABC, abstractmethod
import time
from typing import Union, Tuple
from .base_model import BaseModel
from src.utils import SGD, Sampling


class DeepValueNetwork(BaseModel):

    def __init__(self, torch_model, metric_optimize: str, use_cuda: bool, mode_sampling: str, optim: str,
                 learning_rate: float, weight_decay: float, inf_lr: float, n_steps_inf: int,
                 label_dim: Union[int, Tuple[int]], loss_fn: str, momentum: float, momentum_inf: float = 0.0):

        self.mode_sampling = mode_sampling
        if self.mode_sampling == Sampling.STRAT:
            raise NotImplementedError('Stratified sampling is not yet implemented!')
        elif self.mode_sampling in [Sampling.GT, Sampling.ADV]:
            print(f'Using {self.mode_sampling} Sampling')
        else:
            raise ValueError(f'Invalid sampling strategy = {self.mode_sampling}')

        self.use_hamming_metric = False
        if metric_optimize.lower() == "f1":
            self.score_str = "F1"
            self.oracle_value = lambda x, y: self._f1_score(x, y)
        elif metric_optimize.lower() == "hamming":
            self.score_str = "Hamming"
            self.use_hamming_metric = True
            self.oracle_value = lambda x, y: self._scaled_hamming_loss(x, y)
        elif metric_optimize.lower() == "iou":
            self.score_str = "IOU"
            self.oracle_value = lambda x, y: self._iou_score(x, y)
        else:
            raise ValueError(f"Invalid metric_optimize provided = {metric_optimize}")

        if loss_fn.lower() == "bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.use_bce = True
            print('Using Binary Cross Entropy with Logits Loss')
        elif loss_fn.lower() == "mse":
            self.loss_fn = torch.nn.MSELoss()
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
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.label_dim = label_dim
        self.training = False

        # Model and optimizer
        self.model = torch_model.to(self.device)
        if optim.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                             momentum=momentum, weight_decay=weight_decay)
        elif optim.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Invalid optimizer provided: {optim}')

    @abstractmethod
    def generate_output(self, x, gt_labels):
        pass

    def _get_tensor_init_labels(self, x):
        if isinstance(self.label_dim, Tuple):
            # image or other 2D input
            return torch.zeros(x.size()[0], 1, self.label_dim[0], self.label_dim[1],
                               dtype=torch.float32, device=self.device)
        else:
            # 1D input
            return torch.zeros(x.size()[0], self.label_dim, dtype=torch.float32, device=self.device)

    def get_ini_labels(self, x, gt_labels=None):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = self._get_tensor_init_labels(x)

        if gt_labels is not None:
            # 50% Start from GT; rest: start from zeros
            gt_indices = torch.rand(gt_labels.shape[0]).float().to(self.device) > 0.5
            y[gt_indices] = gt_labels[gt_indices]

        # Set requires_grad=True after in_place operation (changing the indices)
        y.requires_grad = True
        return y

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

    def _loop_inference(self, gt_labels, x, y, optim_inf) -> torch.Tensor:
        if gt_labels is not None:  # Adversarial
            output = self.model(x, y)
            oracle = self.oracle_value(y, gt_labels)
            # this is the BCE loss with logits
            value = self.loss_fn(output, oracle)
        else:
            output = self.model(x, y)
            value = torch.sigmoid(output)

        grad = torch.autograd.grad(value, y, grad_outputs=torch.ones_like(value), only_inputs=True)

        y_grad = grad[0].detach()

        if gt_labels is None and self.use_hamming_metric:
            # We want to reduce !! the Hamming loss in this case
            y = y - optim_inf.update(y_grad)
        else:
            y = y + optim_inf.update(y_grad)

        y = y + optim_inf.update(y_grad)
        # Project back to the valid range
        y = torch.clamp(y, 0, 1)
        return y

    @abstractmethod
    def inference(self, x, y, gt_labels=None, n_steps=20):
        pass

    def train(self, loader):

        self.model.train()
        self.training = True
        n_train = len(loader.dataset)

        time_start = time.time()
        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels = self.generate_output(inputs, targets)
            output = self.model(inputs, pred_labels)
            oracle = self.oracle_value(pred_labels, targets)
            loss = self.loss_fn(output, oracle)

            if torch.isnan(loss):
                print('Loss has NaN! Loss={:.5f}'.format(loss.item()))
                raise ValueError('Loss has Nan')

            t_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; Avg_Loss = {:.5f}; '
                      'Pred_{} = {:.2f}%; Real_{} = {:.2f}%'
                      ''.format(t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size,
                                self.score_str, 100 * torch.sigmoid(output).mean(),
                                self.score_str, 100 * oracle.mean()),
                      end='')

        t_loss /= t_size
        self.training = False
        print('')
        return t_loss

    def valid(self, loader):

        self.model.eval()
        self.training = False
        self.new_ep = True
        loss, t_size = 0, 0
        mean_oracle = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels = self.generate_output(inputs, gt_labels=None)
                output = self.model(inputs, pred_labels)
                oracle = self.oracle_value(pred_labels, targets)

                loss += self.loss_fn(output, oracle)
                for o in oracle:
                    mean_oracle.append(o)
                self.new_ep = False

        mean_oracle = torch.stack(mean_oracle)
        mean_oracle = torch.mean(mean_oracle)
        mean_oracle = mean_oracle.cpu().numpy()
        loss /= t_size

        print('Validation: Loss = {:.5f}; Pred_{} = {:.2f}%, Real_{} = {:.2f}%'
              ''.format(loss.item(), self.score_str, 100 * torch.sigmoid(output).mean(),
                        self.score_str, 100 * mean_oracle))

        return loss.item(), mean_oracle

    @abstractmethod
    def test(self, loader):
        pass
