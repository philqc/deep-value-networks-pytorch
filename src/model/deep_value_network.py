import torch
from abc import ABC, abstractmethod
import time
from typing import Union, Tuple
from .base_model import BaseModel
from src.utils import SGD, Sampling


class DeepValueNetwork(BaseModel):

    def __init__(self, torch_model, metric_optimize: str, use_cuda: bool, mode_sampling: str, optim: str,
                 learning_rate: float, weight_decay: float, inf_lr: float, n_steps_inf: int,
                 label_dim: Union[int, Tuple[int]], loss_fn: str):
        super().__init__(metric_optimize, use_cuda, label_dim)

        self.model = torch_model.to(self.device)

        self.mode_sampling = mode_sampling
        if self.mode_sampling == Sampling.STRAT:
            raise ValueError('Stratified sampling is not yet implemented!')

        if self.loss_fn.lower() == "bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.loss_fn.lower() == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss_fn provided = {loss_fn}")

        if optim.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Invalid optimizer provided: {optim}')

        self.inf_lr = inf_lr
        # monitor norm gradients
        self.norm_gradient_inf = [[] for _ in range(n_steps_inf)]
        self.norm_gradient_adversarial = [[] for _ in range(n_steps_inf)]

    @abstractmethod
    def generate_output(self, x, gt_labels):
        pass

    def _get_tensor_init_labels(self, x):
        if len(self.label_dim) == 2:
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

    def inference(self, x, y, gt_labels=None, num_iterations=20):

        if self.training:
            self.model.eval()

        optim_inf = SGD(y, lr=self.inf_lr, momentum=0)

        with torch.enable_grad():

            for i in range(num_iterations):

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
                y = y + optim_inf.update(y_grad)
                # Project back to the valid range
                y = torch.clamp(y, 0, 1)

                if gt_labels is not None:  # Adversarial
                    self.norm_gradient_adversarial[i].append(y_grad.norm())
                else:
                    self.norm_gradient_inf[i].append(y_grad.norm())

        if self.training:
            self.model.train()

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
        mean_iou = []

        with torch.no_grad():
            for (raw_inputs, inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels = self.generate_output(inputs, gt_labels=None)
                output = self.model(inputs, pred_labels)
                oracle = self.oracle_value(pred_labels, targets)

                loss += self.loss_fn(output, oracle)
                for o in oracle:
                    mean_iou.append(o)
                self.new_ep = False

        mean_iou = torch.stack(mean_iou)
        mean_iou = torch.mean(mean_iou)
        mean_iou = mean_iou.cpu().numpy()
        loss /= t_size

        print('Validation: Loss = {:.5f}; Pred_{} = {:.2f}%, Real_{} = {:.2f}%'
              ''.format(loss.item(), self.score_str, 100 * torch.sigmoid(output).mean(),
                        self.score_str, 100 * mean_iou))

        return loss.item(), mean_iou
