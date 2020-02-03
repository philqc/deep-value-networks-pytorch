import torch
from torch.utils.data import DataLoader
from abc import abstractmethod
import time
from typing import Union, Tuple, Optional
from src.utils import SGD
from .energy_model import EnergyModel


class DeepValueNetwork(EnergyModel):

    Sampling_GT = "ground_truth"
    Sampling_Adv = "adversarial"
    Sampling_Strat = "stratified"

    def __init__(self, torch_model, metric_optimize: str, mode_sampling: str, optim: str,
                 learning_rate: float, weight_decay: float, inf_lr: float, n_steps_inf: int,
                 label_dim: Union[int, Tuple[int]], loss_fn: str, momentum: float = 0., momentum_inf: float = 0.):

        super().__init__(torch_model, optim, learning_rate, weight_decay, inf_lr,
                         n_steps_inf, label_dim, loss_fn, momentum, momentum_inf)

        self.mode_sampling = mode_sampling
        if mode_sampling == self.Sampling_Strat:
            raise NotImplementedError('Stratified sampling is not yet implemented!')
        elif mode_sampling in [self.Sampling_GT, self.Sampling_Adv]:
            print(f'Using {mode_sampling} sampling')
        else:
            raise ValueError(f'Invalid sampling strategy = {mode_sampling}')

        self.use_hamming_metric = False
        if metric_optimize.lower() == "f1":
            self.score_str = "F1"
            self.oracle_value = lambda x, y, z: self._f1_score(x, y, z)
        elif metric_optimize.lower() == "hamming":
            self.score_str = "Hamming"
            self.use_hamming_metric = True
            self.oracle_value = lambda x, y, z: self._scaled_hamming_loss(x, y, z)
        elif metric_optimize.lower() == "iou":
            self.score_str = "IOU"
            self.oracle_value = lambda x, y, z: self._iou_score(x, y, z)
        else:
            raise ValueError(f"Invalid metric_optimize provided = {metric_optimize}")

    @abstractmethod
    def generate_output(self, x: torch.Tensor, training: bool, gt_labels: Optional[torch.Tensor] = None):
        pass

    def get_ini_labels(self, x: torch.Tensor, gt_labels: Optional[torch.Tensor] = None):
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

    def _loop_inference(
            self, gt_labels: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
            optim_inf: torch.optim, training: bool
    ) -> torch.Tensor:
        if gt_labels is not None:  # Adversarial
            output = self.model(x, y)
            oracle = self.oracle_value(y, gt_labels, training)
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

    def inference(
            self, x: torch.Tensor, y: torch.Tensor, training: bool,
            gt_labels: Optional[torch.Tensor] = None, n_steps=20
    ) -> torch.Tensor:
        if training:
            self.model.eval()

        optim_inf = SGD(y, lr=self.inf_lr, momentum=self.momentum_inf)

        with torch.enable_grad():
            for i in range(n_steps):
                y = self._loop_inference(gt_labels, x, y, optim_inf, training)

        if training:
            self.model.train()

        return y

    def train(self, loader: DataLoader):

        self.model.train()
        n_train = len(loader.dataset)

        time_start = time.time()
        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels = self.generate_output(inputs, True, targets)
            output = self.model(inputs, pred_labels)
            oracle = self.oracle_value(pred_labels, targets, True)
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
        print('')
        return t_loss

    def valid(self, loader: DataLoader):

        self.model.eval()
        loss, t_size = 0, 0
        mean_oracle = []

        with torch.no_grad():
            for (inputs, targets) in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels = self.generate_output(inputs, False)
                output = self.model(inputs, pred_labels)
                oracle = self.oracle_value(pred_labels, targets, False)

                loss += self.loss_fn(output, oracle)
                for o in oracle:
                    mean_oracle.append(o)

        mean_oracle = torch.stack(mean_oracle)
        mean_oracle = torch.mean(mean_oracle)
        mean_oracle = mean_oracle.cpu().numpy()
        loss /= t_size

        print('Validation: Loss = {:.5f}; Pred_{} = {:.2f}%, Real_{} = {:.2f}%'
              ''.format(loss.item(), self.score_str, 100 * torch.sigmoid(output).mean(),
                        self.score_str, 100 * mean_oracle))

        return loss.item(), mean_oracle

    def test(self, loader: DataLoader):
        return self.valid(loader)

    def using_adv_sampling(self):
        return self.mode_sampling == self.Sampling_Adv

    def using_gt_sampling(self):
        return self.mode_sampling == self.Sampling_GT

    def using_strat_sampling(self):
        return self.mode_sampling == self.Sampling_Strat
