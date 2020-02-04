import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from typing import Optional
import argparse

from src.utils import create_path_that_doesnt_exist
from src.visualization_utils import plot_results
from src.model.deep_value_network import DeepValueNetwork
from src.image_segmentation.weizmann_horse_dataset import (
    PATH_DATA_WEIZMANN, PATH_SAVE_HORSE, load_test_set_horse, load_train_set_horse
)
from src.image_segmentation.utils import average_over_crops, show_preds_test_time
from src.image_segmentation.model.conv_net import ConvNet

__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"


class DVNHorse(DeepValueNetwork):

    def __init__(self, metric_optimize: str, optim: str, loss_fn: str,
                 mode_sampling=DeepValueNetwork.Sampling_GT, learning_rate=0.01,
                 momentum: float = 0, weight_decay=1e-3, inf_lr=50, momentum_inf=0,
                 n_steps_inf=30, label_dim=(24, 24)):
        # Deep Value Network is just a ConvNet
        model = ConvNet()

        super().__init__(model, metric_optimize, mode_sampling, optim, learning_rate, weight_decay,
                         inf_lr, n_steps_inf, label_dim, loss_fn, momentum, momentum_inf)

    def generate_output(self, x, training: bool, gt_labels: Optional[torch.Tensor] = None):
        """
        Generate an output y to compute
        the loss v(y, y*) --> we can use different
        techniques to generate the output
        1) Gradient based inference
        2) Simply add the ground truth outputs
        2) Generating adversarial tuples
        3) TODO: Stratified Sampling: Random samples from Y, biased towards y*
        """
        if self.using_adv_sampling() and training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels)
            pred_labels = self.inference(x, init_labels, training, gt_labels, self.n_steps_inf)
        elif self.using_gt_sampling() and training and np.random.rand() >= 0.5:
            # In training: If add_ground_truth=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            pred_labels = self.inference(x, init_labels, training, n_steps=self.n_steps_inf)

        return pred_labels.detach().clone()

    def test(self, loader: DataLoader, show_n_samples: int = 0):
        """
        At Test time, we are averaging our predictions
        over 36 crops of 24x24 mask to predict a 32x32 mask
        """
        self.model.eval()
        mean_iou = []
        n_shown = 0

        with torch.no_grad():
            for batch_idx, (raw_inputs, inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()

                # For test: inputs is a 5d tensor
                bs, ncrops, channels, h, w = inputs.size()

                pred_labels = self.generate_output(inputs.view(-1, channels, h, w), False)
                # fuse batch size and ncrops to know our estimated IOU
                # output = self.model(inputs.view(-1, channels, h, w), pred_labels)
                # go back to normal shape and take the mean in the 1 dim
                # output = output.view(bs, ncrops, 1).mean(1)
                pred_labels = pred_labels.view(bs, ncrops, h, w)

                final_pred = average_over_crops(pred_labels, self.device)
                oracle = self.oracle_value(final_pred, targets, False)
                for o in oracle:
                    mean_iou.append(o)

                # Visualization
                if n_shown < show_n_samples:
                    n_shown += 1
                    show_preds_test_time(raw_inputs, final_pred)

        mean_iou = torch.stack(mean_iou)
        mean_iou = torch.mean(mean_iou)
        mean_iou = mean_iou.cpu().numpy()

        print('Test set: IOU = {:.2f}%'.format(100 * mean_iou))

        return mean_iou


def run_the_model(dvn: DVNHorse, train_loader: DataLoader, valid_loader: DataLoader, path_save: str,
                  save_model: bool, n_epochs: int, step_size_scheduler: int, gamma_scheduler: float):
    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(dvn.optimizer, step_size=step_size_scheduler,
                                                gamma=gamma_scheduler)
    name = "dvn_whorse"
    loss_train, loss_valid = [], []
    list_iou_valid = []
    best_iou_valid = 0

    path_model_save = create_path_that_doesnt_exist(path_save, name, ".pth")

    for epoch in range(n_epochs):
        t_loss = dvn.train(train_loader)
        v_loss, iou_valid = dvn.valid(valid_loader)
        scheduler.step()
        loss_train.append(t_loss)
        loss_valid.append(v_loss)
        list_iou_valid.append(iou_valid)

        if epoch > 2 and save_model and iou_valid > best_iou_valid:
            best_iou_valid = iou_valid
            print('--- Saving model at IOU: {:.2f}% ---'.format(100 * best_iou_valid))
            torch.save(dvn.model.state_dict(), path_model_save)

    plot_results("IOU", loss_train, loss_valid, list_iou_valid, None)


def run_test_set(dvn: DVNHorse, valid_loader: DataLoader, path_best_model: str):
    """ Compute IOU on test set using 36 crops averaging """
    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dvn.model.load_state_dict(torch.load(path_best_model, map_location=device))
    dvn.model.eval()

    # Compute IOU single prediction on 24x24 crops and 36 crops averaging on 32x32
    for i in range(2):
        thirtysix_crops = False if i == 0 else True

        test_loader = load_test_set_horse(PATH_DATA_WEIZMANN, use_cuda,
                                          batch_size=8, thirtysix_crops=thirtysix_crops)
        print('-------------------------------------------')
        if i == 0:
            print('Single crop IOU prediction')
            dvn.valid(valid_loader)
        else:
            print('36 Crops IOU prediction')
            dvn.test(test_loader, show_n_samples=1)


def main():
    parser = argparse.ArgumentParser(description='DVN Img Segmentation')

    parser.add_argument('--train', type=bool, default=True,
                        help='If set to true train model, else show predictions/results on test set')

    parser.add_argument('--path_best_model', type=str, default=None,
                        help='If running on test set, need to provide path for model')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--batch_size_valid', type=int, default=8, help='batch size for evaluation')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')

    parser.add_argument('--optimizer', type=str, default="adam", help='adam or sgd optimizers')
    parser.add_argument('--lr', type=float, default=1e-4, help='SGD/Adam learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='SGD/Adam weight decay')

    parser.add_argument('--momentum_inf', type=float, default=0., help='SGD momentum for inference')
    parser.add_argument('--inf_lr', type=float, default=5e3, help='learning rate for inference')
    parser.add_argument('--n_steps_inf', type=int, default=20, help='number of inference steps/iterations')

    parser.add_argument('--mode_sampling', type=str, default=DeepValueNetwork.Sampling_GT,
                        help='Sampling method for DVN (Ground truth, Adversarial, Stratified)')

    parser.add_argument('--step_size_scheduler', type=int, default=100,
                        help='Number of iterations to reduce learning rate')
    parser.add_argument('--gamma_scheduler', type=float, default=0.1,
                        help='Multiply the learning rate by this float at each *step_size_scheduler* epochs')

    args = parser.parse_args()

    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    train_loader, valid_loader = load_train_set_horse(
        PATH_DATA_WEIZMANN, use_cuda, args.batch_size, args.batch_size_valid
    )

    dvn = DVNHorse(
        "iou", args.optimizer, "bce", args.mode_sampling, args.lr, args.momentum,
        args.weight_decay, args.inf_lr, args.momentum_inf, args.n_steps_inf
    )

    if args.train:
        run_the_model(
            dvn, train_loader, valid_loader, PATH_SAVE_HORSE, True, args.epochs,
            args.step_size_scheduler, args.gamma_scheduler
        )
    else:
        if args.path_best_model is None or not os.path.exists(args.path_best_model):
            raise ValueError(f"Invalid path_best_model provided = {args.path_best_model}")

        run_test_set(dvn, valid_loader, args.path_best_model)


if __name__ == "__main__":
    main()
