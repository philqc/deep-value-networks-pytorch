import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import os

from src.image_segmentation.weizmann_horse_dataset import (
    PATH_DATA_WEIZMANN, load_train_set_horse, load_test_set_horse, PATH_SAVE_HORSE
)
from src.image_segmentation.utils import show_preds_test_time, get_iou_batch
from src.image_segmentation.model.fcn import FCN
from src.model.base_model import BaseModel
from src.visualization_utils import plot_results

__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"

FILE_SAVE_FCN = "fcn_best.pth"


class FCNModel(BaseModel):

    def __init__(self, lr: float, momentum: float):
        super().__init__(model=FCN())
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    @staticmethod
    def _compute_pred_and_loss(output, target):
        n, c, h, w = output.size()
        log_p = F.log_softmax(output, dim=1)
        # prediction (pick higher probability after log softmax)
        pred = torch.argmax(log_p, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3)
        target = target.transpose(1, 2).transpose(2, 3)
        log_p = log_p[target.repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]

        loss = F.nll_loss(log_p, target.long())
        return pred, loss

    def train(self, loader: DataLoader):
        self.model.train()
        train_loss = 0
        total_iou = 0
        # define the operation batch-wise
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets[targets > 0] = 1

            # clear the gradient in the optimizer in the begining of each backpropagation
            self.optimizer.zero_grad()

            output = self.model(inputs)

            pred, loss = self._compute_pred_and_loss(output, targets)

            total_iou += torch.mean(get_iou_batch(pred.detach(), targets.long()))

            loss.backward()
            # update the parameters
            self.optimizer.step()
            # train loss
            train_loss += loss.item()

        return train_loss / len(loader), total_iou / len(loader)

    def valid(self, loader: DataLoader):
        self.model.eval()
        test_loss = 0
        total_iou = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                targets[targets > 0] = 1
                output = self.model(inputs)

                pred, loss = self._compute_pred_and_loss(output, targets)

                iou = get_iou_batch(pred, targets.long())
                total_iou += torch.mean(iou)

                # if epochs % 25== 0:
                #    visualize(iou, inputs, pred, targets)

                # Average the loss (batch_wise)
                test_loss += loss.item()

        return test_loss / len(loader), total_iou / len(loader)

    def test(self, loader: DataLoader, show_n_samples: int = 0):
        self.model.eval()
        total_iou = 0
        test_loss = 0
        n_shown = 0
        with torch.no_grad():
            for raw_inputs, inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets[targets > 0] = 1
                output = self.model(inputs)
                n, c, h, w = output.size()

                pred, loss = self._compute_pred_and_loss(output, targets)

                iou = get_iou_batch(pred, targets.long())
                total_iou += torch.mean(iou)
                test_loss += loss.item()

                if n_shown < show_n_samples:
                    n_shown += 1
                    pred = pred.view(n, 1, h, w)
                    show_preds_test_time(raw_inputs, pred)

        print('Test set: IOU = {:.2f}%'.format(100 * total_iou / len(loader)))
        return test_loss / len(loader), total_iou / len(loader)

    @staticmethod
    def _visualize(iou, inputs, targets, pred):
        i = iou.max(0)[1].item()
        plt.figure()
        plt.imshow(inputs[i].to('cpu').numpy().transpose((1, 2, 0)))
        plt.figure()
        plt.imshow(np.squeeze(targets[i].to('cpu').numpy().transpose((1, 2, 0))))
        plt.figure()
        plt.imshow(pred[i].to('cpu').numpy())


def run_the_model(fcn: FCNModel, path_save_model: str, use_cuda: bool,
                  epochs: int, batch_size: int):

    train_loader, valid_loader = load_train_set_horse(
        PATH_DATA_WEIZMANN, use_cuda, batch_size, batch_size
    )

    train_loss, validation_loss = [], []
    train_iou, validation_iou = [], []
    best_val_iou = 0

    start_time = time.time()
    # Start Training
    for epoch in range(1, epochs + 1):
        # train loss
        t_loss, t_mean_iou = fcn.train(train_loader)
        print('Train Epoch: {} \t Loss: {:.6f}\t Mean_IOU: {:.2f}%'.format(
            epoch, t_loss, 100 * t_mean_iou))

        # validation loss
        v_loss, v_mean_iou = fcn.valid(valid_loader)
        print('Validation Epoch: {} \t Loss: {:.6f}\t Mean_IOU: {:.2f}%'.format(
            epoch, v_loss, 100 * v_mean_iou))

        train_loss.append(t_loss)
        train_iou.append(t_mean_iou)
        validation_loss.append(v_loss)
        validation_iou.append(v_mean_iou)
        if v_mean_iou > best_val_iou:
            best_val_iou = v_mean_iou
            print('--- Saving model at IOU: {:.2f}% ---'.format(100 * best_val_iou))
            torch.save(fcn.model.state_dict(), path_save_model)

        print('-------------------------------------------------------')

    print("--- {:.2f} seconds ---".format(time.time() - start_time))

    # Visualization
    plot_results("IOU", train_loss, validation_loss, validation_iou, train_iou)

    print("Best Validation Mean IOU: {:.2f}%".format(100 * best_val_iou))


def run_on_test_set(fcn: FCNModel, path_save_model: str, use_cuda: bool):
    device = torch.device("cuda" if use_cuda else "cpu")
    fcn.model.load_state_dict(torch.load(path_save_model, map_location=device))
    test_loader = load_test_set_horse(PATH_DATA_WEIZMANN, use_cuda,
                                      batch_size=8, thirtysix_crops=False)
    fcn.test(test_loader, show_n_samples=1)


def main():
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')

    parser.add_argument('--train', type=bool, default=False, help='to train or tu run model on test set')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=256, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')

    args = parser.parse_args()

    path_save_model = os.path.join(PATH_SAVE_HORSE, FILE_SAVE_FCN)
    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    fcn = FCNModel(args.lr, args.momentum)
    # print the model summary
    print(fcn.model)

    # Visualize the output of each layer via torchSummary
    summary(fcn.model, input_size=(3, 24, 24))

    if args.train:
        run_the_model(fcn, path_save_model, use_cuda, args.epochs, args.batch_size)
    else:
        run_on_test_set(fcn, path_save_model, use_cuda)


if __name__ == "__main__":
    main()
