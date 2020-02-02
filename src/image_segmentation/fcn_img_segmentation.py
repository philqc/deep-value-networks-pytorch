from __future__ import print_function, division
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
import warnings

from src.image_segmentation.utils import average_over_crops
from src.visualization_utils import show_grid_imgs
from .utils import get_iou_batch
from .weizmann_horse_dataset import WeizmannHorseDataset


warnings.filterwarnings("ignore")
__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)


class FCN(nn.Module):

    # define each layer of neural network
    def __init__(self):
        super(FCN, self). __init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 64, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)
        # Deconvolution
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #                    output_padding=0, groups=1, bias=True, dilation=1)
        self.deconv1 = nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x


def train(args, model, device, train_loader, optimizer, epochs):
    model.train()
    train_loss = 0
    total_iou = 0
    # define the operation batch-wise
    for batch_idx, (raw_inputs, data, target) in enumerate(train_loader):
        # send the data into GPU or CPU
        data, target = data.to(device), target.to(device)
        target[target > 0] = 1

        # clear the gradient in the optimizer in the begining of each backpropagation
        optimizer.zero_grad()
        # get out
        output = model(data)
        # define loss
        n, c, h, w = output.size()
        log_p = F.log_softmax(output, dim=1)

        # prediction (pick higher probability after log softmax)
        pred = torch.argmax(log_p, dim=1)

        total_iou += torch.mean(get_iou_batch(pred.detach(), target.long()))
        # adapted from cross_entropy_2d
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3)
        target = target.transpose(1, 2).transpose(2, 3)
        log_p = log_p[target.repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]

        loss = F.nll_loss(log_p, target.long())
        # do backprobagation to get gradient
        loss.backward()
        # update the parameters
        optimizer.step()
        # train loss
        train_loss += loss.item()

    return train_loss/len(train_loader), total_iou/len(train_loader)


def visualize(iou, data, target, pred):
    i = iou.max(0)[1].item()
    plt.figure()
    print(iou[i])
    plt.imshow(data[i].to('cpu').numpy().transpose((1, 2, 0)))
    plt.figure()
    plt.imshow(np.squeeze(target[i].to('cpu').numpy().transpose((1, 2, 0))))
    plt.figure()
    plt.imshow(pred[i].to('cpu').numpy())


def valid(args, model, device, test_loader, epochs):
    model.eval()
    test_loss = 0
    total_iou = 0

    with torch.no_grad():
        for (raw_inputs, data, target) in test_loader:
            data, target = data.to(device), target.to(device)

            target[target > 0] = 1
            output = model(data)

            # prediction (pick higher probability after log softmax)
            n, c, h, w = output.size()
            log_p = F.log_softmax(output, dim=1)
            pred = torch.argmax(log_p, dim=1)
            iou = get_iou_batch(pred, target.long())
            total_iou += torch.mean(iou)

            # if epochs % 25== 0:
            #    visualize(iou, data, pred, target)

            log_p = log_p.transpose(1, 2).transpose(2, 3)
            target = target.transpose(1, 2).transpose(2, 3)
            log_p = log_p[target.repeat(1, 1, 1, c) >= 0]
            log_p = log_p.view(-1, c)
            mask = target >= 0
            target = target[mask]

            loss = F.nll_loss(log_p, target.long())
            # Average the loss (batch_wise)
            test_loss += loss.item()

    return test_loss/len(test_loader), total_iou/len(test_loader)


def test(model, loader, device):
    """
    At Test time, we are averaging our predictions
    over 36 crops of 24x24 mask to predict a 32x32 mask
    """
    model.eval()
    mean_iou = []

    with torch.no_grad():
        for batch_idx, (raw_inputs, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # For test: inputs is a 5d tensor
            bs, ncrops, channels, h, w = inputs.size()

            # fuse batch size and ncrops to know our estimated IOU
            output = model(inputs.view(-1, channels, h, w))
            log_p = F.log_softmax(output, dim=1)
            pred = torch.argmax(log_p, dim=1)
            # go back to normal shape and take the mean in the 1 dim
            pred = pred.view(bs, ncrops, h, w)
            #                output = output.view(bs, ncrops, h, w)

            final_pred = average_over_crops(pred, device)
            oracle = get_iou_batch(final_pred, targets.float())
            for o in oracle:
                mean_iou.append(o)

            print('------ Test: IOU = {:.2f}% ------'.format(100 * oracle.mean()))
            img = raw_inputs.detach().cpu()
            show_grid_imgs(img)
            mask = final_pred.detach().cpu()
            show_grid_imgs(mask.float())
            print('Mask binary')
            bin_mask = mask >= 0.50
            show_grid_imgs(bin_mask.float())
            print('---------------------------------------')

    mean_iou = torch.stack(mean_iou)
    mean_iou = torch.mean(mean_iou)
    mean_iou = mean_iou.cpu().numpy()

    print('Test set: IOU = {:.2f}%'.format(100 * mean_iou))

    return mean_iou


def main():
    # Version of Pytorch
    print("Pytorch Version:", torch.__version__)

    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # root directory of dataset
    image_dir = '../../data/weizmann_horse/images'
    mask_dir = '../../data/weizmann_horse/masks'

    # Create FCN
    model = FCN().to(device)
    # print the model summery
    print(model)

    # Visualize the output of each layer via torchSummary
    summary(model, input_size=(3, 24, 24))

    batch_size = args.batch_size
    batch_size_valid = batch_size

    train_set = WeizmannHorseDataset(image_dir, mask_dir, subset='train',
                                     random_mirroring=False, thirty_six_cropping=False)
    valid_set = WeizmannHorseDataset(image_dir, mask_dir, subset='valid',
                                     random_mirroring=False, thirty_six_cropping=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_valid
    )

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_loss = []
    train_iou = []
    validation_loss = []
    validation_iou = []
    best_val_iou = 0

    start_time = time.time()
    # Start Training
    for epoch in range(1, args.epochs + 1):
        # train loss
        t_loss, t_mean_iou = train(args, model, device, train_loader, optimizer, epoch)
        print('Train Epoch: {} \t Loss: {:.6f}\t Mean_IOU(%):{}%'.format(
            epoch, t_loss, t_mean_iou))

        # validation loss
        v_loss, v_mean_iou = valid(args, model, device, valid_loader, epoch)
        print('Validation Epoch: {} \t Loss: {:.6f}\t Mean_IOU(%):{}%'.format(
            epoch, v_loss, v_mean_iou))

        train_loss.append(t_loss)
        train_iou.append(t_mean_iou)
        validation_loss.append(v_loss)
        validation_iou.append(v_mean_iou)
        if v_mean_iou > best_val_iou:
            best_val_iou = v_mean_iou
            print('--- Saving model at IOU_{:.2f} ---'.format(100 * best_val_iou))
            torch.save(model.state_dict(), 'FCN_best.pth')

        print('-------------------------------------------------------')

    print("--- %s seconds ---" % (time.time() - start_time))
    print("training:", len(train_loader))
    print("validation:", len(valid_loader))
    x = list(range(1, args.epochs + 1))
    # plot train/validation loss versus epoch
    plt.figure()
    plt.title("Train/Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.plot(x, train_loss, label="train loss")
    plt.plot(x, validation_loss, color='red', label="validation loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # plot train/validation loss versus epoch
    plt.figure()
    plt.title("Train/Validation IOU")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IOU")
    plt.plot(x, train_iou, label="train iou")
    plt.plot(x, validation_iou, color='red', label="validation iou")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # test set
    print("Best Validation Mean IOU:", best_val_iou)

    # Test on 36 crop
    fcn_test = FCN().to(device)
    fcn_test.load_state_dict(torch.load('FCN_best.pth'))
    fcn_test.eval()

    # Compute IOU single prediction on 24x24 crops and 36 crops averaging on 32x32
    for i in range(2):
        thirtysix_crops = False if i == 0 else True
        test_set = WeizmannHorseDataset(image_dir, mask_dir, subset='test',
                                        random_mirroring=False, thirty_six_cropping=thirtysix_crops)

        batch_size_eval = 8

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size_eval,
        )

        print('-------------------------------------------')
        if i == 0:
            mean_iou = 0
            print('Single crop IOU prediction')
            for epoch in range(1, 100):
                # validation loss
                v_loss, v_mean_iou = valid(args, model, device, valid_loader, epoch)
                mean_iou += v_mean_iou
            print('Validation Mean_IOU(%):{}%'.format(mean_iou / 100))
        else:
            print('36 Crops IOU prediction')
            test(fcn_test, test_loader, device)


if __name__ == "__main__":
    main()
