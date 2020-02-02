from __future__ import print_function, division
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import warnings

from src.image_segmentation.utils import average_over_crops, show_preds_test_time
from .utils import get_iou_batch, load_train_set_horse, load_test_set_horse, PATH_DATA_WEIZMANN
from .model.fcn import FCN

warnings.filterwarnings("ignore")
__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"

FILE_BEST_MODEL = "fcn_best.pth"


def train(model, device, train_loader, optimizer):
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

        output = model(data)

        pred, loss = compute_pred_and_loss(output, target)

        total_iou += torch.mean(get_iou_batch(pred.detach(), target.long()))

        loss.backward()
        # update the parameters
        optimizer.step()
        # train loss
        train_loss += loss.item()

    return train_loss/len(train_loader), total_iou/len(train_loader)


def compute_pred_and_loss(output, target):
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


def visualize(iou, data, target, pred):
    i = iou.max(0)[1].item()
    plt.figure()
    print(iou[i])
    plt.imshow(data[i].to('cpu').numpy().transpose((1, 2, 0)))
    plt.figure()
    plt.imshow(np.squeeze(target[i].to('cpu').numpy().transpose((1, 2, 0))))
    plt.figure()
    plt.imshow(pred[i].to('cpu').numpy())


def valid(model, device, test_loader):
    model.eval()
    test_loss = 0
    total_iou = 0

    with torch.no_grad():
        for (raw_inputs, data, target) in test_loader:
            data, target = data.to(device), target.to(device)

            target[target > 0] = 1
            output = model(data)

            pred, loss = compute_pred_and_loss(output, target)

            iou = get_iou_batch(pred, target.long())
            total_iou += torch.mean(iou)

            # if epochs % 25== 0:
            #    visualize(iou, data, pred, target)

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

            show_preds_test_time(raw_inputs, final_pred, oracle)

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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create FCN
    model = FCN().to(device)
    # print the model summary
    print(model)

    # Visualize the output of each layer via torchSummary
    summary(model, input_size=(3, 24, 24))

    train_loader, valid_loader = load_train_set_horse(PATH_DATA_WEIZMANN, use_cuda,
                                                      args.batch_size, args.batch_size)

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
        t_loss, t_mean_iou = train(model, device, train_loader, optimizer)
        print('Train Epoch: {} \t Loss: {:.6f}\t Mean_IOU(%):{}%'.format(
            epoch, t_loss, t_mean_iou))

        # validation loss
        v_loss, v_mean_iou = valid(model, device, valid_loader)
        print('Validation Epoch: {} \t Loss: {:.6f}\t Mean_IOU(%):{}%'.format(
            epoch, v_loss, v_mean_iou))

        train_loss.append(t_loss)
        train_iou.append(t_mean_iou)
        validation_loss.append(v_loss)
        validation_iou.append(v_mean_iou)
        if v_mean_iou > best_val_iou:
            best_val_iou = v_mean_iou
            print('--- Saving model at IOU_{:.2f} ---'.format(100 * best_val_iou))
            torch.save(model.state_dict(), FILE_BEST_MODEL)

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
    fcn_test.load_state_dict(torch.load(FILE_BEST_MODEL))
    fcn_test.eval()

    # Compute IOU single prediction on 24x24 crops and 36 crops averaging on 32x32
    for i in range(2):
        thirtysix_crops = False if i == 0 else True

        test_loader = load_test_set_horse(PATH_DATA_WEIZMANN, use_cuda,
                                          batch_size=8, thirtysix_crops=thirtysix_crops)

        print('-------------------------------------------')
        if i == 0:
            mean_iou = 0
            print('Single crop IOU prediction')
            for epoch in range(1, 100):
                # validation loss
                v_loss, v_mean_iou = valid(model, device, valid_loader)
                mean_iou += v_mean_iou
            print('Validation Mean_IOU(%):{}%'.format(mean_iou / 100))
        else:
            print('36 Crops IOU prediction')
            test(fcn_test, test_loader, device)


if __name__ == "__main__":
    main()
