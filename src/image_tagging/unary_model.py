import numpy as np
import os
import random
import torch
import torch.nn as nn
import torchvision.models
from torch.utils.data import DataLoader
import time
import pickle
from src.image_tagging.utils import calculate_hamming_loss
from .load_flickr import FlickrTaggingDataset, show_pred_labels, inv_normalize
from src.visualization_utils import show_grid_imgs, plot_results
from src.utils import project_root, create_path_that_doesnt_exist

PATH_PREPROCESSED = os.path.join(project_root(), "saved_results",
                                 "image_segmentation", "preprocessed")


class ConvNet(nn.Module):
    def __init__(self, label_dim):
        super().__init__()

        # Load pretrained AlexNet on ImageNet
        self.unary_model = torchvision.models.alexnet(pretrained=True)

        # Replace the last FC layer
        tmp = list(self.unary_model.classifier)
        tmp[-1] = nn.Linear(4096, label_dim)
        self.unary_model.classifier = nn.Sequential(*tmp)

    def forward(self, x):
        # send image through AlexNet
        x = self.unary_model(x)
        return torch.sigmoid(x)


class BaselineNetwork:

    def __init__(self, use_cuda, learning_rate=0.01, weight_decay=1e-4, label_dim=24):
        """
        Parameters
        ----------
        use_cuda: boolean
            true if we are using gpu, false if using cpu
        """
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.label_dim = label_dim
        self.new_ep = True

        self.model = ConvNet(label_dim).to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.training = False

    def train(self, loader, ep):

        self.model.train()
        self.training = True
        self.new_ep = True
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, t_size = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            output = self.model(inputs)
            loss = calculate_hamming_loss(output, targets)
            t_loss += loss.detach().clone().item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; Avg_Hamming_Loss = {:.5f};'
                      ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size),
                      end='')

            self.new_ep = False

        t_loss /= n_train
        self.training = False
        print('')
        return t_loss

    def valid(self, loader, ep):

        self.model.eval()
        self.training = False
        self.new_ep = True
        loss, t_size = 0, 0
        mean_f1 = []
        int_show = random.randint(0, 20)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                output = self.model(inputs)
                loss += calculate_hamming_loss(output, targets)

                self.new_ep = False
                f1 = self.get_f1_score(output, targets)
                for f in f1:
                    mean_f1.append(f)

                if batch_idx == int_show:
                    idx = np.random.choice(np.arange(len(inputs)), 3, replace=False)
                    inputs_unnormalized = inputs[idx].cpu()
                    inputs_unnormalized = [inv_normalize(i) for i in inputs_unnormalized]
                    show_grid_imgs(inputs_unnormalized, black_and_white=False)
                    for i, j in enumerate(idx):
                        print('({}) pred labels: '.format(i), end='')
                        show_pred_labels(output[j], False)
                        print('({}) true labels: '.format(i), end='')
                        show_pred_labels(targets[j], True)
                        print('------------------------------------')

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        loss /= t_size

        print('Validation: Hamming_Loss = {:.4f}; F1_Score = {:.2f}%'
              ''.format(loss.item(), mean_f1 * 100))

        return loss.item(), mean_f1


def run_the_model(train_loader, valid_loader, path_save: str, use_cuda: bool):

    baseline = BaselineNetwork(use_cuda, learning_rate=1e-4, weight_decay=0)

    path_results = create_path_that_doesnt_exist(path_save, "results", ".pkl")
    path_model = create_path_that_doesnt_exist(path_save, "model", ".pth")

    results = {'name': 'Baseline_on_1k', 'loss_train': [], 'loss_valid': [], 'f1_valid': []}

    best_val_valid = 100
    save_model = True

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(baseline.optimizer, step_size=30, gamma=0.1)

    for epoch in range(13):
        loss_train = baseline.train(train_loader, epoch)
        loss_valid, f1_valid = baseline.valid(valid_loader, epoch)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)

        with open(path_results, 'wb') as fout:
            pickle.dump(results, fout)

        if save_model and loss_valid < best_val_valid:
            best_val_valid = loss_valid
            print('--- Saving model at Hamming_Loss = {:.5f} ---'.format(loss_valid))
            torch.save(baseline.model.state_dict(), path_model)

    plot_results(results, iou=False)


def strip_classifier(model):
    tmp = list(model.classifier)
    tmp = tmp[:-1]
    model.classifier = nn.Sequential(*tmp)


def save_features(train_loader, valid_loader, dir_path: str):
    """Save features of the Unary Model code mainly taken from
    https://github.com/cgraber/NLStruct/blob/master/experiments/train_tagging_unary.py"""

    def save(model, dataloader, device, features_path):
        results = []
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            results.append(model(inputs).data.cpu())
        results = torch.cat(results)
        torch.save(results, features_path)

    only_features = True
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    conv = ConvNet(label_dim=24).to(device)
    conv.load_state_dict(torch.load(dir_path + '/unary_1k_2_73.pth'))

    # Remove last layer of the model to store only the features
    if only_features:
        strip_classifier(conv.unary_model)

    conv.eval()

    if only_features:
        print('Saving Features for training data')
        save(conv, train_loader, device, dir_path + '/preprocessed/train_1k_features_20_epochs.pt')
        # print('Saving Features for validation data')
        # save_features(conv, valid_loader, device, dir_path + '/preprocessed/val_features_20_epochs.pt')
    else:
        print('Saving Unary for training data')
        save(conv, train_loader, device, dir_path + '/preprocessed/train_unary_20_epochs.pt')
        print('Saving Unary for validation data')
        save(conv, valid_loader, device, dir_path + '/preprocessed/val_unary_20_epochs.pt')


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    type_dataset = 'full'
    img_dir = 'mirflickr/'
    label_dir = dir_path + '/annotations/'
    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    train_label_file = os.path.join(PATH_PREPROCESSED, 'train_labels_1k.pt')
    val_label_file = os.path.join(PATH_PREPROCESSED, 'val_labels.pt')
    train_save_img_file = os.path.join(PATH_PREPROCESSED, 'train_imgs_1k.pt')
    val_save_img_file = os.path.join(PATH_PREPROCESSED, 'val_imgs.pt')
    load = False  # True if train_save_img_file is not None else False
    print('Loading training set....')
    train_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=train_save_img_file,
                                     annotations_folder=label_dir, save_label_file=train_label_file,
                                     mode='train', load=load)
    print('Loading validation set....')
    valid_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=val_save_img_file,
                                     annotations_folder=label_dir, save_label_file=val_label_file,
                                     mode='valid', load=load)
    print('Using a {} train {} validation split'.format(len(train_set), len(valid_set)))
    batch_size = 16
    batch_size_eval = 32

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=use_cuda
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_eval,
        pin_memory=use_cuda
    )

    run_the_model(train_loader, valid_loader, dir_path)

    # save_features(train_loader, valid_loader)


if __name__ == "__main__":
    main()
