import time
import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import random
import pickle

from src.utils import Sampling, SGD, create_path_that_doesnt_exist
from src.visualization_utils import (
    plot_results, show_grid_imgs, plot_gradients, save_grid_imgs
)
from src.image_segmentation.utils import average_over_crops, PATH_DATA_WEIZMANN
from .weizmann_horse_dataset import WeizmannHorseDataset


__author__ = "HSU CHIH-CHAO and Philippe Beardsell. University of Montreal"


class ConvNet(nn.Module):

    def __init__(self, non_linearity='relu'):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(4, 64, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)

        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(128 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 1)

        # apply dropout on the first FC layer as paper mentioned
        self.dropout = nn.Dropout(p=0.25)

        non_linearity = non_linearity.lower()
        if non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        else:
            raise ValueError('Unknown activation Convnet:', non_linearity)

    def forward(self, x, y):
        # We first concatenate the img and the mask
        z = torch.cat((x, y), 1)

        z = self.non_linearity(self.conv1(z))
        z = self.non_linearity(self.conv2(z))
        z = self.non_linearity(self.conv3(z))

        # flatten before FC layers
        z = z.view(-1, 128 * 6 * 6)
        z = self.non_linearity(self.fc1(z))
        z = self.dropout(z)
        z = self.non_linearity(self.fc2(z))
        z = self.fc3(z)
        return z


class DeepValueNetwork:

    def __init__(self, dir_path, use_cuda, mode_sampling=Sampling.GT, learning_rate=0.01,
                 weight_decay=1e-3, shuffle_n_size=False, inf_lr=50, momentum_inf=0,
                 label_dim=(24, 24), n_steps_inf=30, n_steps_adv=1):
        """
        Parameters
        ----------
        use_cuda: boolean
            true if we are using gpu, false if using cpu
        learning_rate : float
            learning rate for updating the value network parameters
            default: 0.01 in DVN paper
        inf_lr : float
            learning rate for the inference procedure
        mode_sampling: str
            Sampling.ADV:
                Generate adversarial tuples while training.
                (Usually outperforms stratified sampling and adding ground truth)
            Sampling.STRAT: Not yet implemented)
                Sample y proportional to its exponential oracle value.
                Sample from the exponentiated value distribution using stratified sampling.
            Sampling.GT:
                Simply add the ground truth outputs y* with some probably p while training.
        """

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.mode_sampling = mode_sampling
        if self.mode_sampling == Sampling.STRAT:
            raise ValueError('Stratified sampling is not yet implemented!')

        self.label_dim = label_dim

        # Inference hyperparameters
        self.n_steps_adv = n_steps_adv
        self.n_steps_inf = n_steps_inf
        self.inf_lr = inf_lr
        self.momentum_inf = momentum_inf
        self.new_ep = True
        ################################

        # for visualization purpose in "inference" function
        self.filename_train_other, self.filename_train_inf, self.filename_valid = 0, 0, 0
        # if directory doesn't exist, create it
        dir_predict_imgs = dir_path + '/pred/'
        self.dir_train_inf_img = dir_predict_imgs + '/train_inference/'
        self.dir_train_adversarial_img = dir_predict_imgs + '/train_adversarial/'
        self.dir_valid_img = dir_predict_imgs + '/valid/'
        for directory in [self.dir_train_inf_img, self.dir_train_adversarial_img, self.dir_valid_img]:
            if not os.path.isdir(directory):
                os.makedirs(directory)

        # Deep Value Network is just a ConvNet
        self.model = ConvNet().to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Paper use SGD for convnet with learning rate = 0.01
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.shuffle_n_size = shuffle_n_size

        # monitor norm gradients
        self.norm_gradient_inf = [[] for _ in range(n_steps_inf)]
        self.norm_gradient_adversarial = [[] for _ in range(n_steps_inf)]

        # for inference, make sure gradients of convnet don't get accumulated
        self.training = False

    def generate_output(self, x, gt_labels=None, ep=0):
        """
        Generate an output y to compute
        the loss v(y, y*) --> we can use different
        techniques to generate the output
        1) Gradient based inference
        2) Simply add the ground truth outputs
        2) Generating adversarial tuples
        3) TODO: Stratified Sampling: Random samples from Y, biased towards y*
        """

        if self.mode_sampling == Sampling.ADV and self.training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            # n_steps = random.randint(1, self.n_steps_adv)
            pred_labels = self.inference(x, init_labels, self.n_steps_adv,
                                         gt_labels=gt_labels, ep=ep)
        elif self.mode_sampling == Sampling.GT and self.training and np.random.rand() >= 0.5:
            # In training: If add_ground_truth=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            init_labels = self.get_ini_labels(x)
            if self.training and self.shuffle_n_size:
                n_steps = random.randint(1, self.n_steps_inf)
            else:
                n_steps = self.n_steps_inf

            pred_labels = self.inference(x, init_labels, n_steps, ep=ep)

        return pred_labels.detach().clone()

    def inference(self, x, y, num_iterations, gt_labels=None, ep=0):

        if self.training:
            self.model.eval()

        optim_inf = SGD(y, lr=self.inf_lr, momentum=self.momentum_inf)

        # print condition to monitor progress
        print_cdn = False  # ep % 10 == 0 and self.new_ep

        with torch.enable_grad():
            for i in range(num_iterations):

                if gt_labels is not None:  # Adversarial
                    output = self.model(x, y)
                    oracle = self.get_oracle_value(y, gt_labels)
                    # this is the BCE loss with logits
                    value = self.loss_fn(output, oracle)

                    if print_cdn and i == num_iterations - 1:
                        print('\n value = ', value, '(output, oracle) =', torch.cat((torch.sigmoid(output), oracle), 1))
                        img = x.detach().cpu()
                        show_grid_imgs(img)

                        print('pred_mask (left) real (right) =')
                        mask = y.detach().cpu()
                        real_mask = gt_labels.detach().cpu()
                        show_grid_imgs(torch.cat((mask, real_mask)))
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

                if print_cdn:
                    if i == 4 or i == 14 or i == num_iterations - 1:
                        print('-----------INFERENCE(AFTER', i + 1, 'steps)---------------')
                        img = x.detach().cpu()
                        show_grid_imgs(img)

                        print('pred_mask  =')
                        mask = y.detach().cpu()
                        show_grid_imgs(mask)
                    if i == num_iterations - 1:
                        img = y.detach().cpu()
                        if self.training:
                            if gt_labels is not None:
                                directory = self.dir_train_adversarial_img + str(self.filename_train_other)
                                self.filename_train_other += 1
                            else:
                                directory = self.dir_train_inf_img + str(self.filename_train_inf)
                                self.filename_train_inf += 1
                        else:
                            directory = self.dir_valid_img + str(self.filename_valid)
                            self.filename_valid += 1
                        save_grid_imgs(img, directory)

        if self.training:
            self.model.train()

        return y

    def train(self, loader, ep):

        self.model.train()
        self.training = True

        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, t_size = 0, 0

        for batch_idx, (raw_inputs, inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels = self.generate_output(inputs, targets, ep)
            output = self.model(inputs, pred_labels)
            oracle = self.get_oracle_value(pred_labels, targets)

            loss = self.loss_fn(output, oracle)
            t_loss += loss.item()
            if torch.isnan(loss):
                print('Loss has NaN! Loss={:.5f}'.format(loss.item()))
                print('optim.params =', self.optimizer.params)
                raise ValueError('Loss has Nan')

            loss.backward()
            self.optimizer.step()
            self.new_ep = False

            if batch_idx % 3 == 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                      'Avg_Loss = {:.5f}; Pred_IOU = {:.2f}%; Real_IOU = {:.2f}%'
                      ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size,
                                100 * torch.sigmoid(output).mean(), 100 * oracle.mean()),
                      end='')

        t_loss /= t_size
        self.training = False
        print('')
        return t_loss

    def valid(self, loader, ep):

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

                pred_labels = self.generate_output(inputs, gt_labels=None, ep=ep)
                output = self.model(inputs, pred_labels)
                oracle = self.get_oracle_value(pred_labels, targets)

                loss += self.loss_fn(output, oracle)
                for o in oracle:
                    mean_iou.append(o)
                self.new_ep = False

        mean_iou = torch.stack(mean_iou)
        mean_iou = torch.mean(mean_iou)
        mean_iou = mean_iou.cpu().numpy()
        loss /= t_size

        print('Validation: Loss = {:.5f}; Pred_IOU = {:.2f}%, Real_IOU = {:.2f}%'
              ''.format(loss.item(), 100 * torch.sigmoid(output).mean(), 100 * mean_iou))

        return loss.item(), mean_iou

    def test(self, loader):
        """ At Test time, we are averaging our predictions
        over 36 crops of 24x24 mask to predict a 32x32 mask
        """

        self.model.eval()
        self.training = False
        self.new_ep = False
        mean_iou = []

        with torch.no_grad():
            for batch_idx, (raw_inputs, inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()

                # For test: inputs is a 5d tensor
                bs, ncrops, channels, h, w = inputs.size()

                pred_labels = self.generate_output(inputs.view(-1, channels, h, w), gt_labels=None)
                # fuse batch size and ncrops to know our estimated IOU
                output = self.model(inputs.view(-1, channels, h, w), pred_labels)
                # go back to normal shape and take the mean in the 1 dim
                output = output.view(bs, ncrops, 1).mean(1)

                pred_labels = pred_labels.view(bs, ncrops, h, w)

                final_pred = average_over_crops(pred_labels, self.device)
                oracle = self.get_oracle_value(final_pred, targets)
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


def run_the_model(train_loader, valid_loader, dir_path, use_cuda, save_model,
                  n_epochs, mode_sampling=Sampling.GT, shuffle_n_size=False, learning_rate=0.01,
                  weight_decay=1e-3, inf_lr=50., momentum_inf=0, n_steps_inf=30, n_steps_adv=1,
                  step_size_scheduler_main=300, gamma_scheduler_main=1.):

    dvn = DeepValueNetwork(dir_path, use_cuda, mode_sampling, shuffle_n_size=shuffle_n_size,
                           learning_rate=learning_rate, weight_decay=weight_decay, inf_lr=inf_lr,
                           momentum_inf=momentum_inf, n_steps_inf=n_steps_inf, n_steps_adv=n_steps_adv)

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(dvn.optimizer, step_size=step_size_scheduler_main,
                                                gamma=gamma_scheduler_main)

    results = {'name': 'DVN_Whorse', 'loss_train': [],
               'loss_valid': [], 'IOU_valid': [], 'batch_size': train_loader.batch_size,
               'shuffle_n_size': shuffle_n_size, 'learning_rate': learning_rate,
               'weight_decay': weight_decay, 'batch_size_eval': valid_loader.batch_size,
               'mode_sampling': mode_sampling, 'inf_lr': inf_lr, 'n_steps_inf': n_steps_inf,
               'momentum_inf': momentum_inf,
               'step_size_scheduler_main': step_size_scheduler_main,
               'gamma_scheduler_main': gamma_scheduler_main, 'n_steps_adv': n_steps_adv}

    str_model = '_GT_' if mode_sampling == Sampling.GT else '_Adv_'
    dir_save = os.path.join(dir_path, "results")
    path_results = create_path_that_doesnt_exist(dir_save, "results", ".pkl")
    path_model_save = create_path_that_doesnt_exist(dir_save, "model_" + str_model, ".pth")

    best_iou_valid = 0

    for epoch in range(n_epochs):
        loss_train = dvn.train(train_loader, epoch)
        loss_valid, iou_valid = dvn.valid(valid_loader, epoch)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['IOU_valid'].append(iou_valid)

        with open(path_results, 'wb') as fout:
            pickle.dump(results, fout)

        if epoch > 20 and save_model and iou_valid > best_iou_valid:
            best_iou_valid = iou_valid
            print('--- Saving model at IOU_{:.2f} ---'.format(100 * best_iou_valid))
            torch.save(dvn.model.state_dict(), path_model_save)

    plot_results(results, iou=True)
    plot_gradients(dvn.norm_gradient_inf, 'Inference')
    plot_gradients(dvn.norm_gradient_adversarial, 'Adversarial Examples')


def start():

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    image_dir = dir_path + '/images'
    mask_dir = dir_path + '/masks'

    # Use Dataset to resize and convert to Tensor
    train_set = WeizmannHorseDataset(image_dir, mask_dir, subset='train',
                                     random_mirroring=False, thirty_six_cropping=False)

    valid_set = WeizmannHorseDataset(image_dir, mask_dir, subset='valid',
                                     random_mirroring=False, thirty_six_cropping=False)

    batch_size = 1
    batch_size_eval = 20

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

    print('Using a {} train {} validation split'
          ''.format(len(train_loader.dataset), len(valid_loader.dataset)))

    mean_imgs, std_imgs, mean_mask = train_set.compute_mean_and_stddev()
    print('mean_imgs =', mean_imgs, 'std_dev_imgs =', std_imgs)

    # train_set.normalize = transforms.Normalize(mean_imgs, std_imgs)
    # valid_set.normalize = transforms.Normalize(mean_imgs, std_imgs)
    mode_sampling = Sampling.ADV

    # Run the model
    run_the_model(train_loader, valid_loader, dir_path, use_cuda, save_model=True,
                  n_epochs=1000, mode_sampling=mode_sampling, shuffle_n_size=False, learning_rate=1e-4,
                  weight_decay=1e-5, inf_lr=5e3, momentum_inf=0, n_steps_inf=30,
                  n_steps_adv=3, step_size_scheduler_main=800, gamma_scheduler_main=0.10)

    # plot_results(results, iou=True)


def run_test_set():
    """ Compute IOU on test set using 36 crops averaging """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    image_dir = os.path.join(PATH_DATA_WEIZMANN, "images")
    mask_dir = os.path.join(PATH_DATA_WEIZMANN, "masks")

    dvn = DeepValueNetwork(dir_path, use_cuda, learning_rate=1e-4, weight_decay=1e-3,
                           inf_lr=5e2, n_steps_inf=30, n_steps_adv=3)

    dvn.model = ConvNet().to(device)
    dvn.model.load_state_dict(torch.load(dir_path + '/11_Adv_.pth'))
    dvn.model.eval()

    # Compute IOU single prediction on 24x24 crops and 36 crops averaging on 32x32
    for i in range(2):
        thirtysix_crops = False if i == 0 else True
        test_set = WeizmannHorseDataset(image_dir, mask_dir, subset='test',
                                        random_mirroring=False, thirty_six_cropping=thirtysix_crops)

        batch_size_eval = 8

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size_eval,
            pin_memory=use_cuda
        )

        print('-------------------------------------------')
        if i == 0:
            print('Single crop IOU prediction')
            dvn.valid(test_loader, ep=100)
        else:
            print('36 Crops IOU prediction')
            dvn.test(test_loader)


def main():
    start()
    # On test set:
    run_test_set()


if __name__ == "__main__":
    main()
