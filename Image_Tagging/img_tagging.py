## Flickr Dataset can be downloaded at
#

import os
import threading
from queue import Queue, Empty
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import pickle
import pdb
import torch.optim as optim
from auxiliary_functions import *
import random
from PIL import Image


NUM_TRAIN = 20
NUM_TEST = 24970
NUM_VAL = 10

order_full = {'structures.txt': 0,
              'animals.txt': 1,
              'transport.txt': 2,
              'food.txt': 3,
              'portrait.txt': 4,
              'sky.txt': 5,
              'female.txt': 6,
              'male.txt': 7,
              'flower.txt': 8,
              'people.txt': 9,
              'river.txt': 10,
              'sunset.txt': 11,
              'baby.txt': 12,
              'plant_life.txt': 13,
              'indoor.txt': 14,
              'car.txt': 15,
              'bird.txt': 16,
              'dog.txt': 17,
              'tree.txt': 18,
              'sea.txt': 19,
              'night.txt': 20,
              'lake.txt': 21,
              'water.txt': 22,
              'clouds.txt': 23}

class_names = ['structures', 'animals', 'transport',
               'food', 'portrait', 'sky', 'female', 'male',
               'flower', 'people', 'river', 'sunset', 'baby',
               'plant_life', 'indoor', 'car', 'bird', 'dog',
               'tree', 'sea', 'night', 'lake', 'water', 'clouds']

order_r1 = {'baby_r1.txt': 0,
            'bird_r1.txt': 1,
            'car_r1.txt': 2,
            'clouds_r1.txt': 3,
            'dog_r1.txt': 4,
            'female_r1.txt': 5,
            'flower_r1.txt': 6,
            'male_r1.txt': 7,
            'night_r1.txt': 8,
            'people_r1.txt': 9,
            'portrait_r1.txt': 10,
            'river_r1.txt': 11,
            'sea_r1.txt': 12,
            'tree_r1.txt': 13,
            }


def show_pred_labels(label, is_true_label):
    for i, l in enumerate(label):
        if is_true_label:
            if l > 0.99:
               print('{}, '.format(class_names[i]), end='')
        else:
            if l > 0.0001:
                print('{} = {:.0f}%, '.format(class_names[i], l * 100), end='')
                if i + 1 % 8 == 0 and i > 0:
                   print('')
    print('')


class FlickrTaggingDataset(Dataset):
    """ Dataset can be downloaded at
    http://press.liacs.nl/mirflickr/mirdownload.html
    """
    def __init__(self, type_dataset, images_folder, save_img_file, annotations_folder,
                 save_label_file, mode, load=False):
        if type_dataset == 'full':
            order = order_full
        elif type_dataset == 'r1':
            order = order_r1
        else:
            raise Exception('DATASET MUST BE EITHER FULL OR R1 (input = {})'.format(type_dataset))
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        if load:
            print("LOADING PRECOMPUTED IMAGES")
            self.images = torch.load(save_img_file)
            self.labels = torch.load(save_label_file)
        else:
            print("LOADING IMAGES")
            if type_dataset == 'full':
                self.annotations = [None] * 24
            else:
                self.annotations = [None] * 14
            for annotation_file in os.listdir(annotations_folder):
                if type_dataset == 'full' and '_r1' in annotation_file:
                    continue
                elif type_dataset == 'r1' and '_r1' not in annotation_file:
                    continue
                elif 'README' in annotation_file:
                    continue
                vals = set()
                fin = open(os.path.join(annotations_folder, annotation_file), 'r')
                for line in fin:
                    vals.add(int(line.strip()) - 1)
                self.annotations[order[annotation_file]] = vals
            self.img_folder = images_folder
            self.img_files = [img_file for img_file in os.listdir(images_folder) if
                              os.path.isfile(os.path.join(images_folder, img_file)) and 'jpg' in img_file]
            print("NUM IMAGES: ", len(self.img_files))
            self.img_files.sort(key=lambda name: int(name[2:name.find('.jpg')]))

            if mode == 'train':
                self.img_files = self.img_files[:NUM_TRAIN]
            elif mode == 'test':
                self.img_files = self.img_files[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
            else:
                self.img_files = self.img_files[NUM_TRAIN + NUM_TEST:]
            self.images = [None] * len(self.img_files)
            self.labels = []
            for img_file in self.img_files:
                path = os.path.join(self.img_folder, img_file)
                with open(path, 'rb') as f:
                    with Image.open(f) as raw_img:
                        img = self.transform(raw_img.convert('RGB'))
                img_no = int(img_file[2:img_file.find('.jpg')]) - 1
                if mode == 'train':
                    img_ind = img_no
                elif mode == 'test':
                    img_ind = img_no - NUM_TRAIN
                else:
                    img_ind = img_no - NUM_TRAIN - NUM_TEST
                label = [0] * len(self.annotations)
                for i, annotation in enumerate(self.annotations):
                    if img_no in annotation:
                        label[i] = 1
                self.images[img_ind] = img
                self.labels.append(label)
            if save_img_file is not None:
                torch.save(self.images, save_img_file)
            if save_label_file is not None:
                torch.save(self.labels, save_label_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.Tensor(self.labels[idx])


class FlickrTaggingDatasetFeatures(Dataset):
    """ Dataset can be downloaded at
        http://press.liacs.nl/mirflickr/mirdownload.html
    """
    def __init__(self, type_dataset, feature_file, annotations_folder, save_label_file, mode,
                 images_folder=None, load=False):
        if type_dataset == 'full':
            order = order_full
        elif type_dataset == 'r1':
            order = order_r1
        else:
            raise Exception('DATASET MUST BE EITHER FULL OR R1 (input = {})'.format(type_dataset))
        self.features = torch.load(feature_file)
        if load:
            print("LOADING PRECOMPUTED LABELS")
            self.labels = torch.load(save_label_file)
            print("DONE")
        else:
            print("LOADING LABELS")
            if type_dataset == 'full':
                self.annotations = [None] * 24
            else:
                self.annotations = [None] * 14
            for annotation_file in os.listdir(annotations_folder):
                if type_dataset == 'full' and '_r1' in annotation_file:
                    continue
                elif type_dataset == 'r1' and '_r1' not in annotation_file:
                    continue
                elif 'README' in annotation_file:
                    continue
                vals = set()
                fin = open(os.path.join(annotations_folder, annotation_file), 'r')
                for line in fin:
                    vals.add(int(line.strip()) - 1)
                self.annotations[order[annotation_file]] = vals
            self.img_folder = images_folder
            self.img_files = [img_file for img_file in os.listdir(images_folder) if
                              os.path.isfile(os.path.join(images_folder, img_file)) and 'jpg' in img_file]
            print("NUM IMG FILES: ", len(self.img_files))
            self.img_files.sort(key=lambda name: int(name[2:name.find('.jpg')]))

            if mode == 'train':
                self.img_files = self.img_files[:NUM_TRAIN]
            elif mode == 'test':
                self.img_files = self.img_files[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
            else:
                self.img_files = self.img_files[NUM_TRAIN + NUM_TEST:]
            self.images = [None] * len(self.img_files)
            self.labels = []
            for img_file in self.img_files:
                path = os.path.join(self.img_folder, img_file)
                img_no = int(img_file[2:img_file.find('.jpg')]) - 1
                if mode == 'train':
                    img_ind = img_no
                elif mode == 'test':
                    img_ind = img_no - NUM_TRAIN
                else:
                    img_ind = img_no - NUM_TRAIN - NUM_TEST
                label = [0] * len(self.annotations)
                for i, annotation in enumerate(self.annotations):
                    if img_no in annotation:
                        label[i] = 1
                self.labels.append(label)
            print("DONE")
            if save_label_file is not None:
                torch.save(self.labels, save_label_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.labels[idx], self.features[idx, :]


def preprocess(dir_path):
    # path of images
    # !unzip  'drive/My Drive/projet_asp/img_tagging/mirflickr25k'
    img_dir = 'mirflickr/'
    label_dir = dir_path + '/annotations/'
    type_dataset = 'full'

    train_label_file = dir_path + '/preprocessed/train_labels.pt'
    val_label_file = dir_path + '/preprocessed/val_labels.pt'
    train_save_img_file = dir_path + '/preprocessed/train_imgs.pt'
    val_save_img_file = dir_path + '/preprocessed/val_imgs.pt'

    print('Loading training set....')
    train_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=train_save_img_file,
                                     annotations_folder=label_dir, save_label_file=train_label_file,
                                     mode='train', load=False)

    print('Loading validation set....')
    # I know its confusing that I'm calling the validation dataset test - sorry
    valid_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=val_save_img_file,
                                     annotations_folder=label_dir, save_label_file=val_label_file,
                                     mode='val', load=False)


class ConvNet(nn.Module):
    def __init__(self, label_dim, num_hidden, num_pairwise, add_second_layer, non_linearity=nn.Softplus()):
        super().__init__()

        self.non_linearity = non_linearity
        # Load pretrained AlexNet on ImageNet
        self.unary_model = torchvision.models.alexnet(pretrained=True)

        # Replace the last FC layer
        tmp = list(self.unary_model.classifier)

        if add_second_layer:
            tmp[-1] = nn.Linear(4096, num_hidden)

            self.B = torch.nn.Parameter(torch.empty(num_hidden, label_dim))
            # using same initialization as DVN paper
            torch.nn.init.normal_(self.B, mean=0, std=np.sqrt(2.0 / num_hidden))
        else:
            tmp[-1] = nn.Linear(4096, label_dim)
            self.B = None

        self.unary_model.classifier = nn.Sequential(*tmp)

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

    def forward(self, x, y):

        # First, send image through AlexNet
        x = self.unary_model(x)

        # Local energy
        if self.B is not None:
            e_local = self.non_linearity(x)
            e_local = torch.mm(e_local, self.B)
        else:
            e_local = x
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        e_local = e_local.view(e_local.size()[0], 1)

        # Label energy
        e_label = self.non_linearity(torch.mm(y, self.C1))
        e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label, e_local)
        return e_global


class DeepValueNetwork:

    def __init__(self, use_cuda, mode_sampling=Sampling.GT,
                 add_second_layer=False, learning_rate=0.01, weight_decay=1e-4, shuffle_n_size=False,
                 inf_lr=0.50, num_hidden=200, num_pairwise=32, label_dim=24, n_steps_inf=30, n_steps_adv=1):
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
        mode_sampling: int
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
        self.new_ep = True
        ################################

        # Deep Value Network is just a ConvNet
        self.model = ConvNet(label_dim, num_hidden, num_pairwise, add_second_layer).to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Paper use SGD for convnet with learning rate = 0.01
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.shuffle_n_size = shuffle_n_size

        # for inference, make sure gradients of convnet don't get accumulated
        self.training = False

    def get_oracle_value(self, pred_labels, gt_labels):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the F1 Score when training.
        and the discrete F1 when validating/testing
        """
        if pred_labels.shape != gt_labels.shape:
            raise ValueError('Invalid labels shape: gt = ', gt_labels.shape, 'pred = ', pred_labels.shape)

        if not self.training:
            # No relaxation, 0-1 only
            pred_labels = torch.where(pred_labels >= 0.5,
                                      torch.ones(1).to(self.device),
                                      torch.zeros(1).to(self.device))
            pred_labels = pred_labels.float()

        intersect = torch.sum(torch.min(pred_labels, gt_labels), dim=1)
        union = torch.sum(torch.max(pred_labels, gt_labels), dim=1)

        # for numerical stability
        epsilon = torch.full(union.size(), 10 ** -8).to(self.device)

        f1 = 2 * intersect / (intersect + torch.max(epsilon, union))
        # we want a (Batch_size x 1) tensor
        f1 = f1.view(-1, 1)
        return f1

    def get_ini_labels(self, x, gt_labels=None):
        """
        Get the tensor of predicted labels
        that we will do inference on
        """
        y = torch.zeros(x.size()[0], self.label_dim, dtype=torch.float32, device=self.device)

        if gt_labels is not None:
            # 50% of initialization labels: Start from GT; rest: start from zeros
            gt_indices = torch.rand(gt_labels.shape[0]).float().to(self.device) > 0.5
            y[gt_indices] = gt_labels[gt_indices]

        # Set requires_grad=True after in_place operation (changing the indices)
        y.requires_grad = True
        return y

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

        using_inference = False
        if self.mode_sampling == Sampling.ADV and self.training and np.random.rand() >= 0.5:
            # In training: Generate adversarial examples 50% of the time
            init_labels = self.get_ini_labels(x, gt_labels=gt_labels)
            # n_steps = random.randint(1, self.n_steps_adv)
            pred_labels = self.inference(x, init_labels, self.n_steps_adv, gt_labels=gt_labels, ep=ep)
        elif self.mode_sampling == Sampling.GT and self.training and np.random.rand() >= 0.5:
            # In training: If add_ground_truth=True, add ground truth outputs
            # to provide some positive examples to the network
            pred_labels = gt_labels
        else:
            using_inference = True
            init_labels = self.get_ini_labels(x)
            if self.training and self.shuffle_n_size:
                n_steps = random.randint(1, self.n_steps_inf)
            else:
                n_steps = self.n_steps_inf

            pred_labels = self.inference(x, init_labels, n_steps, ep=ep)

        return pred_labels.detach().clone(), using_inference

    def inference(self, x, y, num_iterations, gt_labels=None, ep=0):

        if self.training:
            self.model.eval()

        optim_inf = SGD(y, lr=self.inf_lr)

        with torch.enable_grad():
            for i in range(num_iterations):

                if gt_labels is not None:  # Adversarial
                    output = self.model(x, y)
                    oracle = self.get_oracle_value(y, gt_labels)
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

                # if (ep == 2 or ep == 5 or ep == 10) and self.new_ep and i % 5 == 0:
                #    pdb.set_trace()

        if self.training:
            self.model.train()

        return y

    def train(self, loader, ep):

        self.model.train()
        self.training = True
        self.new_ep = True
        n_train = len(loader.dataset)
        time_start = time.time()
        t_loss, hamming_loss, t_size, inf_size = 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            t_size += len(inputs)

            self.model.zero_grad()

            pred_labels, using_inference = self.generate_output(inputs, targets, ep)
            output = self.model(inputs, pred_labels)
            oracle = self.get_oracle_value(pred_labels, targets)
            loss = self.loss_fn(output, oracle)

            if using_inference:
                inf_size += len(inputs)
                hamming_loss += calculate_hamming_loss(pred_labels, targets)
                t_loss += loss.item()

            if torch.isnan(loss):
                print('Loss has NaN! Loss={:.5f}'.format(loss.item()))
                print('optim.params =', self.optimizer.params)
                raise ValueError('Loss has Nan')

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0 and inf_size > 0:
                print('\rTraining Epoch {} [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; '
                      'Avg_Loss = {:.5f}; Hamming_Loss = {:.4f}; Pred_F1 = {:.2f}%; Real_F1 = {:.2f}%'
                      ''.format(ep, t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / inf_size,
                                hamming_loss / inf_size, 100 * torch.sigmoid(output).mean(), 100 * oracle.mean()),
                      end='')

            self.new_ep = False

        hamming_loss /= inf_size
        t_loss /= inf_size
        self.training = False
        print('')
        return t_loss, hamming_loss

    def valid(self, loader, ep):

        self.model.eval()
        self.training = False
        self.new_ep = True
        loss, hamming_loss, t_size = 0, 0, 0
        mean_f1 = []
        int_show = random.randint(0, 20)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                t_size += len(inputs)

                pred_labels, _ = self.generate_output(inputs, gt_labels=None, ep=ep)
                output = self.model(inputs, pred_labels)

                oracle = self.get_oracle_value(pred_labels, targets)
                hamming_loss += calculate_hamming_loss(pred_labels, targets)

                loss += self.loss_fn(output, oracle)
                for o in oracle:
                    mean_f1.append(o)
                self.new_ep = False

                if batch_idx == int_show:
                    idx = np.random.choice(np.arange(len(inputs)), 3, replace=False)
                    inputs_unnormalized = inputs[idx].cpu()
                    inputs_unnormalized = [inv_normalize(i) for i in inputs_unnormalized]
                    show_grid_imgs(inputs_unnormalized, black_and_white=False)
                    for i, j in enumerate(idx):
                        print('({}) pred labels: '.format(i), end='')
                        show_pred_labels(pred_labels[j], False)
                        print('({}) true labels: '.format(i), end='')
                        show_pred_labels(targets[j], True)
                        print('------------------------------------')

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        loss /= t_size
        hamming_loss /= t_size

        print('Validation: Loss = {:.5f}; Hamming_Loss = {:.4f}; Pred_F1 = {:.2f}%, Real_F1 = {:.2f}%'
              ''.format(loss.item(), hamming_loss, 100 * torch.sigmoid(output).mean(), 100 * mean_f1))

        return loss.item(), hamming_loss, mean_f1


def start():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    type_dataset = 'full'
    label_dir = dir_path + '/annotations/'
    img_dir = dir_path + '/mirflickr25k/mirflickr/'

    # Use GPU if it is available
    use_cuda = torch.cuda.is_available()

    train_label_file = None  # dir_path + '/preprocessed/train_labels.pt'
    val_label_file = None  # dir_path + '/preprocessed/val_labels.pt'
    train_save_img_file = None  # dir_path + '/preprocessed/train_imgs.pt'
    val_save_img_file = None  # dir_path + '/preprocessed/val_imgs.pt'

    print('Loading training set....')
    train_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=train_save_img_file,
                                     annotations_folder=label_dir, save_label_file=train_label_file,
                                     mode='train', load=False)

    print('Loading validation set....')
    valid_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=val_save_img_file,
                                     annotations_folder=label_dir, save_label_file=val_label_file,
                                     mode='val', load=False)

    print('Using a {} train {} validation split'.format(len(train_set), len(valid_set)))

    batch_size = 2
    batch_size_eval = 2

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

    mode_sampling = Sampling.ADV
    str_res = 'Ground_Truth' if mode_sampling == Sampling.GT else 'Adversarial'
    print('Using {} Sampling'.format(str_res))

    DVN = DeepValueNetwork(use_cuda, mode_sampling, add_second_layer=False, shuffle_n_size=False,
                           learning_rate=1e-4, weight_decay=1e-4, inf_lr=0.5, num_hidden=200, num_pairwise=32,
                           n_steps_inf=30, n_steps_adv=1)

    results_path = dir_path + '/results/'
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Increment a counter so that previous results with the same args will not
    # be overwritten. Comment out the next four lines if you only want to keep
    # the most recent results.
    i = 0
    while os.path.exists(results_path + str(i) + '.pkl'):
        i += 1
    results_path = results_path + str(i)

    results = {'name': 'DVN_Inference_and_' + str_res, 'loss_train': [],
               'hamming_loss_train': [], 'hamming_loss_valid': [],
               'loss_valid': [], 'f1_valid': []}

    best_val_valid = 0
    save_model = False

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(DVN.optimizer, step_size=100, gamma=0.1)

    for epoch in range(100):
        loss_train, h_loss_train = DVN.train(train_loader, epoch)
        loss_valid, h_loss_valid, f1_valid = DVN.valid(valid_loader, epoch)
        scheduler.step()
        results['loss_train'].append(loss_train)
        results['loss_valid'].append(loss_valid)
        results['f1_valid'].append(f1_valid)
        results['hamming_loss_train'].append(h_loss_train)
        results['hamming_loss_valid'].append(h_loss_valid)

        with open(results_path + '.pkl', 'wb') as fout:
            pickle.dump(results, fout)

        if epoch > 10 and save_model and h_loss_valid > best_val_valid:
            best_val_valid = h_loss_valid
            print('--- Saving model at Hamming = {:.4f} ---'.format(h_loss_valid))
            torch.save(DVN.model.state_dict(), results_path + '.pth')

    plot_results(results, iou=False)
    plot_hamming_loss(results)


if __name__ == '__main__':

    start()


