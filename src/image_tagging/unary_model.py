import random
import torch
from torch.utils.data import DataLoader
import time
import pickle

from src.image_tagging.utils import calculate_hamming_loss
from src.image_tagging.load_save_flickr import (
    visualize_predictions, load_train_dataset_flickr, PATH_FLICKR_DATA, PATH_SAVE_FLICKR
)
from src.image_tagging.model.conv_net import ConvNet
from src.model.base_model import BaseModel
from src.utils import create_path_that_doesnt_exist


class BaselineNetwork(BaseModel):

    def __init__(self, learning_rate=0.01, weight_decay=1e-4, label_dim=24):
        super().__init__(ConvNet(label_dim))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

            output = self.model(inputs)
            loss = calculate_hamming_loss(output, targets)
            t_loss += loss.detach().clone().item()

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTraining Epoch: [{} / {} ({:.0f}%)]: Time per epoch: {:.2f}s; Avg_Hamming_Loss = {:.5f};'
                      ''.format(t_size, n_train, 100 * t_size / n_train,
                                (n_train / t_size) * (time.time() - time_start), t_loss / t_size),
                      end='')

        t_loss /= n_train
        print('')
        return t_loss

    def valid(self, loader: DataLoader):

        self.model.eval()
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

                f1 = self._f1_score(output, targets, False)
                for f in f1:
                    mean_f1.append(f)

                if batch_idx == int_show:
                    visualize_predictions(inputs, output, targets, False, False)

        mean_f1 = torch.stack(mean_f1)
        mean_f1 = torch.mean(mean_f1)
        mean_f1 = mean_f1.cpu().numpy()
        loss /= t_size

        print('Validation: Hamming_Loss = {:.4f}; F1_Score = {:.2f}%'
              ''.format(loss.item(), mean_f1 * 100))

        return loss.item(), mean_f1

    def test(self, loader: DataLoader):
        return self.valid(loader)


def run_the_model(train_loader: DataLoader, valid_loader: DataLoader, path_save: str, save_model=True):

    baseline = BaselineNetwork(learning_rate=1e-4, weight_decay=0)

    path_results = create_path_that_doesnt_exist(path_save, "results_unary", ".pkl")
    path_model = create_path_that_doesnt_exist(path_save, "model_unary", ".pth")

    results = {'name': 'baseline_on_1k', 'loss_train': [], 'loss_valid': [], 'f1_valid': []}

    best_val_valid = 100

    # Decay the learning rate by a factor of gamma every step_size # of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(baseline.optimizer, step_size=30, gamma=0.1)

    for epoch in range(13):
        loss_train = baseline.train(train_loader)
        loss_valid, f1_valid = baseline.valid(valid_loader)
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


def main():
    use_cuda = torch.cuda.is_available()
    train_loader, valid_loader = load_train_dataset_flickr(PATH_FLICKR_DATA, False, False, use_cuda,
                                                           batch_size=32, batch_size_eval=32)
    run_the_model(train_loader, valid_loader, PATH_SAVE_FLICKR)

    # save_features(train_loader, valid_loader)


if __name__ == "__main__":
    main()
