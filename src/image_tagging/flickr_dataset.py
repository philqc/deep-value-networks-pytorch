import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


NUM_TRAIN = 10000
NUM_TEST = 10000
NUM_VAL = 5000

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

mean_img = [0.485, 0.456, 0.406]
std_img = [0.229, 0.224, 0.225]

inv_normalize = transforms.Normalize(
    mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
    std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]]
)


class FlickrTaggingDataset(Dataset):
    """
    Dataset can be downloaded at
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
            transforms.Normalize(mean=mean_img, std=std_img),
        ])
        self.mode = mode
        self.save_img_file = save_img_file

        self.img_folder = images_folder
        self.img_files = [img_file for img_file in os.listdir(images_folder) if
                          os.path.isfile(os.path.join(images_folder, img_file)) and 'jpg' in img_file]
        print("NUM IMAGES: ", len(self.img_files))
        self.img_files.sort(key=lambda name: int(name[2:name.find('.jpg')]))

        if load:
            print("LOADING PRECOMPUTED IMAGES")
            self.labels = torch.load(save_label_file)
            self.imgs = torch.load(save_img_file)
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

            if mode == 'train':
                self.img_files = self.img_files[:NUM_TRAIN]
            elif mode == 'test':
                self.img_files = self.img_files[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
            else:
                self.img_files = self.img_files[NUM_TRAIN + NUM_TEST:]
            self.labels = []
            self.imgs = [None] * len(self.img_files)
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
                self.imgs[img_ind] = img
                self.labels.append(label)

            if save_label_file is not None:
                torch.save(self.labels, save_label_file)
            if save_img_file is not None:
                torch.save(self.imgs, self.save_img_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], torch.Tensor(self.labels[idx])


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
            img_files = [img_file for img_file in os.listdir(images_folder) if
                         os.path.isfile(os.path.join(images_folder, img_file)) and 'jpg' in img_file]
            print("NUM IMG FILES: ", len(img_files))
            img_files.sort(key=lambda name: int(name[2:name.find('.jpg')]))

            if mode == 'train':
                img_files = img_files[:NUM_TRAIN]
            elif mode == 'test':
                img_files = img_files[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
            else:
                img_files = img_files[NUM_TRAIN + NUM_TEST:]
            self.labels = []
            for img_file in img_files:
                img_no = int(img_file[2:img_file.find('.jpg')]) - 1
                label = [0] * len(self.annotations)
                for i, annotation in enumerate(self.annotations):
                    if img_no in annotation:
                        label[i] = 1
                self.labels.append(label)
            print("DONE")
            if save_label_file is not None:
                torch.save(self.labels, save_label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx, :], torch.Tensor(self.labels[idx])