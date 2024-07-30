from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        list_data = os.listdir(self.data_dir)

        list_label = [f for f in list_data if 'label' in f]
        list_input = [f for f in list_data if 'input' in f]

        list_label.sort()
        list_input.sort()

        self.list_label = list_label
        self.list_input = list_input

    def __len__(self):
        return len(self.list_label)

    def __getitem__(self, index):
        label_ = np.load(os.path.join(self.data_dir, self.list_label[index]))
        input_ = np.load(os.path.join(self.data_dir, self.list_input[index]))

        # 정규화
        label = label_/255.0
        input_ = input_/255.0

        if label_.ndim == 2:
            label_ = label_[:, :, np.newaxis]
        if input_.ndim == 2:
            input_ = input_[:, :, np.newaxis]

        # transform이 정의되어 있다면
        data = {'input': input_, 'label': label_}
        if self.transform:
            data = self.transform(data)

        return data


class ToTensor(object):
    def __call__(self, data):
        label_, input_ = data['label'], data['input']

        label_ = label_.transpose((2, 0, 1)).astype(np.float32)
        input_ = input_.transpose((2, 0, 1)).astype(np.float32)

        return {'label': torch.from_numpy(label_), 'input': torch.from_numpy(input_)}


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label_, input_ = data['label'], data['input']

        input_ = (input_ - self.mean) / self.std

        return {'label': label_, 'input': input_}


class RandomFilp(object):
    def __call__(self, data):
        label_, input_ = data['label'], data['input']
        if np.random.rand() > 0.5:
            label_ = np.fliplr(label_)
            input_ = np.fliplr(input_)

            return {'label': label_, 'input': input_}
        else:
            return {'label': label_, 'input': input_}


if __name__ == "__main__":
    data_dir = './datasets/train'
    transform = transforms.Compose(
        [Normalization(), RandomFilp(), ToTensor()])
    dataset = Dataset(data_dir, transform=transform)
    data = dataset[0]
    input_ = data['input']
    label_ = data['label']

    plt.subplot(122)
    plt.hist(label_.flatten(), bins=20)
    plt.title('label')

    plt.subplot(121)
    plt.hist(input_.flatten(), bins=20)
    plt.title('input')

    plt.show()
