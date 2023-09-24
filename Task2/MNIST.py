import torch
from torch.utils.data import Dataset

import numpy as np

from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd


class MNIST(Dataset):
    """
    Dataset From Kaggle Digit Recognizer Competition.
    Loads Image and Label from a CSV File.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, data_csv_path, transforms=None) -> None:
        super().__init__()

        assert data_csv_path is not None
        data = pd.read_csv(data_csv_path)
        # print(data.head())
        self.labels = data['label']
        self.images = data.drop(['label'], axis=1)

        self.images = np.array(
            self.images, dtype=np.float32).reshape(-1, 28, 28)
        self.labels = np.array(self.labels, dtype=np.uint8)

        self.transforms = transforms

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms:
            img = self.transforms(img)
        label = int(self.labels[index])

        return img, label


if __name__ == "__main__":
    data_csv_path = os.path.join(str(Path(
        __file__).parent), "data/train.csv")

    dataset = MNIST(data_csv_path)

    fig, ax = plt.subplots(nrows=1, ncols=10)

    for i in range(10):
        img, label = dataset.__getitem__(np.random.randint(low=0, high=10000))
        ax[i].imshow(img)
        ax[i].set_title(label)

    plt.show()
