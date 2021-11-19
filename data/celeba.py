import os

import pandas as pd
from torch.utils.data import Dataset
from skimage import io


class CelebA(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.eval_partition = os.path.join(root_dir, 'list_eval_partition.txt')

        if not os.path.isdir(self.img_dir):
            raise Exception(f"Place img_align_celeba folder in {self.img_dir}")
        if not os.path.isfile(self.eval_partition):
            raise Exception(f'Place list_eval_partition.txt file in {self.eval_partition}')
        self.frame = pd.read_csv(self.eval_partition,
                                 header=None,
                                 sep=' ',
                                 index_col=None,
                                 usecols=[0, 1])
        if train:
            self.frame = self.frame[self.frame[1] == 0]
        else:
            self.frame = self.frame[self.frame[1] != 0]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.frame.iloc[idx, 0])
        img = io.imread(img_name)

        if self.transform:
            img = self.transform(img)
        # Return dummy label to be consistent with MNIST dataset.
        return (img, -1)
