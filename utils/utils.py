# Plant Village DataLoader
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class MaizePlantVillageDataset(Dataset):
    """Maize Plant Village dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.maize_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.maize_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        class_dict = {'CR': 0, 'H': 1, 'NLB': 2, 'GLP': 3}
        img_name = os.path.join(
            self.maize_frame.loc[idx, 'preprocessing_location'])
        image = io.imread(img_name)
        target = class_dict[self.maize_frame.loc[idx, 'class']]
        sample = [image, target]
        if self.transform:
            sample[0] = self.transform(Image.fromarray(sample[0]))

        return sample
