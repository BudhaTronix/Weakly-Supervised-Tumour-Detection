import numpy as np
import pandas as pd
import torch
import torchio as tio

torch.set_num_threads(1)
from torch.utils.data.dataset import Dataset


class TeacherCustomDataset(Dataset):
    def __init__(self, isChaos, dataset_path, csv_file, transform):
        """
        Args:
            csv_file (string): csv file name
            dataset_path (string): path to the folder where images are
            transform: pytorch(torchIO) transforms for transforms and tensor conversion
        """
        # Check if chaos
        self.isChaos = isChaos
        # Dataset Path
        self.dataset_path = dataset_path
        # CSV Path
        self.csv_file = csv_file
        # Transforms
        self.transform = transform
        # Read the csv file
        self.data_info = pd.read_csv(self.dataset_path + "/" + self.csv_file, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Open image
        img = tio.ScalarImage(self.dataset_path + "mri/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        # Normalize the data
        img = (img - img.min()) / (img.max() - img.min())

        # Transform image
        img_transformed = self.transform(img).squeeze(0)

        # Get label(class) of the image based on the cropped pandas column
        img_lbl = tio.ScalarImage(self.dataset_path + "mri_gt/" + self.label_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        if self.isChaos:
            np_frame = np.array(img_lbl)
            np_frame[(np_frame < 55) | (np_frame > 70)] = 0
            np_frame[(np_frame >= 55) & (np_frame <= 70)] = 1

            img_lbl = torch.Tensor(np_frame.astype(np.float))
        lbl_transformed = self.transform(img_lbl).squeeze(0)

        return img_transformed, lbl_transformed

    def __len__(self):
        return self.data_len
