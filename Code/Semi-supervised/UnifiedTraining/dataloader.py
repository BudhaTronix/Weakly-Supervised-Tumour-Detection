import numpy as np
import pandas as pd
import torchio as tio
import torch
import torch.nn.functional as f

torch.set_num_threads(1)
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_path, csv_file, transform, t_ct):
        """
        Args:
            csv_file (string): csv file name
            dataset_path (string): path to the folder where images are
            transform: pytorch(torchIO) transforms for transforms and tensor conversion
        """
        # Dataset Path
        self.dataset_path = dataset_path
        # CSV Path
        self.csv_file = csv_file
        # Transforms
        self.transform = transform
        self.t_ct = t_ct
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
        img = tio.ScalarImage(self.dataset_path + "images/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        # Normalize the data
        img = (img - img.min()) / (img.max() - img.min())
        # Transform image
        mri_transformed = self.transform(img).squeeze(0)

        # Open image
        img = tio.ScalarImage(self.dataset_path + "ct/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        # Normalize the data
        img = (img - img.min()) / (img.max() - img.min())
        # Transform image
        ct_transformed = self.transform(img).squeeze(0)
        ct_actualSize = self.t_ct(img).squeeze(0)

        # Get label(class) of the image based on the cropped pandas column
        img_lbl = tio.ScalarImage(self.dataset_path + "gt/" + self.label_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        np_frame = np.array(img_lbl)
        np_frame[np_frame < 240] = 0
        np_frame[np_frame >= 240] = 1

        img_lbl = torch.Tensor(np_frame.astype(np.float))
        lbl_transformed = self.transform(img_lbl).squeeze(0)

        mri_transformed = f.interpolate(mri_transformed, scale_factor=.1)  # .1 means 90% reduction
        lbl_transformed = f.interpolate(lbl_transformed, scale_factor=.1)
        ct_actualSize = f.interpolate(ct_actualSize, scale_factor=.1)

        return mri_transformed, ct_transformed, lbl_transformed, ct_actualSize

    def __len__(self):
        return self.data_len