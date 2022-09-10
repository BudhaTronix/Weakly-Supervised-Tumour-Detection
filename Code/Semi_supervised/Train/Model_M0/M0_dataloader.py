import numpy as np
import pandas as pd
import torch
import torchio as tio
import torch.nn.functional as f
from os.path import exists
import _pickle as cPickle

torch.set_num_threads(1)
from torch.utils.data.dataset import Dataset


class TeacherCustomDataset(Dataset):
    def __init__(self, isChaos, dataset_path, csv_file, transform, transform_val):
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
        self.transform_val = transform_val
        # Read the csv file
        self.data_info = pd.read_csv(self.dataset_path + "/" + self.csv_file, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        # Check for MRI or CT
        self.isMRI = False

        self.temp_location = self.dataset_path + "temp/"

    def __getitem__(self, index):
        if self.isMRI:
            # Open image
            img = tio.ScalarImage(self.dataset_path + "mri/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
            # Normalize the data
            img = (img - img.min()) / (img.max() - img.min())

            # Transform image
            img_transformed = self.transform(img).squeeze(0)

            # Get label(class) of the image based on the cropped pandas column
            img_lbl = tio.ScalarImage(self.dataset_path + "mri_gt/" + self.label_arr[index])[tio.DATA].permute(0, 3, 1,
                                                                                                               2)
            if self.isChaos:
                np_frame = np.array(img_lbl)
                np_frame[(np_frame < 55) | (np_frame > 70)] = 0
                np_frame[(np_frame >= 55) & (np_frame <= 70)] = 1

                img_lbl = torch.Tensor(np_frame.astype(np.float))
            lbl_transformed = self.transform(img_lbl).squeeze(0)

            return img_transformed, lbl_transformed
        else:
            # Open image
            img = tio.ScalarImage(self.dataset_path + "ct_mri_reg/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
            # Normalize the data
            img = (img - img.min()) / (img.max() - img.min())

            # Transform image
            img_transformed = f.interpolate(img.unsqueeze(0), size=self.transform_val).squeeze()

            # Get label(class) of the image based on the cropped pandas column
            img_lbl = tio.ScalarImage(self.dataset_path + "ct_mri_reg_gt/" + self.label_arr[index])[tio.DATA].permute(0, 3, 1,
                                                                                                              2)
            img_lbl = (img_lbl - img_lbl.min()) / (img_lbl.max() - img_lbl.min())
            lbl_transformed = f.interpolate(img_lbl.unsqueeze(0), size=self.transform_val).squeeze()

            return img_transformed, lbl_transformed

    def __len__(self):
        return self.data_len
