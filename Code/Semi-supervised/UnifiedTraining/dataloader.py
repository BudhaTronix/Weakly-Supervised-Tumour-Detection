import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import torchio as tio

torch.set_num_threads(1)
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_path, csv_file, transform_val):
        """
        Args:
            csv_file (string)    : csv file name
            dataset_path (string): path to the folder where images are
            transform            : pytorch(torchIO) transforms for transforms and tensor conversion
        """
        # Dataset Path
        self.dataset_path = dataset_path
        # CSV Path
        self.csv_file = csv_file
        # Transforms
        self.transform_val = transform_val
        self.transform = tio.CropOrPad(self.transform_val)
        # Read the csv file
        self.data_info = pd.read_csv(self.dataset_path + "/" + self.csv_file, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Open MRI image
        img = tio.ScalarImage(self.dataset_path + "images/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        mri_actualSize = self.normalize(img)

        # Open CT image
        img = tio.ScalarImage(self.dataset_path + "ct/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        ct_actualSize = self.normalize(img)

        # Open Labels image
        img_lbl = tio.ScalarImage(self.dataset_path + "gt/" + self.label_arr[index])[tio.DATA].permute(0, 3, 1, 2)
        np_frame = np.array(img_lbl)
        np_frame[np_frame < 240] = 0
        np_frame[np_frame >= 240] = 1
        img_lbl = torch.Tensor(np_frame.astype(np.float))

        # Transform CT with size mentioned
        ct_transformed = f.interpolate(ct_actualSize.unsqueeze(0), size=self.transform_val) #optional, try without - Soumick

        # Transform MRI with the size of CT
        mri_transformed = f.interpolate(mri_actualSize.unsqueeze(0), size=ct_transformed.shape[2:])

        # Transform MRI label with the size of CT
        lbl_transformed = f.interpolate(img_lbl.unsqueeze(0), size=ct_transformed.shape[2:])

        return mri_transformed.squeeze(0), lbl_transformed.squeeze(0), ct_transformed.squeeze(0)

    def __len__(self):
        return self.data_len

    @staticmethod
    def normalize(img):
        return (img - img.min()) / (img.max() - img.min())
