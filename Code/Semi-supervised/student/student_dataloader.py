import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class StudentCustomDataset(Dataset):
    def __init__(self, dataset_path, csv_file, transform):
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
        # Read the csv file
        self.data_info = pd.read_csv(self.dataset_path + "/" + self.csv_file, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        ds = pydicom.dcmread(self.dataset_path + "images/" + single_image_name).pixel_array
        img_transformed = torch.Tensor(ds.astype(np.float))
        img_transformed = img_transformed.unsqueeze(0).unsqueeze(0)
        # Transform image
        img_transformed = self.transform(img_transformed).squeeze(0)
        # Get label(class) of the image based on the cropped pandas column
        im_frame = Image.open(self.dataset_path + "gt/" + self.label_arr[index])
        np_frame = np.array(im_frame)
        np_frame[np_frame < 240] = 0
        np_frame[np_frame >= 240] = 1
        lbl_transformed = torch.Tensor(np_frame.astype(np.float))
        lbl_transformed = lbl_transformed.unsqueeze(0).unsqueeze(0)
        lbl_transformed = self.transform(lbl_transformed).squeeze(0)
        warp_factor = ""

        return img_transformed, lbl_transformed, warp_factor

    def __len__(self):
        return self.data_len
