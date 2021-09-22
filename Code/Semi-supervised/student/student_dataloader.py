from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import torchio as tio


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
        self.data_info = pd.read_csv(self.csv_file, header=None)
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
        img = tio.ScalarImage(self.dataset_path+single_image_name)[tio.DATA].permute(0, 3, 1, 2)
        # Transform image
        img_transformed = self.transform(img).squeeze()
        # Get label(class) of the image based on the cropped pandas column
        image_label = self.label_arr[index]

        return img_transformed, image_label

    def __len__(self):
        return self.data_len





