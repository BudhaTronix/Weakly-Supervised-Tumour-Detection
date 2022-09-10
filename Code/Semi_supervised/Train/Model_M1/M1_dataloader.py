import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import torchio as tio
import _pickle as cPickle
from os.path import exists


torch.set_num_threads(1)
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_path, csv_file, transform_val, isChaos, level, window):
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
        # Third column is the X delta
        self.x_delta = np.asarray(self.data_info.iloc[:, 2])
        # Third column is the Y delta
        self.y_delta = np.asarray(self.data_info.iloc[:, 3])
        # Third column is the Z delta
        self.z_delta = np.asarray(self.data_info.iloc[:, 4])
        # Calculate len
        self.data_len = len(self.data_info.index)
        # Check if Chaos of Clinical
        self.chaos = isChaos
        # CT transformations
        self.level = level
        self.window = window

        self.temp_location = self.dataset_path + "temp/"

    def __getitem__(self, index):
        if exists(self.temp_location + "mri_transformed_{}.pickle".format(index)) \
                and exists(self.temp_location + "mri_gt_transformed_{}.pickle".format(index)) \
                and exists(self.temp_location + "ct_transformed_{}.pickle".format(index)) \
                and exists(self.temp_location + "ct_gt_transformed_{}.pickle".format(index)):
            with open(self.temp_location + "mri_transformed_{}.pickle".format(index), "rb") as input_file:
                mri = cPickle.load(input_file)

            with open(self.temp_location + "mri_gt_transformed_{}.pickle".format(index), "rb") as input_file:
                mri_gt = cPickle.load(input_file)

            with open(self.temp_location + "ct_transformed_{}.pickle".format(index), "rb") as input_file:
                ct = cPickle.load(input_file)

            with open(self.temp_location + "ct_gt_transformed_{}.pickle".format(index), "rb") as input_file:
                ct_gt = cPickle.load(input_file)
            return mri, mri_gt, ct, ct_gt

        else:
            # Open MRI image
            mri = tio.ScalarImage(self.dataset_path + "mri/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
            mri = self.normalize(mri)
            new_shape = (int(mri.shape[1] * self.z_delta[index]),
                         int(mri.shape[2] * self.x_delta[index]),
                         int(mri.shape[3] * self.y_delta[index]))
            mri = f.interpolate(mri.unsqueeze(0), size=new_shape)
            mri_actualSize = mri

            # Open CT image
            ct = tio.ScalarImage(self.dataset_path + "ct/" + self.image_arr[index])[tio.DATA].permute(0, 3, 1, 2)
            for i in range(ct.shape[0]):
                ct[i, :, :] = self.ct_slice_window(slice=ct[i, :, :])
            ct_actualSize = self.normalize(ct)

            # Transform CT with size mentioned
            ct_transformed = f.interpolate(ct_actualSize.unsqueeze(0),
                                           size=self.transform_val)  # optional, try without - Soumick

            # Transform MRI with the size of CT
            mri_transformed = f.interpolate(mri_actualSize, size=ct_transformed.shape[2:])

            # Open Labels image
            mri_gt = tio.ScalarImage(self.dataset_path + "mri_gt/" + self.label_arr[index])[tio.DATA].permute(0, 3, 1, 2)
            if self.chaos:
                # Using the liver section in the MRI label
                np_frame = np.array(mri_gt)
                np_frame[(np_frame < 55) | (np_frame > 70)] = 0
                np_frame[(np_frame >= 55) & (np_frame <= 70)] = 1
                mri_gt = torch.Tensor(np_frame.astype(np.float))
                mri_gt = f.interpolate(mri_gt.unsqueeze(0), size=new_shape)

                # Open CT Labels
                img_ct_lbl = tio.ScalarImage(self.dataset_path + "ct_gt/" + self.label_arr[index])[tio.DATA]
                gt_ct_actualSize = self.normalize(img_ct_lbl)

                # Transform MRI label with the size of CT
                mri_gt_transformed = f.interpolate(mri_gt, size=ct_transformed.shape[2:])

                # Transform CT label with size mentioned
                ct_gt_transformed = f.interpolate(gt_ct_actualSize.unsqueeze(0), size=self.transform_val)

                with open(self.temp_location + "mri_transformed_{}.pickle".format(index), "wb") as output_file:
                    cPickle.dump(mri_transformed.squeeze(0), output_file)

                with open(self.temp_location + "mri_gt_transformed_{}.pickle".format(index), "wb") as output_file:
                    cPickle.dump(mri_gt_transformed.squeeze(0), output_file)

                with open(self.temp_location + "ct_transformed_{}.pickle".format(index), "wb") as output_file:
                    cPickle.dump(ct_transformed.squeeze(0), output_file)

                with open(self.temp_location + "ct_gt_transformed_{}.pickle".format(index), "wb") as output_file:
                    cPickle.dump(ct_gt_transformed.squeeze(0), output_file)

                return mri_transformed.squeeze(0), mri_gt_transformed.squeeze(0), ct_transformed.squeeze(
                    0), ct_gt_transformed.squeeze(0)
            else:
                mri_gt = f.interpolate(mri_gt.unsqueeze(0), size=new_shape)
                # Transform MRI label with the size of CT
                lbl_transformed = f.interpolate(mri_gt.unsqueeze(0), size=ct_transformed.shape[2:])

                return mri_transformed.squeeze(0), lbl_transformed.squeeze(0), ct_transformed.squeeze(0)

    def __len__(self):
        return self.data_len

    @staticmethod
    def normalize(img):
        return (img - img.min()) / (img.max() - img.min())

    def ct_slice_window(self, slice):
        """
        Function to display an image slice
        Input is a numpy 2D array
        """
        max = self.level + self.window / 2
        min = self.level - self.window / 2
        slice = slice.clip(min, max)

        return slice
