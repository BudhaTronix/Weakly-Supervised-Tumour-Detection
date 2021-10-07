import os
import sys

import torch
import torch.optim as optim
import torchio as tio
from torchvision import transforms

from student_dataloader import StudentCustomDataset
from student_train import train

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'

try:
    from Code.Utils.CSVGenerator import checkCSV
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/LiverTumorSeg/Code/Utils/')
    from CSVGenerator import checkCSV


class StudentPipeline:
    def __init__(self):
        self.dataset_Path = ""
        self.batch_size = 32

    @staticmethod
    def defineModel():
        # Define Model
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=1, out_channels=1, init_features=32, pretrained=False)
        return model

    @staticmethod
    def defineOptimizer(model):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return optimizer

    @staticmethod
    def getWarp(img1, img2):
        return 0

    @staticmethod
    def getTransform(m, s):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ])

        return preprocess

    def trainModel(self):
        model = self.defineModel()
        optimizer = self.defineOptimizer(model)
        modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet.pth"
        modelPath_bestweight = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_bw.pth"
        csv_file = "dataset.csv"
        transform_val = (1, 256, 256)
        transform = tio.CropOrPad(transform_val)
        num_epochs = 1000
        dataset_path = "/project/cmandal/liver_seg/datasets/chaos/"
        checkCSV(dataset_Path=dataset_path, csv_FileName=csv_file, overwrite=True)
        dataset = StudentCustomDataset(dataset_path, csv_file, transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        dataloaders = [train_loader, validation_loader]
        train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, optimizer, log=True)


obj = StudentPipeline()
obj.trainModel()
