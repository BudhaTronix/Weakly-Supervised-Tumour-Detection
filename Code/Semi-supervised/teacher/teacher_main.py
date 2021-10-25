import os
import sys

import torch

torch.set_num_threads(1)

import torch.optim as optim
import torchio as tio
from torchvision import transforms

from teacher_dataloader import TeacherCustomDataset
from teacher_train import train

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from Model.unet3d import U_Net
try:
    from Code.Utils.CSVGenerator import checkCSV
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/LiverTumorSeg/Code/Utils/')
    from CSVGenerator import checkCSV


class TeacherPipeline:

    def __init__(self):
        self.dataset_Path = ""
        self.batch_size = 1

    @staticmethod
    def defineModel():
        # Define Model
        #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               #in_channels=30, out_channels=130, init_features=32, pretrained=False)

        model = U_Net()
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
        modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_Teacher.pth"
        modelPath_bestweight = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_bw_Teacher.pth"
        csv_file = "dataset_teacher.csv"
        transform_val = (32, 256, 256)
        transform = tio.CropOrPad(transform_val)
        num_epochs = 1000
        dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        checkCSV(dataset_Path=dataset_path, csv_FileName=csv_file, overwrite=True)
        dataset = TeacherCustomDataset(dataset_path, csv_file, transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        dataloaders = [train_loader, validation_loader]
        train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, optimizer, log=True)


obj = TeacherPipeline()
obj.trainModel()
