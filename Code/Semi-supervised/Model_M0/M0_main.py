import os
import sys
import torch
import torch.optim as optim
import torchio as tio
from torchvision import transforms
from M0_dataloader import TeacherCustomDataset
from M0_train import train

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.set_num_threads(1)

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
        self.modelPath_M0 = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_Teacher.pth"
        self.modelPath_bestweight_M0 = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_bw_Teacher.pth"
        self.dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        self.csv_file = "dataset_teacher.csv"
        self.transform_val = (32, 256, 256)
        self.num_epochs = 1000

    @staticmethod
    def defineModel():
        # Define Model
        # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        # in_channels=30, out_channels=130, init_features=32, pretrained=False)

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

        transform = tio.CropOrPad(self.transform_val)


        checkCSV(dataset_Path=self.dataset_path, csv_FileName=csv_file, overwrite=True)
        dataset = TeacherCustomDataset(self.dataset_path, self.csv_file, transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        dataloaders = [train_loader, validation_loader]
        train(dataloaders, self.modelPath, self.modelPath_bestweight, self.num_epochs, model, optimizer, log=True)


obj = TeacherPipeline()
obj.trainModel()
