import os
import sys
import torch
import torch.optim as optim
import torchio as tio
from torchvision import transforms
from dataloader import CustomDataset
from train import train
from Code.Utils.antsImpl import getWarp_antspy, applyTransformation
from torch.nn.functional import interpolate

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

torch.set_num_threads(1)
from Model.unet3d import U_Net

try:
    from Code.Utils.CSVGenerator import checkCSV_Student
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/LiverTumorSeg/Code/Utils/')
    from CSVGenerator import checkCSV_Student


class TeacherPipeline:

    def __init__(self):
        self.dataset_Path = ""
        self.batch_size = 1
        # Model Weights
        self.modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/"
        self.M0_model_path = self.modelPath + "M0.pth"
        self.M0_bw_path = self.modelPath + "M0_bw.pth"

        self.M1_model_path = self.modelPath + "M1.pth"
        self.M1_bw_path = self.modelPath + "M1_bw.pth"

        self.M2_model_path = self.modelPath + "M2.pth"
        self.M2_bw_path = self.modelPath + "M2_bw.pth"

        self.csv_file = "dataset.csv"
        self.num_epochs = 1000
        self.dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"

    @staticmethod
    def defineModelM1():
        model = U_Net()
        return model

    @staticmethod
    def defineModelM2():
        model = U_Net()
        return model

    @staticmethod
    def defineOptimizer(modelM1, modelM2):
        optimizer = optim.Adam((list(modelM1.parameters()) + list(modelM2.parameters())), lr=0.01)
        return optimizer

    @staticmethod
    def getTransform(m, s):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ])
        return preprocess

    def storeWarp(self):
        csv_file = "dataset.csv"
        transform_val = (16, 16, 16)
        transform = tio.CropOrPad(transform_val)
        dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        checkCSV_Student(dataset_Path=dataset_path, csv_FileName=csv_file, overwrite=True)
        dataset = CustomDataset(dataset_path, csv_file, transform)

        dataloaders = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for batch in dataloaders:
            mri_batch, ct_batch, labels_batch = batch

            getWarpVal = getWarp_antspy(mri_batch.detach().cpu().squeeze().numpy(),
                                        ct_batch.detach().cpu().squeeze().numpy())

            print(getWarpVal)

    def trainModel(self):
        modelM0 = torch.load(self.M0_model_path)
        modelM1 = self.defineModelM1()
        modelM2 = self.defineModelM2()

        optimizer = self.defineOptimizer(modelM1, modelM2)

        transform_val = (32, 256, 256)
        transform = tio.CropOrPad(transform_val)
        t_ct = tio.CropOrPad((32, 256, 256))

        checkCSV_Student(dataset_Path=self.dataset_path, csv_FileName=self.csv_file, overwrite=True)
        dataset = CustomDataset(self.dataset_path, self.csv_file, transform, t_ct)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        dataloaders = [train_loader, validation_loader]

        train(dataloaders, self.M1_model_path, self.M1_bw_path, self.M2_model_path , self.M1_bw_path,
              self.num_epochs, modelM0, modelM1, modelM2,
              optimizer, log=True, )


obj = TeacherPipeline()
obj.trainModel()
