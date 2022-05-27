import sys

import torch
import torch.optim as optim
import torchio as tio

from Code.Semi_supervised.Train.Model_M0.M0_dataloader import TeacherCustomDataset
from Code.Semi_supervised.Train.Model_M0.M0_train import train
from Model.M0 import U_Net_M0

torch.set_num_threads(1)

try:
    from Code.Utils.CSVGenerator import checkCSV
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/LiverTumorSeg/Code/Utils/')
    from CSVGenerator import checkCSV


class M0_Pipeline:
    def __init__(self, dataset_path, M0_model_path, M0_bw_path, device="cuda", log_path="runs/Training/", epochs=100):
        self.batch_size = 1

        # Model Weights
        self.M0_model_path = M0_model_path
        self.M0_bw_path = M0_bw_path

        self.dataset_path = dataset_path
        self.logPath = log_path + "_Model_M0/"
        self.csv_file = "dataset_teacher.csv"
        self.transform_val = (32, 256, 256)
        self.num_epochs = epochs
        self.device = device

    @staticmethod
    def defineModel():
        model = U_Net_M0()
        return model

    @staticmethod
    def defineOptimizer(model):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return optimizer

    def displayDetails(self, logging):
        print("\n" + "#"*150)
        print("Logging Path     : ", self.logPath)
        print("Model M0 Path    : ", self.M0_model_path)
        print("Model M0 BW Path : ", self.M0_bw_path)
        print("Device           : ", self.device)
        print("Logging Enabled  : ", logging)
        print("Epochs total     : ", self.num_epochs)
        print("#" * 150 + "\n")

    def trainModel(self, logging):
        self.displayDetails(logging)

        model = self.defineModel()
        optimizer = self.defineOptimizer(model)

        transform = tio.CropOrPad(self.transform_val)

        checkCSV(dataset_Path=self.dataset_path, csv_FileName=self.csv_file, overwrite=True)
        dataset = TeacherCustomDataset(self.dataset_path, self.csv_file, transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        dataloaders = [train_loader, validation_loader]
        train(dataloaders, self.M0_model_path, self.M0_bw_path, self.num_epochs, model, optimizer, self.device,
              log=logging, logPath=self.logPath)
