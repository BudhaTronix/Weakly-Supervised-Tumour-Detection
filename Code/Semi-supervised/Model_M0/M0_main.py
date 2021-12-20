import sys
import torch
import torch.optim as optim
import torchio as tio
from M0_dataloader import TeacherCustomDataset
from M0_train import train
from Model.M0 import U_Net_M0

torch.set_num_threads(1)

try:
    from Code.Utils.CSVGenerator import checkCSV
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/LiverTumorSeg/Code/Utils/')
    from CSVGenerator import checkCSV


class Pipeline:
    def __init__(self):
        self.batch_size = 1

        # Model Weights
        self.modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/"
        self.M0_model_path = self.modelPath + "M0.pth"
        self.M0_bw_path = self.modelPath + "M0_bw.pth"

        self.dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        self.csv_file = "dataset_teacher.csv"
        self.transform_val = (32, 256, 256)
        self.num_epochs = 1000

    @staticmethod
    def defineModel():
        model = U_Net_M0()
        return model

    @staticmethod
    def defineOptimizer(model):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return optimizer

    def trainModel(self):
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
        train(dataloaders, self.M0_model_path, self.M0_bw_path, self.num_epochs, model, optimizer, log=False)


obj = Pipeline()
obj.trainModel()
