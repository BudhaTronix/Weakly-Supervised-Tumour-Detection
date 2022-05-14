import os
import sys
from layers import *


os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

torch.set_num_threads(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(ROOT_DIR)
sys.path.insert(1, ROOT_DIR + "/")
sys.path.insert(0, ROOT_DIR + "/")

from Code.Semi_supervised.mscgunet.train import Mscgunet
from Code.Semi_supervised.UnifiedTraining.dataloader import CustomDataset
from Code.Semi_supervised.UnifiedTraining.train import train

from Code.Utils.CSVGenerator import checkCSV_Student
from Model.M0 import U_Net_M0


class Pipeline:
    def __init__(self):
        self.batch_size = 1
        # Model Weights
        self.modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/"
        self.M0_model_path = self.modelPath + "M0.pth"
        self.M0_bw_path = self.modelPath + "M0_bw.pth"

        self.M1_model_path = self.modelPath + "M1.pth"
        self.M1_bw_path = self.modelPath + "M1_bw.pth"

        self.train_split = .9

        self.csv_file = "dataset.csv"
        self.num_epochs = 5000
        self.dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        self.logPath = "runs/Training/"

        self.scale_factor = 0.4
        self.transform_val = (32, 128, 128)

        self.GPU_ID_M0 = "cuda:5"
        self.GPU_ID_M1 = "cuda:5"

        self.lr = 1e-4

    @staticmethod
    def defineModelM0():
        model = U_Net_M0()
        return model

    @staticmethod
    def defineOptimizer(modelM0, modelM1):
        optimizer = torch.optim.Adam(
            list(modelM1.feature_extractor_training.parameters()) + list(modelM1.scg_training.parameters()) +
            list(modelM1.upsampler1_training.parameters()) + list(modelM1.upsampler2_training.parameters()) +
            list(modelM1.upsampler3_training.parameters()) + list(modelM1.upsampler4_training.parameters()) +
            list(modelM1.upsampler5_training.parameters()) + list(modelM1.graph_layers1_training.parameters()) +
            list(modelM1.graph_layers2_training.parameters()) + list(modelM1.conv_decoder1_training.parameters()) +
            list(modelM1.conv_decoder2_training.parameters()) + list(modelM1.conv_decoder3_training.parameters()) +
            list(modelM1.conv_decoder4_training.parameters()) + list(modelM1.conv_decoder5_training.parameters()) +
            list(modelM1.conv_decoder6_training.parameters()) + list(modelM0.parameters()),
            lr=modelM1.lr)

        return optimizer

    def trainModel(self):
        modelM0 = self.defineModelM0()
        modelM0.load_state_dict(torch.load(self.M0_model_path))
        modelM0.to(self.GPU_ID_M0)

        modelM1 = Mscgunet(device=self.GPU_ID_M1)

        optimizer = self.defineOptimizer(modelM0, modelM1)

        checkCSV_Student(dataset_Path=self.dataset_path, csv_FileName=self.csv_file, overwrite=False)
        dataset = CustomDataset(self.dataset_path, self.csv_file, self.transform_val)

        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        dataloaders = [train_loader, validation_loader]

        train(dataloaders, self.M1_model_path, self.M1_bw_path, self.M2_model_path, self.M2_bw_path,
              self.num_epochs, modelM0, modelM1, optimizer, log=True, logPath=self.logPath)


obj = Pipeline()
obj.trainModel()
