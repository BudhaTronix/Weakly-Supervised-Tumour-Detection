import os
import sys
import torch

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

torch.set_num_threads(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(ROOT_DIR)
sys.path.insert(1, ROOT_DIR + "/")
sys.path.insert(0, ROOT_DIR + "/")

from Code.Semi_supervised.mscgunet.train import Mscgunet
from Code.Semi_supervised.Test.UnifiedTraining.dataloader import CustomDataset
from Code.Semi_supervised.Test.test import test
from Code.Utils.CSVGenerator import checkCSV_Student
from Model.M0 import U_Net_M0


class Pipeline:
    def __init__(self, device):
        self.batch_size = 1
        # Model Weights
        self.modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/"
        self.M0_model_path = self.modelPath + "M0.pth"
        self.M0_bw_path = self.modelPath + "M0_bw.pth"

        self.M1_model_path = self.modelPath + "M2.pth"
        self.M1_bw_path = self.modelPath + "M2_bw.pth"

        self.train_split = .9

        self.csv_file = "dataset.csv"
        self.num_epochs = 5000
        self.dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        self.logPath = "runs/Training/"

        self.scale_factor = 0.4
        self.transform_val = (32, 128, 128)

        self.device = device

        self.lr = 1e-4

    @staticmethod
    def defineModelM0():
        model = U_Net_M0()
        return model

    def testModel(self):
        print("Loading Models")
        modelM0 = self.defineModelM0()
        modelM0.load_state_dict(torch.load(self.M0_model_path))
        modelM0.to(self.device)

        modelM1 = Mscgunet(device=self.device)
        modelM1.initializeModel(self.M1_model_path)

        print("Models Loaded")

        checkCSV_Student(dataset_Path=self.dataset_path, csv_FileName=self.csv_file, overwrite=False)
        test_dataset = CustomDataset(self.dataset_path, self.csv_file, self.transform_val)

        print("Entering testing!")

        # Training and Validation Section
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        test(test_loader, modelM0, modelM1, log=False, logPath=self.logPath)


obj = Pipeline()
obj.testModel()
