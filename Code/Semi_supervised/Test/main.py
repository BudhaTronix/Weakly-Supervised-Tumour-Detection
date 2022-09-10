import os
import sys
import torch
import logging

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

torch.set_num_threads(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(ROOT_DIR)
sys.path.insert(1, ROOT_DIR + "/")
sys.path.insert(0, ROOT_DIR + "/")

from Code.Semi_supervised.mscgunet.train import Mscgunet
from Code.Semi_supervised.Train.Model_M1.M1_dataloader import CustomDataset
from Code.Semi_supervised.Test.test import test, test_clinical
from Code.Utils.CSVGenerator import checkCSV_Student
from Model.M0 import U_Net_M0
from Model.DeepSupAttUNet3D import DeepSupAttentionUnet


class Test_Pipeline:
    def __init__(self, M0_model_path, M0_bw_path, M1_model_path, M1_bw_path, dataset_path, logPath,
                 device, loss_fn, model_type):
        # Model Weights
        self.M0_model_path = M0_model_path
        self.M0_bw_path = M0_bw_path
        self.M1_model_path = M1_model_path
        self.M1_bw_path = M1_bw_path
        self.logPath = logPath + "_Test/"

        self.batch_size = 1
        self.lr = 1e-4

        self.csv_file = "dataset.csv"

        self.dataset_path = dataset_path

        self.scale_factor = 0.4
        self.transform_val = (32, 128, 128)
        self.ct_level = 50
        self.ct_window = 350

        self.device = device
        self.isChaos = True

        self.loss_fn = loss_fn
        self.model_type = model_type

    def defineModelM0(self):
        if self.model_type == "DeepSup":
            model = DeepSupAttentionUnet(1, 1)
        else:
            model = U_Net_M0()
        return model

    def displayDetails(self, logger):
        logging.info("\n\n\n")
        logging.info("############################# START Model Testing #############################")
        logging.info("Logging Path     : " + self.logPath)
        logging.info("Model M0 Path    : " + self.M0_model_path)
        logging.info("Model M0 BW Path : " + self.M0_bw_path)
        logging.info("Model M1 Path    : " + self.M1_model_path)
        logging.info("Model M1 BW Path : " + self.M1_bw_path)
        logging.info("Device           : " + str(self.device))
        logging.info("Logging Enabled  : " + str(logger))
        logging.info("M0 Loss Function : " + self.loss_fn)
        logging.info("M0 Model Type    : " + self.model_type)

    def testModel(self, test_loader=None, logger=True):
        self.displayDetails(logger)

        logging.debug("Loading Models for Testing")
        modelM0 = self.defineModelM0()
        modelM0.load_state_dict(torch.load(self.M0_bw_path,map_location=self.device))

        modelM1 = Mscgunet(device=self.device)
        modelM1.initializeModel(self.M1_bw_path)
        logging.debug("Models Loaded")

        if test_loader is None:
            checkCSV_Student(dataset_Path=self.dataset_path, csv_FileName=self.csv_file, overwrite=False)
            test_dataset = CustomDataset(self.dataset_path, self.csv_file, self.transform_val,
                                         self.isChaos, self.ct_level, self.ct_window)
            # Training and Validation Section
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        # test(test_loader, modelM0, modelM1, model_type=self.model_type, logPath=self.logPath, device=self.device)
        test_clinical(test_loader, modelM0, modelM1, device=self.device)


"""m0 = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/M0_chaos_M1_Frozen_TFL_DeepSup.pth"
m0_bw = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/M0_bw_chaos_M1_Frozen_TFL_DeepSup.pth"

m1 = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/M1_chaos_M0_Frozen_TFL_DeepSup.pth"
m1_bw = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/M1_bw_chaos_M0_Frozen_TFL_DeepSup.pth"

d = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
l = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/Logs/runs/"

model_type = "DeepSup"
# model_type = "Unet"

loss_fn = "TFL"
# loss_fn = "Dice"

ob = Test_Pipeline(M0_model_path=m0, M0_bw_path=m0_bw, M1_model_path=m1, M1_bw_path=m1_bw, dataset_path=d,
                   logPath=l, device="cuda:4", model_type=model_type, loss_fn=loss_fn)
ob.testModel()"""
