import os
import sys
import numpy as np
import torch
import logging

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.set_num_threads(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
sys.path.insert(1, ROOT_DIR + "/")
sys.path.insert(0, ROOT_DIR + "/")

from Code.Semi_supervised.Train.Model_M1.M1_dataloader import CustomDataset
from Code.Semi_supervised.Train.Model_M1.M1_train import train
from Code.Semi_supervised.mscgunet.train import Mscgunet
from Code.Utils.CSVGenerator import checkCSV_Student


class M1_Pipeline:
    def __init__(self, dataset_path, M1_model_path, M1_bw_path, device="cuda", log_path="runs/Training/",
                 isChaos=False, isUnified=False, epochs=3000):
        # Model Weights
        self.M1_model_path = M1_model_path
        self.M1_bw_path = M1_bw_path

        self.train_split = 0.8
        self.test_split = 0.15
        self.batch_size = 1
        self.lr = 1e-4

        self.csv_file = "dataset.csv"
        self.num_epochs = epochs
        self.dataset_path = dataset_path
        self.logPath = log_path + "_Model_M1/"

        self.scale_factor = 0.4
        self.transform_val = (32, 128, 128)
        self.ct_level = 50
        self.ct_window = 350

        self.device = device

        self.isChaos = isChaos
        self.isUnified = isUnified

    @staticmethod
    def defineOptimizer_unified(modelM0, modelM1):
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

    @staticmethod
    def defineOptimizer(modelM1):
        optimizer = torch.optim.Adam(
            list(modelM1.feature_extractor_training.parameters()) + list(modelM1.scg_training.parameters()) +
            list(modelM1.upsampler1_training.parameters()) + list(modelM1.upsampler2_training.parameters()) +
            list(modelM1.upsampler3_training.parameters()) + list(modelM1.upsampler4_training.parameters()) +
            list(modelM1.upsampler5_training.parameters()) + list(modelM1.graph_layers1_training.parameters()) +
            list(modelM1.graph_layers2_training.parameters()) + list(modelM1.conv_decoder1_training.parameters()) +
            list(modelM1.conv_decoder2_training.parameters()) + list(modelM1.conv_decoder3_training.parameters()) +
            list(modelM1.conv_decoder4_training.parameters()) + list(modelM1.conv_decoder5_training.parameters()) +
            list(modelM1.conv_decoder6_training.parameters()), lr=modelM1.lr)
        return optimizer

    def displayDetails(self, logger):

        logging.info("Logging Path     : " + self.logPath)
        logging.info("Model M1 Path    : " + self.M1_model_path)
        logging.info("Model M1 BW Path : " + self.M1_bw_path)
        logging.info("Device           : " + self.device)
        logging.info("Logging Enabled  : " + str(logger))
        logging.info("Epochs total     : " + str(self.num_epochs))

    def train_val_test_slit(self):
        logging.info("\n\n\n")
        if self.isUnified:
            logging.info("########################### START Unified M1 + M0 Model Training ###########################")
        else:
            logging.info("########################### START M1 Model Training ###########################")
        # Check dataset csv file
        checkCSV_Student(dataset_Path=self.dataset_path, csv_FileName=self.csv_file, overwrite=False)
        dataset = CustomDataset(self.dataset_path, self.csv_file, self.transform_val,
                                self.isChaos, self.ct_level, self.ct_window)

        # Train-Test Split
        train_size = int(self.train_split * len(dataset))
        test_size = int(self.test_split * len(dataset))
        val_size = len(dataset) - train_size - test_size

        logging.info("Train Subjects      : " + str(train_size))
        logging.info("Validation Subjects : " + str(val_size))
        logging.info("Test Subjects       : " + str(test_size))

        rs = np.random.RandomState(seed=1701)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                 [train_size, val_size, test_size],
                                                                                 random_state=rs)

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, validation_loader, test_loader

    def trainModel(self, modelM0, dataloaders, logging, M0_model_path=None, M0_bw_path=None):
        self.displayDetails(logging)
        # Initialize Model M1
        modelM1 = Mscgunet(device=self.device)

        # Initialize Optimizer
        if self.isUnified:
            optimizer = self.defineOptimizer_unified(modelM0, modelM1)
        else:
            optimizer = self.defineOptimizer(modelM1)

        train(dataloaders, self.M1_model_path, self.M1_bw_path, self.num_epochs, modelM0, modelM1, optimizer,
              self.isChaos, self.isUnified, self.device,
              log=logging,
              logPath=self.logPath,
              M0_model_path=M0_model_path,
              M0_bw_path=M0_bw_path)
