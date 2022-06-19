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
    def __init__(self, dataset_path, M1_model_path, M1_bw_path,loss_fn,model_type, device="cuda", log_path="runs/Training/",
                 isChaos=False, isM0Frozen=False, isM1Frozen=False, epochs=3000, seed_val=42):
        # Model Weights
        self.M1_model_path = M1_model_path
        self.M1_bw_path = M1_bw_path

        self.train_size = 5
        self.val_size = 1
        self.test_size = 2
        self.batch_size = 1
        self.lr = 1e-4

        self.loss_fn = loss_fn
        self.model_type = model_type

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
        self.isM0Frozen = isM0Frozen
        self.isM1Frozen = isM1Frozen

        self.seed = seed_val

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
    def defineOptimizer_M1(modelM1):
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

    @staticmethod
    def defineOptimizer_M0(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return optimizer

    def displayDetails(self, logger):
        logging.info("Logging Path     : " + self.logPath)
        logging.info("Model M1 Path    : " + self.M1_model_path)
        logging.info("Model M1 BW Path : " + self.M1_bw_path)
        logging.info("Device           : " + self.device)
        logging.info("Logging Enabled  : " + str(logger))
        logging.info("Epochs total     : " + str(self.num_epochs))
        logging.info("M0 Loss Function : " + str(self.loss_fn))
        logging.info("M0 Model Type    : " + str(self.model_type))

    def train_val_test_slit(self):
        logging.info("\n\n\n")
        if not self.isM0Frozen and not self.isM1Frozen:
            logging.info("########################### M0 - Train | M1 - Train  ###########################")
        elif self.isM0Frozen:
            logging.info("########################### M0 - Frozen| M1 - Train  ###########################")
        elif self.isM1Frozen:
            logging.info("########################### M0 - Train | M1 - Frozen ###########################")
        logging.info("Using Seed value : "+str(self.seed))
        # Check dataset csv file
        checkCSV_Student(dataset_Path=self.dataset_path, csv_FileName=self.csv_file, overwrite=False)
        dataset = CustomDataset(self.dataset_path, self.csv_file, self.transform_val,
                                self.isChaos, self.ct_level, self.ct_window)

        logging.info("Train Subjects      : " + str(self.train_size))
        logging.info("Validation Subjects : " + str(self.val_size))
        logging.info("Test Subjects       : " + str(self.test_size))

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                 [self.train_size, self.val_size, self.test_size],
                                                                                 generator=torch.Generator().manual_seed(self.seed))
        logging.info("Train Indices  : " + str(train_dataset.indices))
        logging.info("Val   Indices  : " + str(val_dataset.indices))
        logging.info("Test  Indices  : " + str(test_dataset.indices))

        print("Train Indices  : " + str(train_dataset.indices))
        print("Val   Indices  : " + str(val_dataset.indices))
        print("Test  Indices  : " + str(test_dataset.indices))

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   shuffle=True,generator=torch.Generator().manual_seed(self.seed))
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                                        shuffle=True,generator=torch.Generator().manual_seed(self.seed))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                  shuffle=True,generator=torch.Generator().manual_seed(self.seed))

        return train_loader, validation_loader, test_loader

    def trainModel(self, modelM0, dataloaders, logger, M0_model_path=None, M0_bw_path=None):
        self.displayDetails(logger)
        # Initialize Model M1
        modelM1 = Mscgunet(device=self.device)

        # Initialize Optimizer
        if not self.isM0Frozen and not self.isM1Frozen:
            # Optimizer for both M0 and M1
            optimizer = self.defineOptimizer_unified(modelM0, modelM1)
        elif not self.isM0Frozen and self.isM1Frozen:
            # Optimizer for both M0 only
            optimizer = self.defineOptimizer_M0(modelM0)
        else:
            # Optimizer for both M1 only
            optimizer = self.defineOptimizer_M1(modelM1)

        # if M1 is frozen - load best weights
        if self.isM1Frozen:
            modelM1.initializeModel(self.M1_bw_path)

        train(dataloaders, self.M1_model_path, self.M1_bw_path, self.num_epochs, modelM0, modelM1, optimizer,
              self.isChaos, self.isM0Frozen, self.isM1Frozen, self.device, self.loss_fn, self.model_type,
              log=logging,
              logPath=self.logPath,
              M0_model_path=M0_model_path,
              M0_bw_path=M0_bw_path)
