import os
import sys
import torch

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.set_num_threads(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(ROOT_DIR)
sys.path.insert(1, ROOT_DIR + "/")
sys.path.insert(0, ROOT_DIR + "/")

from Code.Semi_supervised.Train.Model_M0.M0_main import M0_Pipeline
from Code.Semi_supervised.Train.Model_M1.M1_main import M1_Pipeline
from Code.Semi_supervised.Test.main import Test_Pipeline


class Pipeline:
    def __init__(self, dataset_path, modelWeights_path, log_path, dataset_type, loss_fn, model_type, M1_model_path=None,
                 M1_bw_path=None,
                 isM0Frozen=False, isM1Frozen=False, device="cuda", seed_value=42):
        self.dataset_type = dataset_type

        if self.dataset_type == "chaos":
            self.isChaos = True
        else:
            self.isChaos = False
            self.dataset_type = "clinical"

        self.isM0Frozen = isM0Frozen
        self.isM1Frozen = isM1Frozen

        self.loss_fn = loss_fn
        self.model_type = model_type

        # Model Weights
        if not self.isM0Frozen and not self.isM1Frozen:
            self.train_type = "Unified"
            self.temp_model_train_type = self.dataset_type + "_" + self.train_type
        elif self.isM0Frozen:
            self.train_type = "M0_Frozen"
            self.temp_model_train_type = self.dataset_type + "_" + self.train_type
        elif self.isM1Frozen:
            self.train_type = "M1_Frozen"
            self.temp_model_train_type = self.dataset_type + "_" + self.train_type

        self.temp_model_train_type = self.temp_model_train_type + "_" + self.loss_fn + "_" \
                                     + self.model_type + "_" + str(seed_value)

        self.modelWeights_path = modelWeights_path

        # Paths for pre training model M0
        self.M0_model_base = self.modelWeights_path + "M0_" + self.model_type + "_" + str(seed_value) + ".pth"
        self.M0_model_bw_base = self.modelWeights_path + "M0_bw_" + self.model_type + "_" + str(seed_value) + ".pth"

        # Paths for main training model M0 + M1
        self.M0_model_path = self.modelWeights_path + "M0_" + self.temp_model_train_type + ".pth"
        self.M0_bw_path = self.modelWeights_path + "M0_bw_" + self.temp_model_train_type + ".pth"

        if M1_model_path is None and M1_bw_path is None:
            self.M1_model_path = self.modelWeights_path + "M1_" + self.temp_model_train_type + ".pth"
            self.M1_bw_path = self.modelWeights_path + "M1_bw_" + self.temp_model_train_type + ".pth"
        else:
            self.M1_model_path = M1_model_path
            self.M1_bw_path = M1_bw_path

        self.csv_file = "dataset.csv"
        self.dataset_path = dataset_path
        self.logPath = log_path + "runs/" + self.temp_model_train_type

        self.device = device
        self.seed_value = seed_value

    def getModelM0(self, model_weights_path):
        obj = M0_Pipeline(self.dataset_path, self.M0_model_base, self.M0_model_bw_base, self.logPath, self.model_type)
        modelM0 = obj.defineModel()
        # Loading the best weights for Model M0
        modelM0.load_state_dict(torch.load(model_weights_path))
        modelM0.to(self.device)
        return modelM0

    def trainModel_M0(self, epochs):
        obj = M0_Pipeline(self.dataset_path, self.M0_model_base, self.M0_model_bw_base, self.loss_fn, self.model_type,
                          self.isChaos, self.device, self.logPath,
                          epochs=epochs, seed_val=self.seed_value)
        obj.trainModel()

    def trainModel_M1(self, model_M0, epochs, logger, TestModel=False):
        obj_M1 = M1_Pipeline(self.dataset_path, self.M1_model_path, self.M1_bw_path, self.loss_fn, self.model_type,
                             self.device, self.logPath,
                             self.isChaos, self.isM0Frozen, self.isM1Frozen, epochs, self.seed_value)
        train_loader, validation_loader, test_loader = obj_M1.train_val_test_slit()
        dataloaders = [train_loader, validation_loader]
        obj_M1.trainModel(model_M0, dataloaders, logger, self.M0_model_path, self.M0_bw_path)

        if self.dataset_type == "chaos" and TestModel:
            obj_Test = Test_Pipeline(self.M0_model_path, self.M0_bw_path, self.M1_model_path, self.M1_bw_path,
                                     self.dataset_path, self.logPath, self.device, self.loss_fn, self.model_type)
            obj_Test.testModel(test_loader)

        if self.dataset_type == "clinical" and TestModel:
            obj_Test = Test_Pipeline(self.M0_model_path, self.M0_bw_path, self.M1_model_path, self.M1_bw_path,
                                     self.dataset_path, self.logPath, self.device, self.loss_fn, self.model_type)
            obj_Test.testModel(test_loader)
