import os
import sys
import torch
import random
import numpy
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(1, ROOT_DIR + "/")
sys.path.insert(0, ROOT_DIR + "/")
from Code.Semi_supervised.Train.Pipeline import Pipeline

torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
random.seed(42)
numpy.random.seed(42)

modelWeights_path = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/"
log_path = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/Logs/"
chaos_dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
clinical_dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"


##################################################
def chaos_unified():
    """
        Step 1:
        Train Chaos - Unified training
        Test Chaos
    """
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", isUnified=True, device="cuda:1")
    obj.trainModel_M0(epochs=200, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=5000, logger=True, M0_model_path=obj.M0_model_path, M0_bw_path=obj.M0_bw_path)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def chaos_frozen():
    """
        Step 2 :
        Train Chaos - Freeze M0 training
        Test Chaos
    """
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", isUnified=False, device="cuda:2")
    obj.trainModel_M0(epochs=500, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=5000, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


##################################################
def clinical_unified():
    """
        Step 3 :
        Train Clinical - Unified training
        Test Clinical
    """
    obj = Pipeline(clinical_dataset_path, modelWeights_path, log_path, "clinical", isUnified=True, device="cuda:3")
    obj.trainModel_M0(epochs=200, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=5000, logger=True, M0_model_path=obj.M0_model_path, M0_bw_path=obj.M0_bw_path)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def clinical_frozen():
    """
        Step 4 :
        Train Clinical - Freeze M0 training
        Test Clinical
    """
    obj = Pipeline(clinical_dataset_path, modelWeights_path, log_path, "clinical", isUnified=False, device="cuda:7")
    obj.trainModel_M0(epochs=500, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=5000, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


##################################################
def main():
    print('cmd entry:', sys.argv)
    if sys.argv[1] == "1":
        print("Executing chaos unified")
        chaos_unified()
    elif sys.argv[1] == "2":
        print("Executing chaos M0 Frozen")
        chaos_frozen()
    elif sys.argv[1] == "3":
        print("Executing Clinical unified")
        clinical_unified()
    elif sys.argv[1] == "4":
        print("Executing Clinical M0 frozen")
        clinical_frozen()
    else:
        print("Wrong Argument")


if __name__ == "__main__":
    main()
