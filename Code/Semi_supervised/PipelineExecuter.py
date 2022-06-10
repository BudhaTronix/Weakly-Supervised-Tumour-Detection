import os
import sys
import torch
import random
import numpy
import logging
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(1, ROOT_DIR + "/")
sys.path.insert(0, ROOT_DIR + "/")
from Code.Semi_supervised.Train.Pipeline import Pipeline

torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
random.seed(42)
numpy.random.seed(42)

modelWeights_path = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/temp/"
log_path = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/Logs/runs/"
chaos_dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
clinical_dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"

SEED = 42
M0_EPOCHS = 250
M1_EPOCHS = 1200

log_date = datetime.now().strftime("%Y.%m.%d")
logging.getLogger('matplotlib.font_manager').disabled = True


def preTrainM0(isChaos=True, train=False):
    """
    Use for pre training model M0
    :return: best weights model path
    """
    if isChaos:
        obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", device="cuda:1", seed_value=SEED)
    else:
        obj = Pipeline(clinical_dataset_path, modelWeights_path, log_path, "chaos", device="cuda:1", seed_value=SEED)
    if train:
        obj.trainModel_M0(epochs=M0_EPOCHS)
    else:
        logging.info("############################# Model M0 : Using saved weights #############################")
    M0_model_path = obj.M0_model_base
    M0_model_path_bw = obj.M0_model_bw_base

    return M0_model_path, M0_model_path_bw


##################################################
def chaos_unified_M0_M1():
    """
        Step 1:
        Train Chaos - Unified training
        Test Chaos
    """
    log_file_path = log_path + "Chaos_Unified_{}_{}".format(SEED, log_date) + "_log.txt"
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG)

    # Pre train the M0 model
    M0_model_path, M0_model_path_bw = preTrainM0(train=True)

    # Train the mode in combined mode : isM0Frozen=False, isM1Frozen=False
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", device="cuda:1", seed_value=SEED)
    modelM0 = obj.getModelM0(M0_model_path_bw)
    obj.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def chaos_frozen_M0():
    """
        Step 2 :
        Train Chaos - Freeze M0 training
        Test Chaos
    """
    # Logging
    log_file_path = log_path + "Chaos_M0_Frozen_{}_{}".format(SEED, log_date) + "_log.txt"
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG)

    # Pre train the M0 model
    M0_model_path, M0_model_path_bw = preTrainM0()

    # Part 2: Train model M1 while M0 frozen [isM0Frozen=False, isM1Frozen=False]
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos",
                   isM0Frozen=True, isM1Frozen=False, device="cuda:6", seed_value=SEED)
    modelM0 = obj.getModelM0(M0_model_path_bw)
    obj.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def chaos_frozen_M0_M1():
    """
        Step 2 :
        Train Chaos
            - Train M0 model for 250 epochs
            - Train M1 model while freezing M0 Model
            - Train M0 model again while freezing M1 model
        Test Chaos
    """
    CUDA = "cuda:6"
    # Logging
    log_file_path = log_path + "Chaos_M0_M1_Frozen_{}_{}".format(SEED, log_date) + "_log.txt"
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG)
    # logging.getLogger().removeHandler(logging.getLogger().handlers[0])

    # Part 1: Pre train the M0 model
    M0_model_path, M0_model_path_bw = preTrainM0(train=False)

    # Part 2: Train model M1 while M0 frozen [isM0Frozen=False, isM1Frozen=False]
    obj_0 = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos",
                     isM0Frozen=True, isM1Frozen=False, device=CUDA, seed_value=SEED)
    modelM0 = obj_0.getModelM0(M0_model_path_bw)
    obj_0.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True)

    # Part 3: Train model M0 while M1 frozen [isM0Frozen=False, isM1Frozen=True]
    # Passing the M1 weight paths and Model M0 from previous object
    obj_1 = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos",
                     M1_model_path=obj_0.M1_model_path, M1_bw_path=obj_0.M1_bw_path,
                     isM0Frozen=False, isM1Frozen=True, device=CUDA, seed_value=SEED)
    obj_1.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True, TestModel=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


##################################################
def clinical_unified_M0_M1():
    """
        Step 3 :
        Train Clinical - Unified training
        Test Clinical
    """
    # Logging
    log_file_path = log_path + "Clinical_Unified_{}_{}".format(SEED, log_date) + "_log.txt"
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG)

    # Pre train the M0 model
    M0_model_path, M0_model_path_bw = preTrainM0(isChaos=False, train=True)

    # Train the mode in combined mode : isM0Frozen=False, isM1Frozen=False
    obj = Pipeline(clinical_dataset_path, modelWeights_path, log_path, "clinical",
                   isM0Frozen=True, isM1Frozen=False, device="cuda:5", seed_value=SEED)
    modelM0 = obj.getModelM0(M0_model_path_bw)
    obj.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def clinical_frozen_M0():
    """
        Step 4 :
        Train Clinical - Freeze M0 training
        Test Clinical
    """
    # Logging
    log_file_path = log_path + "Clinical_M0_Frozen_{}_{}".format(SEED, log_date) + "_log.txt"
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG)

    # Pre train the M0 model
    M0_model_path, M0_model_path_bw = preTrainM0(isChaos=False, train=True)
    # Train the mode in combined mode : isM0Frozen=False, isM1Frozen=False
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "clinical",
                   isM0Frozen=True, isM1Frozen=False, device="cuda:4", seed_value=SEED)
    modelM0 = obj.getModelM0(M0_model_path_bw)
    obj.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


##################################################
def main():
    print('cmd entry:', sys.argv)
    if sys.argv[1] == "1":
        print("Executing chaos unified")
        # chaos_unified()
    elif sys.argv[1] == "2":
        print("Executing chaos M0 Frozen")
        # chaos_M0frozen()
    elif sys.argv[1] == "3":
        print("Executing Clinical unified")
        # clinical_unified()
    elif sys.argv[1] == "4":
        print("Executing Clinical M0 frozen")
        # clinical_frozen()
    else:
        print("Wrong Argument")


"""if __name__ == "__main__":
    main()"""

chaos_frozen_M0_M1()
