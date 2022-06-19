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

log_date = datetime.now().strftime("%Y.%m.%d")
logging.getLogger('matplotlib.font_manager').disabled = True


def preTrainM0(SEED, device="cuda", isChaos=True, train=False):
    """
    Use for pre training model M0
    :return: best weights model path
    """
    if isChaos:
        obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", device=device, seed_value=SEED,
                       loss_fn=Loss_fn, model_type=Model_name)
    else:
        obj = Pipeline(clinical_dataset_path, modelWeights_path, log_path, "chaos", device=device, seed_value=SEED,
                       loss_fn=Loss_fn, model_type=Model_name)
    if train:
        obj.trainModel_M0(epochs=M0_EPOCHS)
    else:
        logging.info("############################# Model M0 : Using saved weights #############################")
    M0_model_path = obj.M0_model_base
    M0_model_path_bw = obj.M0_model_bw_base

    return M0_model_path, M0_model_path_bw


##################################################
def chaos_unified(cuda, seed):
    """
        Step 1:
            Pre train the M0 model
        Step 2 :
            Train Chaos - Unified training
        Step 3 :
            Test Chaos
    """
    CUDA = "cuda:{}".format(cuda)
    log_file_path = log_path + "Chaos_{}_{}_Unified_{}_{}".format(Model_name, Loss_fn, seed, log_date) + "_log.txt"
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG)

    # Pre train the M0 model
    M0_model_path, M0_model_path_bw = preTrainM0(SEED=seed, device=CUDA, train=False)

    # Train the mode in combined mode : isM0Frozen=False, isM1Frozen=False
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", isM0Frozen=False, isM1Frozen=False,
                   device=CUDA, seed_value=seed, loss_fn=Loss_fn, model_type=Model_name)
    modelM0 = obj.getModelM0(M0_model_path_bw)
    obj.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True, TestModel=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def chaos_sequential(cuda, seed):
    """
        Step 1:
            Pre train the M0 model
        Step 2 :
            Train Chaos
                - Train M1 model while freezing M0 Model
                - Train M0 model again while freezing M1 model
        Step 3:
            Test Chaos
    """
    CUDA = "cuda:{}".format(cuda)

    # Logging
    log_file_path = log_path + "Chaos_{}_{}_Sequential_{}_{}".format(Model_name, Loss_fn, seed, log_date) + "_log.txt"
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG)

    # Part 1: Pre train the M0 model
    M0_model_path, M0_model_path_bw = preTrainM0(SEED=seed, device=CUDA, train=False)

    # Part 2: Train model M1 while M0 frozen [isM0Frozen=False, isM1Frozen=False]
    obj_0 = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos",
                     isM0Frozen=True, isM1Frozen=False, device=CUDA,
                     seed_value=seed, loss_fn=Loss_fn, model_type=Model_name)
    modelM0 = obj_0.getModelM0(M0_model_path_bw)
    obj_0.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True)

    # Part 3: Train model M0 while M1 frozen [isM0Frozen=False, isM1Frozen=True]
    # Passing the M1 weight paths and Model M0 from previous object
    obj_1 = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos",
                     M1_model_path=obj_0.M1_model_path, M1_bw_path=obj_0.M1_bw_path,
                     isM0Frozen=False, isM1Frozen=True, device=CUDA,
                     seed_value=seed, loss_fn=Loss_fn, model_type=Model_name)

    obj_1.trainModel_M1(modelM0, epochs=M1_EPOCHS, logger=True, TestModel=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


##################################################
def clinical_unified(SEED):
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


def clinical_sequential(SEED):
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
        print("Executing Chaos Unified - cuda:{} , seed-value:{}, Loss-Function:{}, Model:{}".format(sys.argv[2], sys.argv[3], Loss_fn, Model_name))
        chaos_unified(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "2":
        print("Executing Chaos Sequential - cuda:{} , seed-value:{}, Loss-Function:{}, Model:{}".format(sys.argv[2], sys.argv[3], Loss_fn, Model_name))
        chaos_sequential(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "3":
        print("Executing Clinical unified")
        # clinical_unified()
    elif sys.argv[1] == "4":
        print("Executing Clinical M0 frozen")
        # clinical_frozen()
    else:
        print("Wrong Argument")

##################################################
modelWeights_path = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/"
log_path = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/Logs/runs/"
chaos_dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
clinical_dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"

M0_EPOCHS = 250
M1_EPOCHS = 1200

Loss_fn = "TFL"
# Loss_fn = "Dice"

# Model_name = "DeepSup"
Model_name = "Unet"


if __name__ == "__main__":
    main()
    # PipelineExecutor.py --ExecutionType --CUDA --SEED
