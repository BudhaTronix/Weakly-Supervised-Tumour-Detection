from Code.Semi_supervised.Train.Pipeline import Pipeline
import logging
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
    obj.trainModel_M0(epochs=1, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=1, logger=True, M0_model_path=obj.M0_model_path, M0_bw_path=obj.M0_bw_path)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def chaos_frozen():
    """
        Step 2 :
        Train Chaos - Freeze M0 training
        Test Chaos
    """
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", isUnified=False, device="cuda:2")
    obj.trainModel_M0(epochs=1, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=1, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


##################################################
def clinical_unified():
    """
        Step 3 :
        Train Clinical - Unified training
        Test Clinical
    """
    obj = Pipeline(clinical_dataset_path, modelWeights_path, log_path, "clinical", isUnified=True, device="cuda:3")
    obj.trainModel_M0(epochs=1, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=1, logger=True, M0_model_path=obj.M0_model_path, M0_bw_path=obj.M0_bw_path)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def clinical_frozen():
    """
        Step 4 :
        Train Clinical - Freeze M0 training
        Test Clinical
    """
    obj = Pipeline(clinical_dataset_path, modelWeights_path, log_path, "clinical", isUnified=False, device="cuda:7")
    obj.trainModel_M0(epochs=1, logger=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=1, logger=True)
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])


##################################################
chaos_unified()
chaos_frozen()
clinical_unified()
clinical_frozen()
