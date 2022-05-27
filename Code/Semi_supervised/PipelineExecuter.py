from Code.Semi_supervised.Train.Pipeline import Pipeline

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
    obj = Pipeline(chaos_dataset_path, modelWeights_path,log_path, "chaos", isUnified=True, device="cuda:7")
    obj.trainModel_M0(epochs=200, logging=True)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0,epochs=5000, logging=False)


def chaos_frozen():
    """
        Step 2 :
        Train Chaos - Freeze M0 training
        Test Chaos
    """
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "chaos", isUnified=False, device="cuda:6")
    obj.trainModel_M0(epochs=500, logging=False)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=5000, logging=False)


##################################################
def clinical_unified():
    """
        Step 3 :
        Train Clinical - Unified training
        Test Clinical
    """
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "clinical", isUnified=True, device="cuda:6")
    obj.trainModel_M0(epochs=200)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=5000)


def clinical_frozen():
    """
        Step 4 :
        Train Clinical - Freeze M0 training
        Test Clinical
    """
    obj = Pipeline(chaos_dataset_path, modelWeights_path, log_path, "clinical", isUnified=False, device="cuda:6")
    obj.trainModel_M0(epochs=500)
    modelM0 = obj.getModelM0()
    obj.trainModel_M1(modelM0, epochs=5000)
##################################################
chaos_unified()