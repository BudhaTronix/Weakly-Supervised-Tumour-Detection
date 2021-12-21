import os
import sys

import torch.optim as optim
import torchio as tio
from torchvision import transforms

from Code.Utils.loss import DiceLoss
from Model.unet import U_Net
from dataloader import CustomDataset
from train import train

try:
    from Code.Utils.CSVGenerator import checkCSV_Student
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/LiverTumorSeg/Code/Utils/')
    from CSVGenerator import checkCSV_Student

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'


def defineModel():
    # Define Model
    model = U_Net()
    return model


def defineOptimizer(model):
    params_to_update = model.parameters()
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    return optimizer


def defineCriterion():
    dsc_loss = DiceLoss()
    return dsc_loss


def getTransform(m, s):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s),
    ])

    return preprocess


def trainModel():
    model = defineModel()
    optimizer = defineOptimizer(model)
    criterion = defineCriterion
    modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Supervised/model_weights/m1.pth"
    modelPath_bestweight = "/project/mukhopad/tmp/LiverTumorSeg/Code/Supervised/model_weights/m1_bw.pth"
    dataset_path = "/project/tawde/DL_Liver/NewDataforReg/Dataset/"
    csv_file = "Data.csv"
    transform_val = (256, 256, 32)
    transform = tio.CropOrPad(transform_val)
    num_epochs = 1000
    # checkCSV_Student(dataset_Path=dataset_path, csv_FileName=csv_file, overwrite=True)
    csv_file = "/project/tawde/DL_Liver/NewDataforReg/Dataset/Data.csv"
    dataloaders = CustomDataset(dataset_path, csv_file, transform)

    train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, criterion, optimizer)


trainModel()
