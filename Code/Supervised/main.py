import os
import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
from Code.Utils.loss import DiceLoss
from train import train
from dataloader import CustomDataset
os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'


def defineModel():
    # Define Model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=1, out_channels=1, init_features=32, pretrained=False)
    return model


def defineOptimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    modelPath = ""
    modelPath_bestweight = ""
    dataset_path = ""
    csv_file = ""
    inpImg = ""
    m, s = np.mean(inpImg, axis=(0, 1)), np.std(inpImg, axis=(0, 1))
    transform = getTransform(m,s)
    num_epochs = 1000
    dataloaders = CustomDataset(dataset_path, csv_file, transform)
    train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model, criterion, optimizer, log=False)



