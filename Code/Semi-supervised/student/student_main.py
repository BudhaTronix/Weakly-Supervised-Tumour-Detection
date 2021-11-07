import os
import sys
import torch
import torch.optim as optim
import torchio as tio
from torchvision import transforms
from student_dataloader import StudentCustomDataset
from student_train import train
from Code.Utils.antsImpl import getWarp_antspy, applyTransformation

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

torch.set_num_threads(1)
from Model.unet3d import U_Net

try:
    from Code.Utils.CSVGenerator import checkCSV_Student
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/LiverTumorSeg/Code/Utils/')
    from CSVGenerator import checkCSV_Student


class TeacherPipeline:

    def __init__(self):
        self.dataset_Path = ""
        self.batch_size = 1

    @staticmethod
    def defineModel():
        # Define Model
        """model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=30, out_channels=30, init_features=32, pretrained=False)"""
        model = U_Net()
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        return model

    @staticmethod
    def defineOptimizer(model):
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        return optimizer

    @staticmethod
    def getWarp(img1, img2):
        return 0

    @staticmethod
    def getTransform(m, s):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ])

        return preprocess

    def storeWarp(self):
        csv_file = "dataset.csv"
        transform_val = (16, 16, 16)
        transform = tio.CropOrPad(transform_val)
        dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        checkCSV_Student(dataset_Path=dataset_path, csv_FileName=csv_file, overwrite=True)
        dataset = StudentCustomDataset(dataset_path, csv_file, transform)

        dataloaders = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for batch in dataloaders:
            mri_batch, ct_batch, labels_batch = batch

            getWarpVal = getWarp_antspy(mri_batch.detach().cpu().squeeze().numpy(),
                                        ct_batch.detach().cpu().squeeze().numpy())

            print(getWarpVal)

    def trainModel(self):
        model = self.defineModel()
        optimizer = self.defineOptimizer(model)
        modelPath = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_Student.pth"
        modelPath_bestweight = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_bw_Student.pth"
        model_Path_trained = "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi-supervised/model_weights/UNet_Teacher.pth"
        csv_file = "dataset.csv"
        transform_val = (32, 256, 256)
        transform = tio.CropOrPad(transform_val)
        t_ct = tio.CropOrPad((32, 512, 512))
        num_epochs = 1000
        dataset_path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/"
        checkCSV_Student(dataset_Path=dataset_path, csv_FileName=csv_file, overwrite=True)
        dataset = StudentCustomDataset(dataset_path, csv_file, transform, t_ct)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Training and Validation Section
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        dataloaders = [train_loader, validation_loader]

        train(dataloaders, modelPath, modelPath_bestweight, num_epochs, model,
              optimizer, log=True, model_Path_trained=model_Path_trained)


obj = TeacherPipeline()
obj.trainModel()
