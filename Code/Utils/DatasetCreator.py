import glob
import os
import tempfile
from pathlib import Path

import SimpleITK as sitk
import torch
from tqdm import tqdm

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("Current temp directory:", tempfile.gettempdir())
tempfile.tempdir = "/home/mukhopad/tmp/test"
print("Temp directory after change:", tempfile.gettempdir())


def createGT(path, outPath, subject):
    str_to_search = path + str(subject) + "*.png"
    file_names = glob.glob(str_to_search)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    vol = reader.Execute()
    str_outpath = outPath + str(subject) + ".nii.gz"
    sitk.WriteImage(vol, str_outpath)


def createSubject(path):
    path = Path(path)
    for img_file_name in tqdm(sorted(path.glob("*"))):
        print(img_file_name)


path = "/project/mukhopad/"
outPath = "/project/mukhopad/"
subject = 1
