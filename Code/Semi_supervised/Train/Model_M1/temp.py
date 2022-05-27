import torchio as tio
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def getPercentileRates(compute=False):
    if not compute:
        rates = [145.0, 154.0, 166.0, 186.0, 224.0]
        return rates
    df = pd.read_csv("/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/dataset.csv", header=None)
    path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/ct/"
    data = []
    for i in tqdm(range(len(df[0]))):
        ct = tio.ScalarImage(path + df[0][i])[tio.DATA]
        data.extend(torch.flatten(ct.squeeze()).cpu().detach().numpy())

    rates = []
    for j in range(94, 99):
        print(j, " Percentile:", np.percentile(data, j))
        rates.append(j)

    return rates


def show_slice_window(slice, level, window):
   """
   Function to display an image slice
   Input is a numpy 2D array
   """
   max = level + window/2
   min = level - window/2
   slice = slice.clip(min,max)

   return slice
   # slice = (slice - slice.min()) / (slice.max() - slice.min())
   # plt.figure()
   # plt.imshow(slice.T, cmap="gray", origin="lower")
   # plt.show()

rates = [145.0, 154.0, 166.0, 186.0, 224.0]
"""df = pd.read_csv("/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/dataset.csv", header=None)
path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/ct/"
for i in tqdm(range(len(df[0]))):
    ct = tio.ScalarImage(path + df[0][i])[tio.DATA]
    img = ct.squeeze()[:,:,45].numpy()
    show_slice_window(slice=img, level=50, window=350)"""

import glob
path = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/Clinical/ct/*"
files = glob.glob(path)
for file in files:
    ct = tio.ScalarImage(file)[tio.DATA].squeeze()
    for i in range(ct.shape[2]):
        ct[:,:,i] = show_slice_window(slice=ct[:,:,i], level=50, window=350)
        plt.figure()
        plt.imshow(ct[:,:,i], cmap="gray", origin="lower")
        plt.show()
        # img = ct.squeeze()[:,:,int(ct.squeeze().shape[2]/2)].numpy()

    break