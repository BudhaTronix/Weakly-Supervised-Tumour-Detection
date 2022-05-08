import numpy as np
import nibabel as nib
from PIL import Image
import os

folder = "/project/mukhopad/tmp_data_liv/CT/"
fp = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/ct_gt/"

for directory in sorted(os.listdir(folder)):
    print("\nWorking on directory: ", directory)
    images = []
    for filename in sorted(os.listdir(os.path.join(folder, directory, 'Ground')), reverse=True):
        print(filename)
        img = Image.open(os.path.join(folder, directory, 'Ground', filename))
        data = np.array(img) * 1
        data = np.rot90(data, k=3)
        images.append(data)
    path = fp + str(directory) + ".nii.gz"
    nib.save(nib.Nifti2Image(np.array(images, dtype=np.int32), np.eye(4)), path)
