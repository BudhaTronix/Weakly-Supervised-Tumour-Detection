from tqdm import tqdm
import torchio as tio
from pathlib import Path
import torch.nn.functional as f

ctPath = Path("/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/ct")
mriPath = Path("/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/mri")

for ct_file_name in sorted(ctPath.glob("*")):
    for mri_file_name in sorted(mriPath.glob("*")):
        if ct_file_name.name == mri_file_name.name:
            print(mri_file_name)
            xf = 2
            yf = 2
            zf = 2
            mri = tio.ScalarImage(mri_file_name)[tio.DATA].permute(0, 3, 1, 2)
            new_shape = (int(mri.shape[1]*zf),
                         int(mri.shape[2]*xf),
                         int(mri.shape[3]*yf))
            print("Shape:", new_shape)
            mri_transformed = f.interpolate(mri.unsqueeze(0), size=new_shape)
            print(mri_transformed.shape)
            exit()