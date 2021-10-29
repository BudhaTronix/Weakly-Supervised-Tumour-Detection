from pathlib import Path
import ants
import torch
import torchio as tio
import os
import pytorch_ssim

torch.set_num_threads(1)

import tempfile

tempfile.tempdir = '/var/tmp/mukhopad/'
print('gettempdir():', tempfile.gettempdir())

imgPath = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/images/"
gtPath = "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/ct/"
translation_types = ["SyN", "Affine", "BOLDAffine", "BOLDRigid", "QuickRigid", "Rigid", "Similarity", "SyNgrid", "SyNRA", "Translation"]
imgPath = Path(imgPath)
gtPath = Path(gtPath)
save = False

for mri_file_name in sorted(imgPath.glob("*")):
    for ct_file_name in sorted(gtPath.glob("*")):
        if mri_file_name.name == ct_file_name.name:
            print("#"*100)
            print("\nWorking on Subject ID", mri_file_name.name)
            for typ in range(len(translation_types)):
                print("\nType of transformation: ", translation_types[typ])
                m = ants.image_read(str(mri_file_name))
                f = ants.image_read(str(ct_file_name))
                mygr = ants.create_warped_grid(m)
                mytx = ants.registration(fixed=f, moving=m, type_of_transform=translation_types[typ])
                # mywarpedimage = ants.apply_transforms(fixed=f, moving=m, transformlist=mytx['fwdtransforms'])
                img1 = m.numpy()
                img2 = mytx['warpedmovout'].numpy()
                img2 = torch.Tensor(img2)[:,:,0:img1.shape[2]]
                img2 = tio.ScalarImage(tensor=img2.unsqueeze(0))


                transform_val = img1.shape
                transform = tio.CropOrPad(transform_val)
                img2 = transform(img2)
                img2 = img2[tio.DATA].squeeze(0)

                if torch.cuda.is_available():
                    img1 = torch.Tensor(img1)
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                ssim = pytorch_ssim.ssim(img1.permute(2,1,0).unsqueeze(0), img2.permute(2,1,0).unsqueeze(0))
                print("SSIM :", ssim.item())
                print()

                if save:
                    moving_filename = "output_Translation"
                    ants.image_write(mytx['warpedmovout'], '/home/mady/DL/OP/1/' + moving_filename + ".nii.gz",
                                     ri=False)
            break

os.rmdir('/var/tmp/mukhopad/')
