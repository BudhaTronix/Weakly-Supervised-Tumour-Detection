import ants
import numpy as np

"""

#fname1 = ants.get_ants_data('r16')
#fname2 = ants.get_ants_data('r64')
#print(fname1)

#img1 = ants.image_read(fname1)
#img2 = ants.image_read(fname2)
#print(img1)

#fixed = ants.image_read( ants.get_ants_data('r16') ).resample_image((64,64),1,0)
#moving = ants.image_read( ants.get_ants_data('r64') ).resample_image((64,64),1,0)

#f1 = ants.image_read("Pass_path")
#f = ants.image_read('Fixed_CT.nii').resample_image((64,64),1,0)
#f = ants.image_read('I0000815').resample_image((64,64),1,0)

"""
m = ants.image_read(r'/project/tawde/DL_Liver/NewDataforReg/CTDIcom/1/Nifty/MR1nifty.nii.gz')
f = ants.image_read(r'/project/tawde/DL_Liver/NewDataforReg/CTDIcom/1/Nifty/CT1nifty.nii.gz')
# f = ants.dicom_read('I0000815')
# f.plot(overlay=m, title='Before Registration')

# m = ants.dicom_read('MRI_1')
# print("test")
# f = ants.dicom_read('CTTest')
print("test")
mytx = ants.registration(fixed=f, moving=m, type_of_transform='Affine')
print(mytx)
warped_moving = mytx['warpedmovout']
"""f.plot(overlay=warped_moving,
       title='After Registration')"""
mywarpedimage = ants.apply_transforms(fixed=f, moving=m,
                                      transformlist=mytx['fwdtransforms'])
moving_filename = "output_affine"
ants.image_write(warped_moving, '/project/tawde/DL_Liver/NewDataforReg/CTDIcom/1/Nifty/Output/' + moving_filename + ".nii.gz",
ri=False)

# mywarpedimage.plot()

# Intensity normalization
