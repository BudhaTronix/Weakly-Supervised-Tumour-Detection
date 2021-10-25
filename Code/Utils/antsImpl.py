import ants


def getWarp_antspy(moving, fixed):
    #mi = ants.image_read(moving)
    #fi = ants.image_read(fixed)
    mi = ants.from_numpy(moving)
    fi = ants.from_numpy(fixed)
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', verbose=False, outprefix="/scratch/mukhopad/")

    return mytx['fwdtransforms']


def applyTransformation(fixed, moving, transformation):
    mi = ants.from_numpy(moving)
    fi = ants.from_numpy(fixed)
    mywarpedimage = ants.apply_transforms(fixed=fi, moving=mi,
                                          transformlist=transformation)
    return mywarpedimage



def getWarp_simpleITK(img1, img2):
    fi = ants.image_read(ants.get_ants_data('r16'))  # Sample Input
    mi = ants.image_read(ants.get_ants_data('r64'))  # Sample Input
    # mygr = ants.create_warped_grid(mi)
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
    """mywarpedgrid = ants.create_warped_grid(mi, grid_directions=(False, True),
                                           transform=mytx['fwdtransforms'], fixed_reference_image=fi)"""

    return mytx['fwdtransforms']


def getMutalInfo(img1, img2):
    fi = ants.image_read(ants.get_ants_data('r16')).clone('float')
    mi = ants.image_read(ants.get_ants_data('r64')).clone('float')
    mival = ants.image_mutual_information(fi, mi)

    return mival


"""moving = r'/project/tawde/DL_Liver/NewDataforReg/CTDIcom/1/Nifty/MR1nifty.nii.gz'
fixed = r'/project/tawde/DL_Liver/NewDataforReg/CTDIcom/1/Nifty/CT1nifty.nii.gz'
a = getWarp_antspy(moving, fixed)"""
