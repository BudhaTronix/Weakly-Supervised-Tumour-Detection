import ants


def getWarp(img1, img2):
    fi = ants.image_read(ants.get_ants_data('r16'))
    mi = ants.image_read(ants.get_ants_data('r64'))
    mygr = ants.create_warped_grid(mi)
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
    mywarpedgrid = ants.create_warped_grid(mi, grid_directions=(False, True),
                                           transform=mytx['fwdtransforms'], fixed_reference_image=fi)

    return mywarpedgrid


def getMutalInfo(img1, img2):
    fi = ants.image_read(ants.get_ants_data('r16')).clone('float')
    mi = ants.image_read(ants.get_ants_data('r64')).clone('float')
    mival = ants.image_mutual_information(fi, mi)

    return mival


def applyTransformation(img1, transformation):
    fixed = ants.image_read(ants.get_ants_data('r16'))
    moving = ants.image_read(ants.get_ants_data('r64'))
    fixed = ants.resample_image(fixed, (64, 64), 1, 0)
    moving = ants.resample_image(moving, (64, 64), 1, 0)
    mytx = ants.registration(fixed=fixed, moving=moving,
                                  type_of_transform='SyN')
    mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,
                                               transformlist=mytx['fwdtransforms'])

    return mywarpedimage