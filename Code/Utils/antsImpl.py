import ants
from pathlib import Path


def getWarp(img1, img2):
    fi = ants.image_read(ants.get_ants_data('r16'))  # Sample Input
    mi = ants.image_read(ants.get_ants_data('r64'))  # Sample Input
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


"""
dataset_path = "/project/cmandal/liver_seg/datasets/chaos/"
imgPath = Path(str(dataset_path) + "/images")
gtPath = Path(str(dataset_path) + "/gt")
for img_file_name in sorted(imgPath.glob("*")):
    for gt_file_name in sorted(gtPath.glob("*")):
        if img_file_name.name.replace(".dcm", "") == gt_file_name.name.replace(".png", ""):
            print(img_file_name.name, gt_file_name.name)
"""

getWarp("", "")
