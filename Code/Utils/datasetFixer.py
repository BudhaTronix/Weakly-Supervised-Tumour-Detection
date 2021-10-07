from pathlib import Path
import shutil


def checkFile(imgPath_fp, gtPath_fp, subject_id, op_Path):
    imgPath = Path(imgPath_fp)
    gtPath = Path(gtPath_fp)
    img = gt = []
    for file_name in sorted(imgPath.glob("*")):
        img.append(file_name.name.replace(".dcm", ""))
    for file_name in sorted(gtPath.glob("*")):
        gt.append(file_name.name.replace(".png", ""))
    img.sort()
    gt.sort()
    c = 0
    for img_file_name in sorted(imgPath.glob("*")):
        for gt_file_name in sorted(gtPath.glob("*")):
            if img_file_name.name.replace(".dcm", "") == gt_file_name.name.replace(".png", ""):
                old_name = img_file_name
                new_name = op_Path + "images/" + str(subject_id) + "_" + str(c) + ".dcm"
                # os.rename(old_name, new_name)  # Rename the image file
                shutil.copyfile(old_name, new_name)

                old_name = gt_file_name
                new_name = op_Path + "gt/" + str(subject_id) + "_" + str(c) + ".png"
                # os.rename(old_name, new_name)  # Rename the GT file
                shutil.copyfile(old_name, new_name)
                c += 1


def checkFile_CT(imgPath_fp, subject_id, op_Path):
    imgPath = Path(imgPath_fp)
    img = []
    for file_name in sorted(imgPath.glob("*")):
        img.append(file_name.name.replace(".dcm", ""))
    img.sort()
    c = 0
    for img_file_name in sorted(imgPath.glob("*")):
        new_name = op_Path + "images/" + str(subject_id) + "_" + str(c) + ".dcm"
        old_name = img_file_name
        shutil.copyfile(old_name, new_name)
        c += 1


checkFile('C:/Users/budha/Desktop/CHAOS_Train_Sets/Train_Sets/MR/3/T2SPIR/DICOM_anon',
          "C:/Users/budha/Desktop/CHAOS_Train_Sets/Train_Sets/MR/3/T2SPIR/Ground", 3,
          "C:/Users/budha/Desktop/New folder/")
