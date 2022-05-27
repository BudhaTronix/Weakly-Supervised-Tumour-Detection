import csv
import os
import logging
from pathlib import Path

from tqdm import tqdm


def checkCSV(dataset_Path, csv_FileName, overwrite=False):
    if overwrite:
        if os.path.isfile(dataset_Path + csv_FileName):
            os.remove(dataset_Path + csv_FileName)
    if not os.path.isfile(dataset_Path + csv_FileName):
        logging.info(" CSV File missing..Generating new CVS File..")
        GenerateCSV(dataset_Path=dataset_Path, csv_FileName=csv_FileName)
        logging.info(" CSV File Created!")
    else:
        logging.info("\n Dataset file available")


def GenerateCSV(dataset_Path, csv_FileName):
    with open(dataset_Path + csv_FileName, 'w') as f:
        writer = csv.writer(f)
        dataset_Path = Path(dataset_Path)
        imgPath = Path(str(dataset_Path) + "/mri")
        gtPath = Path(str(dataset_Path) + "/mri_gt")
        for img_file_name in tqdm(sorted(imgPath.glob("*"))):
            for gt_file_name in sorted(gtPath.glob("*")):
                if img_file_name.name == gt_file_name.name:
                    writer.writerow([img_file_name.name, gt_file_name.name])


def checkCSV_Student(dataset_Path, csv_FileName, overwrite=False):
    if overwrite:
        if os.path.isfile(dataset_Path + csv_FileName):
            os.remove(dataset_Path + csv_FileName)
    if not os.path.isfile(dataset_Path + csv_FileName):
        logging.info(" CSV File missing..Generating new CVS File..")
        GenerateCSV_Student(dataset_Path=dataset_Path, csv_FileName=csv_FileName)
        logging.info(" CSV File Created!")
    else:
        logging.info("\n Dataset file available")


def GenerateCSV_Student(dataset_Path, csv_FileName):
    with open(dataset_Path + csv_FileName, 'w') as f:
        writer = csv.writer(f)
        dataset_Path = Path(dataset_Path)
        imgPath = Path(str(dataset_Path) + "/mri")
        gtPath = Path(str(dataset_Path) + "/ct")
        for img_file_name in tqdm(sorted(imgPath.glob("*"))):
            for gt_file_name in sorted(gtPath.glob("*")):
                if img_file_name.name == gt_file_name.name:
                    writer.writerow([img_file_name.name, gt_file_name.name])
