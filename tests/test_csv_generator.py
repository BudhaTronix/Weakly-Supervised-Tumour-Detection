import csv
from pathlib import Path

from Code.Utils.CSVGenerator import GenerateCSV_Student


def test_generate_csv_student_pairs_matching_filenames(tmp_path):
    dataset_root = tmp_path / "dataset"
    mri_dir = dataset_root / "mri"
    gt_dir = dataset_root / "ct_mri_reg_gt"
    mri_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    (mri_dir / "case_001.nii.gz").write_text("x", encoding="utf-8")
    (mri_dir / "case_002.nii.gz").write_text("x", encoding="utf-8")
    (gt_dir / "case_001.nii.gz").write_text("y", encoding="utf-8")
    (gt_dir / "case_100.nii.gz").write_text("y", encoding="utf-8")

    csv_name = "dataset.csv"
    GenerateCSV_Student(dataset_Path=str(dataset_root) + "/", csv_FileName=csv_name)

    csv_path = dataset_root / csv_name
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows == [["case_001.nii.gz", "case_001.nii.gz"]]


def test_generate_csv_student_creates_file(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "mri").mkdir(parents=True)
    (dataset_root / "ct_mri_reg_gt").mkdir(parents=True)

    csv_name = "dataset.csv"
    GenerateCSV_Student(dataset_Path=str(dataset_root) + "/", csv_FileName=csv_name)

    assert Path(dataset_root / csv_name).is_file()

