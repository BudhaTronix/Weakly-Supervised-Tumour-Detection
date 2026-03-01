import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    model_weights_path: str
    log_path: str
    chaos_dataset_path: str
    clinical_dataset_path: str


def load_runtime_config() -> RuntimeConfig:
    """
    Load runtime paths from environment variables with backward-compatible defaults.
    """
    return RuntimeConfig(
        model_weights_path=os.getenv(
            "WSTD_MODEL_WEIGHTS_PATH",
            "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/",
        ),
        log_path=os.getenv(
            "WSTD_LOG_PATH",
            "/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/Logs/runs/",
        ),
        chaos_dataset_path=os.getenv(
            "WSTD_CHAOS_DATASET_PATH",
            "/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/",
        ),
        clinical_dataset_path=os.getenv(
            "WSTD_CLINICAL_DATASET_PATH",
            "/project/mukhopad/tmp/LiverTumorSeg/Dataset/Clinical/",
        ),
    )

