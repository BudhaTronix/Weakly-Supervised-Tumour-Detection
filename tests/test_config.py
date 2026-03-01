from app import config


def test_runtime_config_defaults(monkeypatch):
    monkeypatch.delenv("WSTD_MODEL_WEIGHTS_PATH", raising=False)
    monkeypatch.delenv("WSTD_LOG_PATH", raising=False)
    monkeypatch.delenv("WSTD_CHAOS_DATASET_PATH", raising=False)
    monkeypatch.delenv("WSTD_CLINICAL_DATASET_PATH", raising=False)

    cfg = config.load_runtime_config()

    assert cfg.model_weights_path.endswith("/Code/Semi_supervised/model_weights/")
    assert cfg.log_path.endswith("/Code/Semi_supervised/Logs/runs/")
    assert cfg.chaos_dataset_path.endswith("/Dataset/chaos_3D/")
    assert cfg.clinical_dataset_path.endswith("/Dataset/Clinical/")


def test_runtime_config_env_override(monkeypatch):
    monkeypatch.setenv("WSTD_MODEL_WEIGHTS_PATH", "/tmp/weights/")
    monkeypatch.setenv("WSTD_LOG_PATH", "/tmp/logs/")
    monkeypatch.setenv("WSTD_CHAOS_DATASET_PATH", "/tmp/chaos/")
    monkeypatch.setenv("WSTD_CLINICAL_DATASET_PATH", "/tmp/clinical/")

    cfg = config.load_runtime_config()

    assert cfg.model_weights_path == "/tmp/weights/"
    assert cfg.log_path == "/tmp/logs/"
    assert cfg.chaos_dataset_path == "/tmp/chaos/"
    assert cfg.clinical_dataset_path == "/tmp/clinical/"

