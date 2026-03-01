# Weakly-Supervised Tumour Detection

This repository contains supervised and semi-supervised pipelines for 3D liver tumour detection/segmentation from CT and MRI data.

## Project Overview

The original research code has been productionized while preserving core behavior and outputs. Model logic was not rewritten; improvements focus on reliability, portability, packaging, and operability.

## Features

- Semi-supervised training/evaluation pipeline for tumour detection.
- Production CLI wrapper for reproducible execution.
- Minimal Streamlit UI for interactive usage.
- Dockerized runtime.
- Unit tests for utility and runtime wrapper behavior.
- Environment-variable based path configuration.

## Installation

Recommended: Python 3.10 with Conda.

```bash
conda create -n ws_tumour python=3.10 pip -y
conda activate ws_tumour
pip install -r requirements.txt
```

For development/testing:

```bash
pip install -r requirements-dev.txt
```

## Running Locally

Run through the production CLI wrapper:

```bash
python -m app.run_pipeline --mode chaos-unified --cuda 0 --seed 42
```

Supported modes:

- `chaos-unified`
- `chaos-sequential`
- `clinical-sequential`
- `pretrain-m0`

Example with explicit paths:

```bash
python -m app.run_pipeline \
  --mode pretrain-m0 \
  --dataset-type clinical \
  --train-pretrain \
  --cuda 0 \
  --seed 42 \
  --model-weights-path /abs/path/model_weights/ \
  --log-path /abs/path/logs/ \
  --chaos-dataset-path /abs/path/chaos_3D/ \
  --clinical-dataset-path /abs/path/clinical/
```

Optional environment variables:

- `WSTD_MODEL_WEIGHTS_PATH`
- `WSTD_LOG_PATH`
- `WSTD_CHAOS_DATASET_PATH`
- `WSTD_CLINICAL_DATASET_PATH`

## Running the UI

```bash
streamlit run app/ui.py
```

Then open `http://localhost:8501`.

## Running With Docker

Build:

```bash
docker build -t weakly-supervised-tumour-detection:latest .
```

Run:

```bash
docker run --rm -p 8501:8501 weakly-supervised-tumour-detection:latest
```

## Testing

```bash
pytest -q
```

Current test coverage includes:

- CSV generation behavior.
- Runtime configuration loading.
- CLI dispatch/config behavior.

## Project Structure

```text
project/
├── Code/                     # Original training/testing code
├── Model/                    # Model definitions
├── app/
│   ├── config.py             # Runtime config (env vars)
│   ├── run_pipeline.py       # Production CLI wrapper
│   └── ui.py                 # Streamlit UI
├── tests/
├── ANALYSIS_REPORT.md        # Audit findings + fixes
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
```

## Notes

- Core model functionality and expected outputs were preserved.
- Legacy scripts in `Code/` remain available for backward compatibility.
- See `ANALYSIS_REPORT.md` for the full analysis and issue list.
