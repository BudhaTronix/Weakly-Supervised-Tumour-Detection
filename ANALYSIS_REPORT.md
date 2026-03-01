# Codebase Analysis Report

## Scope
- Reviewed all source modules under `Code/` and `Model/`, plus packaging/runtime files.
- Focused on production blockers, reliability, and behavior-preserving improvements.

## Key Issues Found

1. Runtime import blocker:
- `Code/Semi_supervised/Train/Model_M1/M1_main.py` imported `M1_dataloader_clinical`, which is absent in this repository.

2. Supervised runtime bug:
- `Code/Supervised/main.py` passed `defineCriterion` function instead of the instantiated loss object.

3. Unsafe module side effect:
- `Code/Supervised/main.py` executed training at import-time.

4. Fragile CLI entry handling:
- `Code/Semi_supervised/PipelineExecuter.py` used positional args without length checks.

5. Hardcoded absolute paths:
- Dataset/model/log paths were embedded as constants, limiting portability and causing permission failures in non-original environments.

6. Dependency conflict:
- `requirements.txt` pinned `tensorflow==2.15.1` and `tensorboard==2.16.2` (incompatible).

7. Missing production wrappers:
- No stable CLI contract, no lightweight UI, no containerization scaffolding, and no automated test harness in the original repository.

## Improvements Applied (Behavior Preserved)

- Added safe dataloader import fallback for `M1_main.py`.
- Fixed criterion initialization and added `if __name__ == "__main__"` guard in supervised entrypoint.
- Added argument validation to `PipelineExecuter.py`.
- Added environment-variable path configuration (`WSTD_*`) with legacy defaults preserved.
- Added production CLI wrapper (`app/run_pipeline.py`).
- Added minimal Streamlit UI (`app/ui.py`).
- Added tests for CSV generation, runtime config, and CLI dispatch.
- Added Dockerfile and `.dockerignore`.
- Added/updated dependency manifests and `.gitignore`.
- Updated README with production usage documentation.

## Remaining Technical Debt (Not Changed to Avoid Functional Risk)

- Multiple research modules still rely on global side effects and hardcoded proxy settings.
- Legacy training/testing scripts contain duplicated logic and mixed path conventions.
- Several utility scripts appear experimental and are not integrated into automated tests.

