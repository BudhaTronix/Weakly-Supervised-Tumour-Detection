import argparse
import os
from types import SimpleNamespace

from app.config import load_runtime_config

RUNTIME_CONFIG = load_runtime_config()
DEFAULT_MODEL_WEIGHTS = RUNTIME_CONFIG.model_weights_path
DEFAULT_LOG_PATH = RUNTIME_CONFIG.log_path
DEFAULT_CHAOS_DATASET = RUNTIME_CONFIG.chaos_dataset_path
DEFAULT_CLINICAL_DATASET = RUNTIME_CONFIG.clinical_dataset_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production CLI wrapper for semi-supervised tumour detection pipeline."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["chaos-unified", "chaos-sequential", "clinical-sequential", "pretrain-m0"],
        help="Pipeline mode to execute.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Override torch device string.")
    parser.add_argument(
        "--dataset-type",
        choices=["chaos", "clinical"],
        default="chaos",
        help="Dataset type used by pretrain mode.",
    )
    parser.add_argument(
        "--train-pretrain",
        action="store_true",
        help="In pretrain mode, run training instead of loading existing weights.",
    )
    parser.add_argument("--m0-epochs", type=int, default=250, help="Epochs for M0 training.")
    parser.add_argument("--m1-epochs", type=int, default=500, help="Epochs for M1 training.")
    parser.add_argument("--loss-fn", default="Dice", help="Loss function label.")
    parser.add_argument("--model-name", default="Unet", help="Model name label.")
    parser.add_argument("--model-weights-path", default=DEFAULT_MODEL_WEIGHTS, help="Weights directory.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Log directory.")
    parser.add_argument("--chaos-dataset-path", default=DEFAULT_CHAOS_DATASET, help="CHAOS dataset directory.")
    parser.add_argument(
        "--clinical-dataset-path",
        default=DEFAULT_CLINICAL_DATASET,
        help="Clinical dataset directory.",
    )
    return parser


def _load_executor():
    from Code.Semi_supervised import PipelineExecuter as executor

    return executor


def _ensure_dir(path: str) -> None:
    if path:
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot create directory '{path}'. Provide a writable path via CLI or WSTD_* env vars."
            ) from exc


def configure_executor(executor, args) -> None:
    executor.modelWeights_path = args.model_weights_path
    executor.log_path = args.log_path
    executor.chaos_dataset_path = args.chaos_dataset_path
    executor.clinical_dataset_path = args.clinical_dataset_path
    executor.M0_EPOCHS = args.m0_epochs
    executor.M1_EPOCHS = args.m1_epochs
    executor.Loss_fn = args.loss_fn
    executor.Model_name = args.model_name

    _ensure_dir(executor.modelWeights_path)
    _ensure_dir(executor.log_path)


def run(args: argparse.Namespace, executor_module=None):
    executor = executor_module or _load_executor()
    configure_executor(executor, args)

    cuda_as_str = str(args.cuda)
    device = args.device or f"cuda:{args.cuda}"

    if args.mode == "chaos-unified":
        executor.chaos_unified(cuda_as_str, args.seed)
        return SimpleNamespace(mode=args.mode, status="ok")

    if args.mode == "chaos-sequential":
        executor.chaos_sequential(cuda_as_str, args.seed)
        return SimpleNamespace(mode=args.mode, status="ok")

    if args.mode == "clinical-sequential":
        executor.clinical_sequential(cuda_as_str, args.seed)
        return SimpleNamespace(mode=args.mode, status="ok")

    if args.mode == "pretrain-m0":
        is_chaos = args.dataset_type == "chaos"
        model_path, model_bw_path = executor.preTrainM0(
            SEED=args.seed,
            device=device,
            isChaos=is_chaos,
            train=args.train_pretrain,
        )
        return SimpleNamespace(
            mode=args.mode, status="ok", model_path=model_path, model_bw_path=model_bw_path
        )

    raise ValueError(f"Unsupported mode: {args.mode}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    if hasattr(result, "model_path"):
        print(f"model_path={result.model_path}")
        print(f"model_bw_path={result.model_bw_path}")
    print(f"status={result.status}")


if __name__ == "__main__":
    main()
