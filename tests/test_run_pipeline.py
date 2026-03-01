from argparse import Namespace

from app.run_pipeline import configure_executor, run


class DummyExecutor:
    def __init__(self):
        self.calls = []

    def chaos_unified(self, cuda, seed):
        self.calls.append(("chaos_unified", cuda, seed))

    def chaos_sequential(self, cuda, seed):
        self.calls.append(("chaos_sequential", cuda, seed))

    def clinical_sequential(self, cuda, seed):
        self.calls.append(("clinical_sequential", cuda, seed))

    def preTrainM0(self, SEED, device, isChaos, train):
        self.calls.append(("preTrainM0", SEED, device, isChaos, train))
        return "m0.pth", "m0_bw.pth"


def build_args(tmp_path, mode):
    return Namespace(
        mode=mode,
        cuda=2,
        seed=99,
        device=None,
        dataset_type="clinical",
        train_pretrain=True,
        m0_epochs=10,
        m1_epochs=20,
        loss_fn="Dice",
        model_name="Unet",
        model_weights_path=str(tmp_path / "weights"),
        log_path=str(tmp_path / "logs"),
        chaos_dataset_path=str(tmp_path / "chaos"),
        clinical_dataset_path=str(tmp_path / "clinical"),
    )


def test_configure_executor_sets_runtime_values(tmp_path):
    executor = DummyExecutor()
    args = build_args(tmp_path, mode="chaos-unified")

    configure_executor(executor, args)

    assert executor.modelWeights_path == args.model_weights_path
    assert executor.log_path == args.log_path
    assert executor.chaos_dataset_path == args.chaos_dataset_path
    assert executor.clinical_dataset_path == args.clinical_dataset_path
    assert executor.M0_EPOCHS == 10
    assert executor.M1_EPOCHS == 20
    assert executor.Loss_fn == "Dice"
    assert executor.Model_name == "Unet"


def test_run_dispatches_chaos_unified(tmp_path):
    executor = DummyExecutor()
    args = build_args(tmp_path, mode="chaos-unified")

    result = run(args, executor_module=executor)

    assert result.status == "ok"
    assert executor.calls == [("chaos_unified", "2", 99)]


def test_run_dispatches_pretrain_mode(tmp_path):
    executor = DummyExecutor()
    args = build_args(tmp_path, mode="pretrain-m0")

    result = run(args, executor_module=executor)

    assert result.status == "ok"
    assert result.model_path == "m0.pth"
    assert result.model_bw_path == "m0_bw.pth"
    assert executor.calls == [("preTrainM0", 99, "cuda:2", False, True)]

