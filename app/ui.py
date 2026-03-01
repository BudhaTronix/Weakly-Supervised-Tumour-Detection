import shlex
import subprocess
import sys

import streamlit as st


st.set_page_config(page_title="Tumour Detection Runner", layout="centered")
st.title("Weakly-Supervised Tumour Detection")
st.caption("Minimal UI to run existing pipeline modes through the production CLI wrapper.")

mode = st.selectbox(
    "Mode",
    options=["chaos-unified", "chaos-sequential", "clinical-sequential", "pretrain-m0"],
    index=0,
)
seed = st.number_input("Seed", min_value=0, value=42, step=1)
cuda = st.number_input("CUDA device id", min_value=0, value=0, step=1)
loss_fn = st.text_input("Loss function", value="Dice")
model_name = st.text_input("Model name", value="Unet")

dataset_type = "chaos"
train_pretrain = False
if mode == "pretrain-m0":
    dataset_type = st.selectbox("Dataset type", options=["chaos", "clinical"], index=0)
    train_pretrain = st.checkbox("Run pretrain training", value=False)

model_weights_path = st.text_input(
    "Model weights path",
    value="/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/model_weights/",
)
log_path = st.text_input(
    "Log path",
    value="/project/mukhopad/tmp/LiverTumorSeg/Code/Semi_supervised/Logs/runs/",
)
chaos_dataset_path = st.text_input(
    "CHAOS dataset path",
    value="/project/mukhopad/tmp/LiverTumorSeg/Dataset/chaos_3D/",
)
clinical_dataset_path = st.text_input(
    "Clinical dataset path",
    value="/project/mukhopad/tmp/LiverTumorSeg/Dataset/Clinical/",
)

if st.button("Run", type="primary"):
    cmd = [
        sys.executable,
        "-m",
        "app.run_pipeline",
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--cuda",
        str(cuda),
        "--loss-fn",
        loss_fn,
        "--model-name",
        model_name,
        "--model-weights-path",
        model_weights_path,
        "--log-path",
        log_path,
        "--chaos-dataset-path",
        chaos_dataset_path,
        "--clinical-dataset-path",
        clinical_dataset_path,
    ]

    if mode == "pretrain-m0":
        cmd.extend(["--dataset-type", dataset_type])
        if train_pretrain:
            cmd.append("--train-pretrain")

    st.code(" ".join(shlex.quote(part) for part in cmd), language="bash")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    st.subheader("stdout")
    st.text(proc.stdout if proc.stdout else "(empty)")
    st.subheader("stderr")
    st.text(proc.stderr if proc.stderr else "(empty)")
    st.subheader("exit code")
    st.write(proc.returncode)

