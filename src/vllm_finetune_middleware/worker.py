"""RunPod worker for handling fine-tuning jobs."""

import logging
import os
import re
import shutil
import subprocess
import tempfile

import yaml

WORKER_VOLUME_DIR = os.getenv("WORKER_VOLUME_DIR", os.path.expanduser("~/volume"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_uri(path: str) -> bool:
    return re.match(r"^[A-Za-z][0-9A-Za-z+.-]*://", path) is not None


def get_config(s3=None):
    FINE_TUNING_CONFIG_FILE = os.getenv(
        "FINE_TUNING_CONFIG_FILE",
        os.path.join(WORKER_VOLUME_DIR, "default_config.yaml"),
    )

    config = {}

    if is_uri(FINE_TUNING_CONFIG_FILE):
        import fsspec

        with fsspec.open(FINE_TUNING_CONFIG_FILE, "r", s3=s3) as f:
            config = yaml.safe_load(f)
    else:
        if os.path.exists(FINE_TUNING_CONFIG_FILE):
            with open(FINE_TUNING_CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)

    if "methods" not in config:
        config["methods"] = {"supervised": {"command": ["trl", "sft"]}}

    return config


def get_method_system_config(method_type: str, s3=None) -> dict:
    config = get_config(s3=s3)

    if method_type not in config["methods"]:
        raise ValueError(f"Unsupported fine-tuning method: {method_type}")

    method_config = config["methods"][method_type]
    if "command" not in method_config:
        raise ValueError(f"Missing command for method: {method_type}")

    method_command = method_config["command"]
    if isinstance(method_command, str):
        method_command = [method_command]

    assert isinstance(method_command, list) and all(
        isinstance(cmd, str) for cmd in method_command
    ), "Method command must be a list of strings."

    env = method_config.get("env", {})

    return {"command": method_command, "env": env}


def fsspec_move_dir(src: str, dst: str, s3=None):
    import fsspec

    fs_src, path_src = fsspec.core.url_to_fs(src, s3=s3)
    fs_dst, path_dst = fsspec.core.url_to_fs(dst, s3=s3)

    for root, _, files in fs_src.walk(path_src):
        for filename in files:
            relative_basedir = os.path.relpath(root, path_src)
            dst_basedir = os.path.join(path_dst, relative_basedir)

            if not fs_dst.exists(dst_basedir):
                fs_dst.makedirs(dst_basedir, exist_ok=True)

            src_file_path = os.path.join(root, filename)
            dst_file_path = os.path.join(dst_basedir, filename)

            with fs_src.open(src_file_path, "rb") as src_file, fs_dst.open(
                dst_file_path, "wb"
            ) as dst_file:
                shutil.copyfileobj(src_file, dst_file)


def handler(event):
    job_id = event["id"]
    job_input = event["input"]

    s3_config = event.get("s3Config")
    s3_init = (
        None
        if s3_config is None
        else {
            "key": s3_config["accessId"],
            "secret": s3_config["accessSecret"],
            "client_kwargs": {"endpoint_url": s3_config["endpointUrl"]},
        }
    )

    logger.info(f"Processing job {job_id} with input: {job_input}")

    method = job_input.get("method", {})
    method_type = method.get("type", "supervised")
    method_config = method.get(method_type, {})
    hyperparameters = method_config.get("hyperparameters", {})
    method_system_config = get_method_system_config(method_type, s3=s3_init)

    training_file_id = job_input.get("training_file")
    if not training_file_id:
        raise ValueError("training_file is required in job input")

    training_file_path = os.path.join(WORKER_VOLUME_DIR, "files", training_file_id)
    artifacts_dir = os.path.join(WORKER_VOLUME_DIR, "artifacts", job_id)

    extra_args = []

    if "n_epochs" in hyperparameters:
        extra_args.extend(["--num_train_epochs", str(hyperparameters["n_epochs"])])

    integrations = method_config.get("integrations", [])
    for integration in integrations:
        if integration["type"] == "tensorboard":
            extra_args.extend(["--logging_dir", os.path.join(artifacts_dir, "logs")])

    with tempfile.TemporaryDirectory() as tempdir:
        os.makedirs(os.path.join(tempdir, "dataset", "data"), exist_ok=True)

        if is_uri(training_file_path):
            import fsspec

            with fsspec.open(training_file_path, "r", s3=s3_init) as src, open(
                os.path.join(tempdir, "dataset", "data", "train.jsonl"), "w"
            ) as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(
                training_file_path,
                os.path.join(tempdir, "dataset", "data", "train.jsonl"),
            )

        if is_uri(artifacts_dir):
            model_output_dir = os.path.join(tempdir, "model")
            deferred_upload = lambda: fsspec_move_dir(
                model_output_dir, artifacts_dir, s3=s3_init
            )
        else:
            model_output_dir = os.path.join(artifacts_dir, "model")
            deferred_upload = lambda: None

        run_args = [
            *method_system_config["command"],
            "--model_name_or_path",
            job_input["model"],
            "--dataset_name",
            os.path.join(tempdir, "dataset"),
            "--output_dir",
            model_output_dir,
            "--use_peft",
            "--save_only_model",
            "--save_strategy",
            "no",
            *extra_args,
        ]
        process = subprocess.run(
            run_args,
            env=os.environ | method_system_config["env"],
            capture_output=True,
            text=True,
        )

        deferred_upload()

    if process.returncode != 0:
        logger.error("%s", process.stderr)
        raise RuntimeError(f"Fine-tuning job {job_id} failed.\n{process.stderr}")

    return {"id": job_id, "status": "succeeded"}


# Start the Serverless function when the script is run
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
