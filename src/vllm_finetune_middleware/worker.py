"""RunPod worker for handling fine-tuning jobs."""

import logging
import os
import shutil
import subprocess
import tempfile

import yaml

WORKER_VOLUME_DIR = os.getenv("WORKER_VOLUME_DIR", os.path.expanduser("~/volume"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_config():
    FINE_TUNING_CONFIG_FILE = os.getenv(
        "FINE_TUNING_CONFIG_FILE",
        os.path.join(WORKER_VOLUME_DIR, "default_config.yaml"),
    )

    config = {}

    if os.path.exists(FINE_TUNING_CONFIG_FILE):
        with open(FINE_TUNING_CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)

    if "methods" not in config:
        config["methods"] = {"supervised": {"command": ["trl", "sft"]}}

    return config


config = get_config()


def get_method_system_config(method_type: str) -> dict:
    if method_type not in config["methods"]:
        raise ValueError(f"Unsupported fine-tuning method: {method_type}")

    method_config = config["methods"][method_type]
    if "command" not in method_config:
        raise ValueError(f"Missing command for method: {method_type}")

    method_command = method_config["command"]
    if type(method_command) is str:
        method_command = [method_command]

    assert isinstance(method_command, list) and all(
        isinstance(cmd, str) for cmd in method_command
    ), "Method command must be a list of strings."

    env = method_config.get("env", {})

    return {"command": method_command, "env": env}


def handler(event):
    job_id = event["id"]
    job_input = event["input"]

    logger.info(f"Processing job {job_id} with input: {job_input}")

    method = job_input.get("method", {})
    method_type = method.get("type", "supervised")
    method_config = method.get(method_type, {})
    hyperparameters = method_config.get("hyperparameters", {})
    method_system_config = get_method_system_config(method_type)

    training_file_id = job_input.get("training_file")
    if not training_file_id:
        raise ValueError("training_file is required in job input")

    training_file_path = os.path.join(WORKER_VOLUME_DIR, "files", training_file_id)
    artifacts_dir = os.path.join(WORKER_VOLUME_DIR, "artifacts", job_id)

    extra_args = []

    if "n_epochs" in hyperparameters:
        extra_args.extend(["--num_train_epochs", str(hyperparameters["n_epochs"])])

    with tempfile.TemporaryDirectory() as tempdir:
        os.mkdir(os.path.join(tempdir, "data"))
        shutil.copy(
            training_file_path,
            os.path.join(tempdir, "data", "train.jsonl"),
        )

        run_args = [
            *method_system_config["command"],
            "--model_name_or_path",
            job_input["model"],
            "--dataset_name",
            tempdir,
            "--output_dir",
            os.path.join(artifacts_dir, "model"),
            "--logging_dir",
            os.path.join(artifacts_dir, "logs"),
            "--use_peft",
            "--save_only_model",
            "--save_strategy",
            "no",
            *extra_args,
        ]
        process = subprocess.run(
            run_args,
            env={**os.environ, **method_system_config["env"]},
            capture_output=True,
            text=True,
        )

    if process.returncode != 0:
        logger.error("%s", process.stderr)
        raise RuntimeError(f"Fine-tuning job {job_id} failed.\n{process.stderr}")

    return {"id": job_id, "status": "succeeded"}


# Start the Serverless function when the script is run
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
