# vllm-finetune-middleware

Utilities, routers, and middleware that help proxy `/v1/fine_tuning` and file routes to local FastAPI apps while orchestrating RunPod-style fine-tuning jobs backed by an S3-compatible object store or a shared local directory.

## Features
- ASGI/FastAPI apps that mimic the OpenAI `/v1/fine_tuning` and `/v1/files` APIs.
- Middleware (`FineTuningMiddleware`) that intercepts `/v1/fine_tuning/*` requests and creates RunPod jobs to handle fine-tuning requests.
- File router with S3 uploads/downloads, plus local-directory fallback, so training artifacts can be staged for RunPod jobs.

## Installation
```bash
pip install vllm-finetune-middleware
```

## Usage
### 1. Configure environment variables
Set the credentials for your S3-compatible storage and, if needed, the external RunPod endpoint:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=<your-region>
export AWS_S3_ENDPOINT=https://s3.your-provider.com
export AWS_UPLOAD_URL=s3://bucket/upload-prefix
export AWS_ARTIFACTS_URL=s3://bucket/artifacts-prefix
export RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/<your-endpoint-id>/
export RUNPOD_API_KEY=<rpa_your-api-key>
```

If `RUNPOD_ENDPOINT_URL` is unset, fine-tuning job creation, status polling, and cancellation are routed to the internal RunPod-compatible FastAPI app bundled in this package instead of an external RunPod HTTP endpoint.

If you want `/v1/files` to store data locally instead of S3, set `AWS_UPLOAD_URL` to a non-`s3://` path. For example:
```bash
export WORKER_VOLUME_DIR=/runpod-volume/vllm-finetune
export AWS_UPLOAD_URL=files
```
In local mode, relative paths are resolved under `WORKER_VOLUME_DIR`, so the example above writes uploads to `/runpod-volume/vllm-finetune/files/<file id>`. The API process and the worker must share that directory.

If base models are already downloaded on the worker host, set `LOCAL_MODEL_ROOT` to a directory. For a job with model `openai/gpt-oss-120b`, the worker checks `<root>/openai/gpt-oss-120b` first and uses that local path if it exists; otherwise it keeps the original model name and lets the training stack download from the network.
```bash
export WORKER_VOLUME_DIR=/runpod-volume/vllm-finetune
export LOCAL_MODEL_ROOT=path/to/models
```
In this example, `models` is resolved as `/runpod-volume/vllm-finetune/path/to/models`.

### 2. Run the vLLM server with the middleware
```bash
vllm serve qwen/Qwen3-8B --middleware vllm_finetune_middleware.FineTuningMiddleware
```

### 3. Create a RunPod endpoint
Create a RunPod endpoint that uses the script in `vllm_finetune_middleware.worker` and set `WORKER_VOLUME_DIR` to your working directory in a Network Volume (e.g., `/runpod-volume/vllm-finetune`).

### 4. Upload and schedule a job
1. Upload data:
   ```bash
   curl -F "file=@data.jsonl" http://localhost:8000/v1/files
   ```
   The response contains an `id` that becomes the `training_file`.
2. Create a fine-tuning job:
   ```bash
   curl -X POST http://localhost:8000/v1/fine_tuning/jobs \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen/Qwen3-8B","training_file":"<file id>"}'
   ```
3. Poll job status:
   ```bash
   curl http://localhost:8000/v1/fine_tuning/jobs/<job id>
   ```

## Development
Run tests or linting with your preferred tooling. The project follows the standard `src/` layout and exposes the package as `vllm_finetune_middleware`.
