# vllm-finetune-middleware

Utilities, routers, and middleware that help proxy `/v1/fine_tuning` and file routes to local FastAPI apps while orchestrating RunPod-style fine-tuning jobs backed by an S3-compatible object store.

## Features
- ASGI/FastAPI apps that mimic the OpenAI `/v1/fine_tuning` and `/v1/files` APIs.
- Middleware (`FineTuningMiddleware`) that intercepts `/v1/fine_tuning/*` requests and creates RunPod jobs to handle fine-tuning requests.
- File router with S3 uploads/downloads so training artifacts can be staged for RunPod jobs.

## Installation
```bash
pip install vllm-finetune-middleware
```

## Usage
### 1. Configure environment variables
Set the credentials for your S3-compatible storage and RunPod endpoint:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_S3_ENDPOINT_URL=https://s3.your-provider.com
export AWS_UPLOAD_URL=s3://bucket/upload-prefix
export AWS_ARTIFACTS_URL=s3://bucket/artifacts-prefix
export RUNPOD_ENDPOINT_URL=https://your-runpod-endpoint/runpod
```

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
