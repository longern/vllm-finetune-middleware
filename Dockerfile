FROM huggingface/trl:0.25.0

WORKDIR /

# Copy your handler file
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir runpod pipx /app

# Start the container
CMD ["python3", "-m", "vllm_finetune_middleware.worker"]
