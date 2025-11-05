"""RunPod worker for handling fine-tuning jobs."""


def handler(event):
    job_id = event["id"]
    job_input = event["input"]

    training_file = job_input.get("training_file")

    return {"id": job_id, "status": "succeeded", "training_file": training_file}


# Start the Serverless function when the script is run
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
