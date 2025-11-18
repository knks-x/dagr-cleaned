import time
import subprocess
import pynvml

# The full command to run
COMMAND = [
    "python", "scripts/run_test_interframe_det.py",
    "--config", "config/dagr-s-dsec.yaml",
    "--use_image",
    "--img_net", "resnet50",
    "--exp_name", "test_interframe_front_det",
    "--checkpoint", "logs/dsec/detection/train_dsec_front_vis/best_model_mAP_0.3475849682030187.pth",
    "--batch_size", "1",
    "--dataset_directory", "$DSEC_ROOT",
    "--no_eval",
    "--output_directory", "$LOG_DIR",
    "--num_interframe_steps", "10"
]

GPU_INDEX = 0
MEMORY_THRESHOLD_MB = 1000
CHECK_INTERVAL = 60*30

def gpu_is_idle(gpu_index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mb = mem_info.used / (1024 * 1024)

    compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)

    return used_mb <= MEMORY_THRESHOLD_MB and not compute_procs and not graphics_procs

def main():
    pynvml.nvmlInit()
    print(f"Monitoring GPU {GPU_INDEX} until it becomes idle...")

    try:
        while True:
            if gpu_is_idle(GPU_INDEX):
                print("GPU is idle! Starting the job...")
                subprocess.Popen(" ".join(COMMAND), shell=True, executable="/bin/bash")
                break
            else:
                print(f"GPU busy. Checking again in {CHECK_INTERVAL} sec...")
                time.sleep(CHECK_INTERVAL)
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
