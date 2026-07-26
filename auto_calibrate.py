"""Auto-calibration script: waits for training to finish, then runs threshold calibration."""
import subprocess
import sys
import time
import os
import ctypes

PROJECT = r"C:\Users\Administrator\Subway\Subway_defect_detection_main"
PYTHON = r"C:\ProgramData\miniconda3\envs\Subway\python.exe"
TRAIN_PID = 41932
MODEL = os.path.join(PROJECT, "output", "20260725_144612", "stage_5", "weights", "best.pt")
DATA = os.path.join(PROJECT, "data", "subway_crops", "subway_crops.yaml")
OUTDIR = os.path.join(PROJECT, "output", "20260725_144612", "calibrated_thresholds")


def pid_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if h:
        kernel32.CloseHandle(h)
        return True
    return False


def main():
    print(f"[AutoCalib] Waiting for training PID {TRAIN_PID} to finish...", flush=True)
    while pid_alive(TRAIN_PID):
        time.sleep(60)

    print("[AutoCalib] Training finished. Waiting 10s for file flush...", flush=True)
    time.sleep(10)

    if not os.path.exists(MODEL):
        print(f"[AutoCalib] ERROR: Stage 5 best.pt not found at {MODEL}")
        print("[AutoCalib] Calibration skipped.")
        sys.exit(1)

    print("[AutoCalib] Starting threshold calibration...", flush=True)
    print(f"[AutoCalib] Model:  {MODEL}")
    print(f"[AutoCalib] Data:   {DATA}")
    print(f"[AutoCalib] Output: {OUTDIR}")

    ret = subprocess.run(
        [PYTHON, os.path.join(PROJECT, "scripts", "calibrate_thresholds.py"),
         "--model", MODEL, "--data", DATA, "--output", OUTDIR, "--device", "0"],
        cwd=PROJECT,
    )

    if ret.returncode == 0:
        print("[AutoCalib] Calibration completed successfully.", flush=True)
    else:
        print(f"[AutoCalib] Calibration FAILED with exit code {ret.returncode}")
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
