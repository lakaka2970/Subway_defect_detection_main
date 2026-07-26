@echo off
setlocal
set PROJECT=C:\Users\Administrator\Subway\Subway_defect_detection_main
set PYTHON=C:\ProgramData\miniconda3\envs\Subway\python.exe
set TRAIN_PID=41932
set MODEL=%%PROJECT%%\output\20260725_144612\stage_5\weights\best.pt
set DATA=%%PROJECT%%\data\subway_crops\subway_crops.yaml
set OUTDIR=%%PROJECT%%\output\20260725_144612\calibrated_thresholds

echo [AutoCalib] Waiting for training PID %%TRAIN_PID%% to finish...
:wait_loop
tasklist /FI "PID eq %%TRAIN_PID%%" 2>nul | find "%%TRAIN_PID%%" >nul
if %0%==0 (
    timeout /t 60 /nobreak >nul
    goto wait_loop
)

echo [AutoCalib] Training finished at %2026/07/25 ÷‹¡˘% %23:09:54.32%
timeout /t 10 /nobreak >nul

if not exist "%%MODEL%%" (
    echo [AutoCalib] ERROR: Stage 5 best.pt not found at %%MODEL%%
    echo [AutoCalib] Calibration skipped.
    goto :eof
)

echo [AutoCalib] Starting threshold calibration...
echo [AutoCalib] Model: %%MODEL%%
echo [AutoCalib] Data:  %%DATA%%
echo [AutoCalib] Output: %%OUTDIR%%
cd /d %%PROJECT%%
%%PYTHON%% scripts/calibrate_thresholds.py --model "%%MODEL%%" --data "%%DATA%%" --output "%%OUTDIR%%" --device 0

if %0%==0 (
    echo [AutoCalib] Calibration completed successfully at %2026/07/25 ÷‹¡˘% %23:09:54.32%
) else (
    echo [AutoCalib] Calibration FAILED with exit code %0%
)
endlocal
