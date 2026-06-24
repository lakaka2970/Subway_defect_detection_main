"""
Subway Defect Detection — catenary defect detection system.

A two-stage ROI-guided inference pipeline using Ultralytics YOLO with
custom attention modules (EMA, SimAM) for detecting defects on subway
catenary infrastructure from ultra-high-resolution (127 MP) imagery.

Key packages:
    modules      — EMA (Efficient Multi-Scale Attention), SimAM (Simple Attention)
    models       — YOLO11 model YAML configs (s/m scales, P2 variant)
    pipeline     — SmartSlicer, TwoStagePipeline, WBFFusion
    train        — 3-stage training: ROI proposer + defect detector (warmup/full/finetune)
    augmentations — Scene-specific augmentations (tunnel, blur, weather) + CopyPaste
    deployment   — TensorRT export + FastAPI inference server (vehicle/ground modes)
    synthetic    — Inpainting-based synthetic defect generation
"""
