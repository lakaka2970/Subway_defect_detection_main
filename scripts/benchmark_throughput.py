#!/usr/bin/env python3
"""Benchmark training throughput to find optimal batch size for maximum samples/sec."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subway_yolo.utils.torch_utils import autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = "cuda:0"
IMGSZ = 1024
N_WARMUP = 5
N_ITERS = 20


def benchmark_batch(bs: int):
    """Run training step benchmark at given batch size. Returns (ms/iter, samples/s, VRAM GB)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ckpt = torch.load("yolo_weights/yolo11s.pt", map_location=DEVICE, weights_only=False)
    model = ckpt["model"].float().train()
    for p in model.parameters():
        p.requires_grad = True
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, fl_gamma=2.0, imgsz=IMGSZ,
        class_weights=None, overlap_mask=False,
    )

    imgs = torch.randn(bs, 3, IMGSZ, IMGSZ, device=DEVICE)
    batch = {
        "img": imgs,
        "cls": torch.zeros(bs, 1, device=DEVICE, dtype=torch.long),
        "bboxes": torch.rand(bs, 4, device=DEVICE) * 0.5 + 0.1,
        "batch_idx": torch.arange(bs, device=DEVICE, dtype=torch.long),
    }

    scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937)

    # Warmup (includes cudnn.benchmark kernel search)
    for _ in range(N_WARMUP):
        opt.zero_grad()
        with autocast(enabled=True):
            loss = model(batch)[0].sum()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        opt.zero_grad()
        with autocast(enabled=True):
            loss = model(batch)[0].sum()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / N_ITERS

    ms_per_iter = elapsed * 1000
    samples_per_sec = bs / elapsed
    vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    del model, opt, scaler, imgs, batch
    torch.cuda.empty_cache()
    return ms_per_iter, samples_per_sec, vram_gb


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: imgsz={IMGSZ}, AMP=True, TF32=True, cudnn.benchmark=True")
    print(f"Warmup: {N_WARMUP} iters, Measured: {N_ITERS} iters")
    print()
    print(f"{'batch':>6} {'ms/iter':>9} {'samples/s':>10} {'VRAM(GB)':>9} {'iters/epoch':>12}")
    print("-" * 52)

    n_train = 655  # training images
    results = []

    for bs in [16, 24, 32, 48, 64]:
        try:
            ms, sps, vram = benchmark_batch(bs)
            iters_per_epoch = (n_train + bs - 1) // bs
            print(f"{bs:>6} {ms:>9.1f} {sps:>10.1f} {vram:>9.2f} {iters_per_epoch:>12}")
            results.append((bs, ms, sps, vram, iters_per_epoch))
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6} {'OOM':>9}")
            torch.cuda.empty_cache()
            break

    print()
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"★ 最优吞吐: batch={best[0]}, {best[2]:.0f} samples/s, "
              f"{best[1]:.0f} ms/iter, {best[3]:.1f} GB VRAM")
        print(f"  每epoch {best[4]} iterations, 预计 {best[4] * best[1] / 1000:.1f}s/epoch")

        # Also show epoch time comparison
        print(f"\n  对比 (655张训练图, 50 epochs):")
        for bs, ms, sps, vram, iters in results:
            epoch_time = iters * ms / 1000
            total_time = epoch_time * 50
            print(f"    batch={bs:>2}: {epoch_time:.1f}s/epoch, C1总计 {total_time/60:.1f}min")


if __name__ == "__main__":
    main()
