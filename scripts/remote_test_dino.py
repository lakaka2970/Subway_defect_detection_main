import os, time, sys
# HF 镜像（国内 huggingface.co 不通）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
import timm

t = time.time()
m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                      num_classes=0, img_size=518)
print("load ok %.1fs  params=%.1fM" % (time.time() - t,
                                       sum(p.numel() for p in m.parameters()) / 1e6),
      flush=True)
m = m.cuda().half().eval()
x = torch.randn(1, 3, 518, 518).cuda().half()
with torch.no_grad():
    y = m.forward_features(x)
print("forward out", tuple(y.shape), flush=True)

# 吞吐测试
torch.cuda.reset_peak_memory_stats()
t = time.time()
N = 32
with torch.no_grad():
    for _ in range(N):
        m.forward_features(x)
dt = time.time() - t
print("throughput: %.1f img/s (518x518 fp16), VRAM peak %.2f GB"
      % (N / dt, torch.cuda.max_memory_allocated() / 1e9), flush=True)
print("DINO_TEST_OK")
