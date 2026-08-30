import torch
print("torch", torch.__version__, "cuda_runtime", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
a = torch.randn(3000, 3000, device="cuda")
b = a @ a
torch.cuda.synchronize()
print("GPU matmul OK, checksum = %.3f" % float(b.sum()))
print("VRAM alloc %.2f GB / total %.2f GB" %
      (torch.cuda.memory_allocated() / 1e9, torch.cuda.get_device_properties(0).total_memory / 1e9))
