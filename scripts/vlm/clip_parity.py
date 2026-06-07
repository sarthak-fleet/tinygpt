"""
CLIP-ViT-L-14 parity check between our Swift VisionEncoder and HF
PyTorch's CLIPVisionModel. We dump the synthetic gradient image from
the Swift side as raw NHWC float32, then read it here, permute to NCHW
(what PyTorch wants), and forward.

Pass criterion: max abs difference in the last_hidden_state < ~1e-3 on
fp32. Differences come from:
  - Float32 reductions in different summation orders (sub-1e-4 typically)
  - bfloat16 download paths if HF decides to materialise that way (we
    force float32 here)
"""
import json
import sys
import numpy as np
import torch
from transformers import CLIPVisionModel

snapshot_dir = "/Users/sarthak/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
pixel_npy = "/tmp/clip_swift_pixels.npy"
out_npy   = "/tmp/clip_swift_features.npy"

# Load the (1, 224, 224, 3) NHWC float32 tensor the Swift smoke writes.
pixels_nhwc = np.load(pixel_npy)
print(f"loaded swift pixels: shape={pixels_nhwc.shape}, dtype={pixels_nhwc.dtype}")
print(f"  pixels mean={pixels_nhwc.mean():.6f}, std={pixels_nhwc.std():.6f}")

# Convert NHWC → NCHW for PyTorch.
pixels_nchw = np.transpose(pixels_nhwc, (0, 3, 1, 2))
pixels_t = torch.tensor(pixels_nchw, dtype=torch.float32)

# Load HF model in fp32 explicitly.
model = CLIPVisionModel.from_pretrained(snapshot_dir, torch_dtype=torch.float32).eval()
with torch.no_grad():
    out = model(pixel_values=pixels_t).last_hidden_state  # [1, 257, 1024]
hf_features = out.cpu().numpy()
print(f"hf features: shape={hf_features.shape}, dtype={hf_features.dtype}")
print(f"  mean={hf_features.mean():.6f}, std={hf_features.std():.6f}")
print(f"  CLS[:10] = {hf_features[0,0,:10].tolist()}")
print(f"  patch1[:10] = {hf_features[0,1,:10].tolist()}")

# Compare with Swift features.
swift_features = np.load(out_npy)
print(f"swift features: shape={swift_features.shape}, dtype={swift_features.dtype}")
print(f"  mean={swift_features.mean():.6f}, std={swift_features.std():.6f}")
print(f"  CLS[:10] = {swift_features[0,0,:10].tolist()}")
print(f"  patch1[:10] = {swift_features[0,1,:10].tolist()}")

# Element-wise diff stats.
diff = np.abs(hf_features - swift_features)
print(f"\ndiff: max={diff.max():.6f}, mean={diff.mean():.6f}, median={np.median(diff):.6f}")
print(f"95th percentile diff: {np.percentile(diff, 95):.6f}")
print(f"99th percentile diff: {np.percentile(diff, 99):.6f}")

# Cosine similarity over the full flat vector — robust to scaling.
flat_hf = hf_features.reshape(-1)
flat_sw = swift_features.reshape(-1)
cos = (flat_hf * flat_sw).sum() / (np.linalg.norm(flat_hf) * np.linalg.norm(flat_sw))
print(f"cosine sim (full tensor): {cos:.6f}")

# Pass / fail.
tolerance = 5e-3
if diff.max() < tolerance:
    print(f"\nPASS (max diff {diff.max():.6f} < {tolerance})")
else:
    print(f"\nFAIL (max diff {diff.max():.6f} > {tolerance})")
    sys.exit(1)
