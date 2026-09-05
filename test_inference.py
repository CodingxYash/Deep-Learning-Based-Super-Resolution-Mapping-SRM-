"""
Test the trained model: run inference on one (or a few) held-out samples,
compute PSNR/SSIM, and save a side-by-side comparison image.

Usage:
    python test_inference.py --sample_idx 0
    python test_inference.py --sample_idx 5 10 15   (test multiple samples)
"""

import argparse
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from dataset import SRDataset
from model import Sen2SRInspired


def tensor_to_image(t):
    """Convert a (C,H,W) float tensor in [0,1] to a uint8 numpy image. Used for METRICS only."""
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def brighten_for_display(t, gamma=0.85):
    """
    Convert a (C,H,W) float tensor to a uint8 image with gentle percentile stretch + 
    mild gamma correction — natural looking brightness, not oversaturated.
    """
    arr = t.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    out = np.zeros_like(arr)

    for c in range(arr.shape[2]):
        channel = arr[:, :, c]
        low, high = np.percentile(channel, (1, 99))  # thoda kam aggressive stretch
        channel = np.clip((channel - low) / (high - low + 1e-8), 0, 1)
        out[:, :, c] = channel

    out = np.power(out, gamma)  # 0.85 = mild brightening, not overdone
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load your dataset the same way train.py does
    ds = SRDataset("train_lr/train_lr", "train_hr_all")
    print("Total samples available:", len(ds))

    # Load the trained model
    model = Sen2SRInspired(scale=3).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    for idx in args.sample_idx:
        if idx >= len(ds):
            print(f"Skipping index {idx} — out of range (max {len(ds)-1})")
            continue

        lr_img, hr_img = ds[idx]
        lr_img_batch = lr_img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(lr_img_batch)[0]  # remove batch dimension

        # ---- Metrics: computed on RAW values, unchanged ----
        pred_np = tensor_to_image(pred).astype(np.float32) / 255.0
        hr_np = tensor_to_image(hr_img).astype(np.float32) / 255.0

        psnr_val = psnr_metric(hr_np, pred_np, data_range=1.0)
        ssim_val = ssim_metric(hr_np, pred_np, data_range=1.0, channel_axis=2)
        print(f"[Sample {idx}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

        # ---- Display: brightness-enhanced versions for saving/viewing ----
        lr_display = brighten_for_display(lr_img)
        pred_display = brighten_for_display(pred)
        hr_display = brighten_for_display(hr_img)

        h, w = hr_display.shape[:2]
        lr_pil = Image.fromarray(lr_display).resize((w, h))
        pred_pil = Image.fromarray(pred_display).resize((w, h))
        hr_pil = Image.fromarray(hr_display)

        combined = Image.new("RGB", (w * 3 + 20, h))
        combined.paste(lr_pil, (0, 0))
        combined.paste(pred_pil, (w + 10, 0))
        combined.paste(hr_pil, (w * 2 + 20, 0))

        out_path = f"comparison_sample_{idx}.png"
        combined.save(out_path)
        print(f"  Saved comparison image: {out_path}  (left=LR input, middle=model output, right=real HR)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, nargs="+", default=[0, 1, 2],
                        help="Which sample indices to test (space separated for multiple)")
    args = parser.parse_args()
    main(args)
 