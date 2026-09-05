"""
Run the trained super-resolution model on ANY custom satellite image 
(jpg/png), not just the Kaggle dataset pairs.

Usage:
    python test_custom_image.py --image my_satellite_photo.jpg
"""

import argparse
import numpy as np
import torch
from PIL import Image

from model import Sen2SRInspired


def brighten_for_display(arr_hwc, gamma=0.85):
    """
    arr_hwc: numpy array (H, W, C), float, any range.
    Percentile stretch + mild gamma correction for natural-looking display.
    """
    out = np.zeros_like(arr_hwc, dtype=np.float32)
    for c in range(arr_hwc.shape[2]):
        channel = arr_hwc[:, :, c]
        low, high = np.percentile(channel, (1, 99))
        channel = np.clip((channel - low) / (high - low + 1e-8), 0, 1)
        out[:, :, c] = channel
    out = np.power(out, gamma)
    return (out * 255).clip(0, 255).astype(np.uint8)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Load trained model ----
    model = Sen2SRInspired(scale=3).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    # ---- Load and prepare the custom image ----
    img = Image.open(args.image).convert("RGB")
    img = img.resize((160, 160))  # model expects 160x160 input

    arr = np.array(img).astype("float32")          # (160,160,3), range 0-255
    arr = arr / 255.0                                # normalize to 0-1
    arr = arr * 0.24 + 0.03                          # map into training range (~0.03-0.27)
    arr_chw = np.transpose(arr, (2, 0, 1))           # (3,160,160), channels-first

    lr_tensor = torch.from_numpy(arr_chw).float().unsqueeze(0).to(device)  # (1,3,160,160)

    # ---- Run inference ----
    with torch.no_grad():
        pred = model(lr_tensor)[0]  # (3,480,480)

    # ---- Prepare images for display ----
    lr_display_arr = np.transpose(arr_chw, (1, 2, 0))         # (160,160,3)
    pred_display_arr = pred.detach().cpu().permute(1, 2, 0).numpy()  # (480,480,3)

    lr_bright = brighten_for_display(lr_display_arr)
    pred_bright = brighten_for_display(pred_display_arr)

    # Resize LR up to match HR size for side-by-side comparison
    lr_pil = Image.fromarray(lr_bright).resize((480, 480))
    pred_pil = Image.fromarray(pred_bright)

    combined = Image.new("RGB", (480 * 2 + 10, 480))
    combined.paste(lr_pil, (0, 0))
    combined.paste(pred_pil, (490, 0))

    out_path = "custom_image_result.png"
    combined.save(out_path)
    print(f"Saved: {out_path}  (left=input upscaled for viewing, right=model super-resolved output)")

    # Also save just the output alone, full size
    pred_pil.save("custom_image_output_only.png")
    print("Saved: custom_image_output_only.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to your custom satellite image (jpg/png)")
    args = parser.parse_args()
    main(args)