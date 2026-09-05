import os
import io
import random
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter


def degrade_lr(arr_chw):
    """
    arr_chw: (3, H, W) float32, satellite reflectance range (small values like 0.03-0.27)
    Randomly degrades the image to simulate real-world conditions:
    blur, JPEG compression, noise, brightness/contrast shifts.
    Returns degraded array in the SAME value range as input.
    """
    lo, hi = arr_chw.min(), arr_chw.max()
    norm = (arr_chw - lo) / (hi - lo + 1e-8)  # -> 0-1
    img_uint8 = (norm * 255).clip(0, 255).astype(np.uint8)
    img_hwc = np.transpose(img_uint8, (1, 2, 0))
    pil_img = Image.fromarray(img_hwc)

    # Random Gaussian blur
    if random.random() < 0.5:
        radius = random.uniform(0.3, 1.5)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius))

    # Random JPEG compression artifacts
    if random.random() < 0.5:
        quality = random.randint(30, 90)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")

    arr_deg = np.array(pil_img).astype(np.float32) / 255.0  # (H,W,3), 0-1

    # Random Gaussian noise
    if random.random() < 0.5:
        sigma = random.uniform(0.01, 0.05)
        arr_deg = arr_deg + np.random.normal(0, sigma, arr_deg.shape).astype(np.float32)
        arr_deg = np.clip(arr_deg, 0, 1)

    # Random brightness/contrast jitter
    if random.random() < 0.5:
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        arr_deg = (arr_deg - 0.5) * contrast + 0.5
        arr_deg = arr_deg * brightness
        arr_deg = np.clip(arr_deg, 0, 1)

    # Scale back to original reflectance range
    arr_deg = arr_deg * (hi - lo) + lo
    return np.transpose(arr_deg, (2, 0, 1)).astype(np.float32)


def geometric_augment(lr, hr):
    """Apply the SAME random flip/rotation to both LR and HR (keeps them aligned)."""
    if random.random() < 0.5:
        lr = np.flip(lr, axis=2).copy()  # horizontal flip (W axis)
        hr = np.flip(hr, axis=2).copy()
    if random.random() < 0.5:
        lr = np.flip(lr, axis=1).copy()  # vertical flip (H axis)
        hr = np.flip(hr, axis=1).copy()
    k = random.choice([0, 1, 2, 3])  # 0/90/180/270 degree rotation
    if k > 0:
        lr = np.rot90(lr, k, axes=(1, 2)).copy()
        hr = np.rot90(hr, k, axes=(1, 2)).copy()
    return lr, hr


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, augment=False):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.augment = augment
        lr_files = set(os.listdir(lr_dir))
        hr_files = set(os.listdir(hr_dir))
        self.files = sorted(lr_files & hr_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        lr = tifffile.imread(os.path.join(self.lr_dir, fname))  # (3,160,160)
        hr = tifffile.imread(os.path.join(self.hr_dir, fname))  # (3,480,480)

        if self.augment:
            lr, hr = geometric_augment(lr, hr)
            lr = degrade_lr(lr)  # only LR gets extra degradation, HR stays clean ground truth

        lr_t = torch.from_numpy(np.ascontiguousarray(lr)).float()
        hr_t = torch.from_numpy(np.ascontiguousarray(hr)).float()
        return lr_t, hr_t