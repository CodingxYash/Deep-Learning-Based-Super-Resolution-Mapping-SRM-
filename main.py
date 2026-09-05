from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch, io, base64
import numpy as np
import tifffile
from PIL import Image
from model import Sen2SRInspired
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sen2SRInspired(scale=3).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

def tensor_to_b64_png(t: torch.Tensor) -> str:
    """(3,H,W) float tensor -> base64 PNG for browser display."""
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))  # (H,W,3)
    # Reflectance values 0.03-0.27 jaisi thi -> visualize ke liye stretch karo
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.post("/infer")
async def infer(lr_file: UploadFile = File(...), hr_file: UploadFile = File(None)):
    lr_bytes = await lr_file.read()
    lr_np = tifffile.imread(io.BytesIO(lr_bytes))  # (3,160,160) float32
    lr_t = torch.from_numpy(lr_np).float().unsqueeze(0).to(device)  # (1,3,160,160)

    with torch.no_grad():
        out_t = model(lr_t).squeeze(0).clamp(lr_np.min(), lr_np.max() * 2)  # (3,480,480)

    result = {
        "input_b64": tensor_to_b64_png(torch.from_numpy(lr_np)),
        "output_b64": tensor_to_b64_png(out_t),
    }

    if hr_file is not None:
        hr_bytes = await hr_file.read()
        hr_np = tifffile.imread(io.BytesIO(hr_bytes))  # (3,480,480)
        hr_t = torch.from_numpy(hr_np).float()

        out_np = out_t.cpu().numpy()
        result["hr_b64"] = tensor_to_b64_png(hr_t)
        result["psnr"] = round(float(psnr(hr_np, out_np, data_range=hr_np.max() - hr_np.min())), 2)
        result["ssim"] = round(float(ssim(
            np.transpose(hr_np, (1, 2, 0)),
            np.transpose(out_np, (1, 2, 0)),
            channel_axis=2,
            data_range=hr_np.max() - hr_np.min()
        )), 3)

    return JSONResponse(result)

@app.get("/health")
def health():
    return {"status": "ok", "device": device}