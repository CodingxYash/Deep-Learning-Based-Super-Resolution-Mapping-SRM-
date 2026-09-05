import torch
from torch.utils.data import DataLoader, Subset
from dataset import SRDataset
from model import Sen2SRInspired
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Two dataset "views" of the same files — one augmented (for training), one clean (for validation)
train_view = SRDataset("train_lr/train_lr", "train_hr_all", augment=True)
val_view = SRDataset("train_lr/train_lr", "train_hr_all", augment=False)

total_len = len(train_view)
print("Total pairs available:", total_len)

# Same split indices used for both views, so train/val don't overlap
indices = list(range(total_len))
random.Random(42).shuffle(indices)
val_size = int(0.1 * total_len)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

train_ds = Subset(train_view, train_indices)
val_ds = Subset(val_view, val_indices)
print(f"Train: {len(train_ds)} | Validation: {len(val_ds)}")

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

model = Sen2SRInspired(scale=3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
loss_fn = torch.nn.L1Loss()

max_epochs = 40
patience = 6
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(max_epochs):
    model.train()
    train_loss = 0
    for lr_img, hr_img in train_dl:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        out = model(lr_img)
        loss = loss_fn(out, hr_img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for lr_img, hr_img in val_dl:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            out = model(lr_img)
            loss = loss_fn(out, hr_img)
            val_loss += loss.item()
    val_loss /= len(val_dl)

    scheduler.step(val_loss)
    current_lr = opt.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{max_epochs} | Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "model.pth")
        print(f"  -> New best model saved (val loss: {val_loss:.5f})")
    else:
        epochs_without_improvement += 1
        print(f"  -> No improvement ({epochs_without_improvement}/{patience})")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

print("Training complete. Best model saved as model.pth")