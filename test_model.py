import torch
from model import Sen2SRInspired

m = Sen2SRInspired(scale=3)
x = torch.randn(1, 3, 160, 160)
y = m(x)
print("Output shape:", y.shape)  # expect: (1, 3, 480, 480)