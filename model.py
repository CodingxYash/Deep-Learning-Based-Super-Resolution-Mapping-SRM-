import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    """Dense connections + residual — core idea SEN2SR/ESRGAN family use karte hain."""
    def __init__(self, ch=64, growth=32):
        super().__init__()
        self.c1 = nn.Conv2d(ch, growth, 3, padding=1)
        self.c2 = nn.Conv2d(ch + growth, growth, 3, padding=1)
        self.c3 = nn.Conv2d(ch + 2 * growth, growth, 3, padding=1)
        self.c4 = nn.Conv2d(ch + 3 * growth, ch, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.relu(self.c1(x))
        x2 = self.relu(self.c2(torch.cat([x, x1], 1)))
        x3 = self.relu(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.c4(torch.cat([x, x1, x2, x3], 1))
        return x + x4 * 0.2  # residual scaling, stability ke liye


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block — 3 RDBs stacked."""
    def __init__(self, ch=64, growth=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(ch, growth)
        self.rdb2 = ResidualDenseBlock(ch, growth)
        self.rdb3 = ResidualDenseBlock(ch, growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class Sen2SRInspired(nn.Module):
    def __init__(self, in_ch=3, feat_ch=64, num_blocks=6, scale=3):
        super().__init__()
        self.head = nn.Conv2d(in_ch, feat_ch, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(feat_ch) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(feat_ch, feat_ch, 3, padding=1)

        # Upsampling: scale=3 ke liye ek hi PixelShuffle step (3x3=9 channels expand)
        self.upsample = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.tail = nn.Conv2d(feat_ch, in_ch, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_conv(self.body(feat))
        feat = feat + body_out  # global residual
        up = self.upsample(feat)
        out = self.tail(up)
        return out