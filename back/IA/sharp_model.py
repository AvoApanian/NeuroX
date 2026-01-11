import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))


class RCAB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.ca = ChannelAttention(ch)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res * 0.1


class ResidualGroup(nn.Module):
    def __init__(self, ch, n_blocks=4):
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(ch) for _ in range(n_blocks)])
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)
        return x + res * 0.1


class SharpSRModel(nn.Module):
    def __init__(self):
        super().__init__()
        nf = 64

        self.head = nn.Conv2d(3, nf, 3, padding=1)

        self.body = nn.Sequential(
            ResidualGroup(nf, 4),
            ResidualGroup(nf, 4)
        )

        self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)

        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 2, nf // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 4, 3, 3, padding=1)
        )

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        res += feat
        residual = self.tail(res) * 0.25
        return torch.clamp(x + residual, 0, 1)
 