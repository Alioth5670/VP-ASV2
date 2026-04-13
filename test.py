import torch.nn as nn


class ImageTextureExtractor(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
        )
        self.conv2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inplanes, inplanes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes*2),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inplanes*2, inplanes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes*4),
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        return [c1, c2, c3]