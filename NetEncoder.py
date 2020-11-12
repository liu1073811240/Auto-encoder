import torch
import torch.nn as nn


class Encoder_Net(nn.Module):
    def __init__(self):
        super(Encoder_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 2, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )  # 3*14*14=588
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 6, 3, 2, 1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True)
        )  # 6*7*7=294
        self.layer3 = nn.Sequential(
            nn.Linear(6 * 7 * 7, 128),
            nn.Sigmoid()
        )  # 128

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.reshape(out, [out.size(0), -1])
        out = self.layer3(out)
        return out
