import torch
import torch.nn as nn


class Decoder_Net(nn.Module):
    def __init__(self):
        super(Decoder_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(128, 6 * 7 * 7),
            nn.BatchNorm1d(6 * 7 * 7),
            nn.ReLU(inplace=True)
        )  # 6*7*7
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(6, 3, 3, 2, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )  # 3*14*14
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(3, 1, 3, 2, 1, 1),
            nn.ReLU(inplace=True)
        )  # 1*28*28

    def forward(self, x):
        out = self.layer1(x)
        out = torch.reshape(out, [out.size(0), 6, 7, 7])
        out = self.layer2(out)
        out = self.layer3(out)
        return out
