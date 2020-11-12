from AE.NetEncoder import Encoder_Net
from AE.NetDecoder import Decoder_Net
import torch


class MainNet(torch.nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder_Net()
        self.decoder = Decoder_Net()

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
