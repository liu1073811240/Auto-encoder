import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torchvision import datasets, transforms
from AE.NetEncoder import Encoder_Net
from AE.NetDecoder import Decoder_Net
import numpy as np

num_epoch = 10
if __name__ == '__main__':

    if not os.path.exists("params"):
        os.mkdir("params")
    if not os.path.exists("./img"):
        os.mkdir("./img")
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist_data = datasets.MNIST("../data", train=True,
                                transform=trans, download=True)
    train_loader = DataLoader(mnist_data, 100, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    en_net = Encoder_Net().to(device)
    de_net = Decoder_Net().to(device)

    en_net.train()
    de_net.train()
    if os.path.exists("./params/en_net.pth" and "./params/de_net.pth"):
        en_net.load_state_dict(
            torch.load("./params/en_net.pth"))
        de_net.load_state_dict(
            torch.load("./params/de_net.pth"))
    else:
        print("No Params!")

    loss_fn = nn.MSELoss()
    en_optimizer = torch.optim.Adam(en_net.parameters())
    de_optimizer = torch.optim.Adam(de_net.parameters())

    for epoch in range(num_epoch):
        for i, (img, label) in enumerate(train_loader):

            img = img.to(device)
            feature = en_net(img)
            out_img = de_net(feature)
            # print(out_img.shape)
            loss = loss_fn(img, out_img)

            en_optimizer.zero_grad()
            de_optimizer.zero_grad()

            loss.backward()
            en_optimizer.step()
            de_optimizer.step()

            if i % 100 == 0:
                print('Epoch [{}/{}], loss: {:.3f}'
                      .format(epoch, num_epoch, loss))
        # images = out_img.cpu().data

        print(np.shape(out_img))  # 拿到这一轮图片的最后一批，torch.Size([100, 1, 28, 28])
        fake_images = out_img.cpu().data
        save_image(fake_images, './img/{}-fake_images.png'
                   .format(epoch + 1), nrow=10)
        real_images = img.cpu().data
        save_image(real_images, './img/{}-real_images.png'  # 可以保存一批图片
                   .format(epoch + 1), nrow=10)
        torch.save(en_net.state_dict(), "./params/en_net.pth")
        torch.save(de_net.state_dict(), "./params/de_net.pth")
