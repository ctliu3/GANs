from __future__ import print_function

import argparse
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description="vanilla gan")
parser.add_argument('--lr', type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--niter', type=int, default=1000,
                    help="training iteration number")
parser.add_argument('--save-image-interval', type=int, default=100,
                    help="save image in each interval")
parser.add_argument('--log-interval', type=int, default=100,
                    help="log interval")
parser.add_argument('--batch-size', type=int, default=2,
                    help="batch size in each iteration")
parser.add_argument('--step-k', type=int, default=1,
                    help="the number of steps to apply to the discriminator")
parser.add_argument('--num-worker', type=int, default=1,
                    help="number of worker to load images")
args = parser.parse_args()


def _flatten(x):
    x = x.view(x.size(0), -1)
    return x


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_channels, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_channels)

    def forward(self, input):
        x = _flatten(input)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.sigmoid(self.fc3(x))
        return x


class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, input):
        x = _flatten(input)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.sigmoid(self.fc3(x))
        return x


def get_data_iter():
    kwargs = {'num_workers': args.num_worker, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return iter(train_loader)


def next_batch(data_iter):
    batch = next(data_iter, None)
    if not batch:
        data_iter = get_data_iter()
        batch = next(data_iter, None)

    return batch


def save_image(images):
    for i in range(images.size(0)):
        image = images[i, :]
        image = image.numpy().reshape(28, 28)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save("image_{}.png".format(i))


def gen_random_data():
    return Variable(torch.randn(args.batch_size, 100)).cuda()


def main():
    in_channels = 100
    out_channels = 28 * 28
    G = Generator(in_channels, out_channels).cuda()
    D = Discriminator(out_channels).cuda()

    label_real = Variable(torch.ones(args.batch_size)).cuda()
    label_fake = Variable(torch.zeros(args.batch_size)).cuda()

    G_opt = torch.optim.Adam(G.parameters(), lr=args.lr)
    D_opt = torch.optim.Adam(D.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    data_iter = get_data_iter()
    for itr in range(args.niter):
        # Update D
        for k in range(args.step_k):
            X_random = gen_random_data()
            X, _ = next_batch(data_iter)
            X = Variable(X).cuda()

            X_fake = G(X_random)
            out_fake = D(X_fake)
            out = D(X)
            real_score = out
            fake_score = out_fake

            real_loss = criterion(out, label_real)
            fake_loss = criterion(out_fake, label_fake)
            D_loss = real_loss + fake_loss

            D_loss.backward()
            D_opt.step()
            D.zero_grad()
            G.zero_grad()

        # Update G
        X_random = gen_random_data()
        X_fake = G(X_random)
        out_fake = D(X_fake)
        G_loss = criterion(out_fake, label_real)

        G_loss.backward()
        G_opt.step()
        D.zero_grad()
        G.zero_grad()

        if (itr + 1) % args.log_interval == 0:
            print('#iter:{}, G-loss {}, D-loss {}, D(G(z)): {}, D(z) {}'.format(
                  itr + 1, G_loss.data[0], D_loss.data[0], fake_score.data.mean(),
                  real_score.data.mean()))

        if (itr + 1) % args.save_image_interval == 0:
            X_random = gen_random_data()
            X_fake = G(X_random)
            save_image(X_fake.data.cpu()[:10, :])


if __name__ == '__main__':
    main()
