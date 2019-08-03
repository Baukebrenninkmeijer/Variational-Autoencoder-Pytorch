from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from fastai.vision import *

class VAE(nn.Module):
    def __init__(self, size_in, size_emb, dec_siz=[400], enc_siz=[400], channels=3):
        super().__init__()
        self.enc = []
        self.dec = []
        self.channels = channels
        self.size_in = size_in
        self.size_in_sqr = size_in**2
        enc_siz = [self.size_in_sqr*channels] + enc_siz
        dec_siz = [size_emb] + dec_siz

        for i, layer in enumerate(enc_siz):
            if i + 1 == len(enc_siz): break
            self.enc += [
                nn.Linear(enc_siz[i], enc_siz[i+1], bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(enc_siz[i+1])
            ]

        for i, layer in enumerate(dec_siz):
            if i + 1 == len(dec_siz): break
            self.dec += [
                nn.Linear(dec_siz[i], dec_siz[i + 1], bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(dec_siz[i + 1])
            ]

        self.encoder = nn.Sequential(*self.enc)
        self.decoder = nn.Sequential(*self.dec)

        self.fc_mu = nn.Linear(enc_siz[-1], size_emb)
        self.fc_std = nn.Linear(enc_siz[-1], size_emb)
        self.out = nn.Linear(dec_siz[-1], self.size_in_sqr*self.channels)

        self.bn_mu = nn.BatchNorm1d(size_emb)
        self.bn_std = nn.BatchNorm1d(size_emb)

        self.mu = None
        self.logvar = None

    def encode(self, x):
        x = self.encoder(x.view(-1, self.size_in_sqr*self.channels))
        return self.bn_mu(self.fc_mu(x)), self.bn_std(self.fc_std(x))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        return torch.sigmoid(self.out(x)).view(-1, self.channels, self.size_in, self.size_in)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decode(z)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def vae_loss_function(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--emb-size', type=int, default=10, help='size of embedding (default 10)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(42)

    device = torch.device("cuda" if args.cuda else "cpu")

    mnist = untar_data(URLs.MNIST_TINY)
    tfms = get_transforms(do_flip=False)

    data = (ImageImageList.from_folder(mnist/'train')
            .split_by_rand_pct(0.1, seed=42)
            .label_from_func(lambda x: x)
            .transform(tfms)
            .databunch(num_workers=0, bs=16)
            .normalize(do_y=True))

    image_size = data.one_batch()[0].shape[-1]
    vae = VAE(image_size, args.emb_size)
    my_learner = Learner(data,
                         vae,
                         opt_func=torch.optim.Adam,
                         loss_func=vae.vae_loss_function)
    my_learner.fit(args.epochs, lr=1e-2)
    my_learner.show_results(rows=4)

    # for epoch in range(1, args.epochs + 1):
    #     train(epoch)
    #     test(epoch)
    #     with torch.no_grad():
    #         sample = torch.randn(64, 20).to(device)
    #         sample = model.decode(sample).cpu()
    #         save_image(sample.view(64, 1, 28, 28),
    #                    'results/sample_' + str(epoch) + '.png')