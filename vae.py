from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision.utils import save_image
import numpy as np
from fastai.vision import *


class VAE(nn.Module):
    def __init__(self, size_in, size_emb=5, enc_siz=[400], dec_siz=[400], channels=3):
        """
        Variational Autoencoder that is easily adaptable to different image sizes
        :param size_in: Size of the image in a single dimension. For example: 28 for MNIST
        :param size_emb: Size of the latent space encoding. (default: 5)
        :param enc_siz: The hidden layer sizes for the encoder. Must contain at least one layer.
        :param dec_siz: The hidden layer size for the decoder. Must contain at least one layer.
        :param channels: Number of channels in the in and output. (default: 3)
        """
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

    def vae_loss_function(self, recon_x, x):
        """Reconstruction + KL divergence losses summed over all elements and batch"""
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        return BCE + KLD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--emb-size', type=int, default=10, help='size of embedding (default 10)')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(42)

    mnist = untar_data(URLs.MNIST_TINY)
    tfms = get_transforms(do_flip=False)

    data = (ImageImageList.from_folder(mnist/'train')
            .split_by_rand_pct(0.1, seed=42)
            .label_from_func(lambda x: x)
            .transform(tfms)
            .databunch(num_workers=0, bs=args.batch_size)
            .normalize(do_y=True))

    image_size = data.one_batch()[0].shape[-1]
    vae = VAE(image_size, args.emb_size)
    my_learner = Learner(data,
                         vae,
                         opt_func=torch.optim.Adam,
                         loss_func=vae.vae_loss_function)
    my_learner.fit(args.epochs, lr=1e-2)
    my_learner.show_results(rows=4)

    print(f'Sampling 64 values and saving reconstruction. ')
    sample = torch.randn(64, args.emb_size)
    sample = vae.decode(sample).cpu()
    save_image(sample.view(64, 3, 28, 28), f'results/sample_{str(args.epochs)}.png')
