from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np


# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)
#
# device = torch.device("cuda" if args.cuda else "cpu")

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, size_in, size_emb):
        super().__init__()
        print(size_in, size_emb)
        self.fc1 = nn.Linear(size_in, 400)
        self.fc21 = nn.Linear(400, size_emb)
        self.fc22 = nn.Linear(400, size_emb)
        self.fc3 = nn.Linear(size_emb, 400)
        self.fc4 = nn.Linear(400, size_in)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc21.weight)
        torch.nn.init.xavier_uniform_(self.fc22.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(size_emb)
        self.bn3 = nn.BatchNorm1d(size_emb)

        self.mu = None
        self.logvar = None

    def encode(self, x):
        x = x[:, 0, ...]
        x = x.view(-1, 784)
        # print(x)
        h1 = self.bn1(F.relu(self.fc1(x)))
        # print(h1)
        return self.bn2(self.fc21(h1)), self.bn2(self.fc22(h1))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        b = z
        d = h3
        # print(np.sum(np.isnan(b.detach().numpy())))
        # print(np.sum(np.isnan(d.detach().numpy())))
        # print(b, d)
        return torch.cat([torch.sigmoid(self.fc4(h3))]*3).view(-1, 3, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3, 784))
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decode(z)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def vae_loss_function(self, recon_x, x):
        # print(recon_x, x)
        # print(recon_x.shape, x.shape)
        # print(np.sum(np.isnan(recon_x.numpy())), np.sum(np.isnan(x.numpy())))
        BCE = F.binary_cross_entropy(recon_x.view(-1, 3, 28, 28), x.view(-1, 3, 28, 28), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        return BCE + KLD


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)




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

# if __name__ == "__main__":
#     for epoch in range(1, args.epochs + 1):
#         train(epoch)
#         test(epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, 20).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        'results/sample_' + str(epoch) + '.png')