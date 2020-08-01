import torch
from torch import nn
from torch.nn import functional as F
import numpy

class Encoder(nn.Module):
    def __init__(self, input_dim, zdim, use_batch_norm=False, mode='vae'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.zdim = zdim
        self.mode = mode
        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.fc1_m = nn.Linear(32 * 16 * 16, 128)
        self.fc2_m = nn.Linear(128, zdim)
        self.fc1_bn1_m = nn.BatchNorm1d(128)

        self.fc1_mv = nn.Linear(32 * 16 * 16, 128)
        self.fc2_mv = nn.Linear(128, zdim)
        self.fc1_bn1_mv = nn.BatchNorm1d(128)

    def forward(self, x):
        if not self.use_batch_norm:
            x = F.relu(self.conv1(x))               # 3 * 32 * 32
            x = F.relu(self.conv2(x))               # 16 * 16 * 16
            x = F.relu(self.conv3(x))               # 32 * 16 * 16
            x = F.relu(self.conv4(x))               # 32 * 16 * 16

            x = x.view(-1, 32 * 16 * 16)

            m = F.relu(self.fc1_m(x))               # N * 128
            m = self.fc2_m(m)                       # N * zdim

            if self.mode == 'vae':
                #v = F.relu(self.fc1_mv(x))
                #v = self.fc2_mv(v)
                v = F.elu(self.fc1_mv(x))
                v = F.elu(self.fc2_mv(v))
            elif self.mode == 'ae':
                v = 0
            return m, v
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.view(-1, 32 * 16 * 16)

            m = F.relu(self.fc1_bn1_m(self.fc1_m(x)))
            m = self.fc2_m(m)
            if self.mode == 'vae':
                #v = F.relu(self.fc1_bn1_mv(self.fc1_mv(x)))
                #v = self.fc2_mv(v)
                v = F.elu(self.fc1_bn1_mv(self.fc1_mv(x)))
                v = F.elu(self.fc2_mv(v))
            elif self.mode == 'ae':
                v = 0
            return m, v

class Decoder(nn.Module):
    def __init__(self, zdim, output_dim, use_batch_norm=False):
        super(Decoder, self).__init__()

        self.zdim = zdim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.defc1 = nn.Linear(zdim, 128)
        self.defc2 = nn.Linear(128, 32 * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 3, padding=1, output_padding=1, stride=2)
        #self.deconv4 = nn.ConvTranspose2d(16, 3, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 3, padding=1)

        self.defc1_bn = nn.BatchNorm1d(128)
        self.defc2_bn = nn.BatchNorm1d(32 * 16 * 16)
        self.deconv1_bn = nn.BatchNorm2d(32)
        self.deconv2_bn = nn.BatchNorm2d(32)
        self.deconv3_bn = nn.BatchNorm2d(16)


    def forward(self, x):
        if self.use_batch_norm:
            x = F.relu(self.defc1_bn(self.defc1(x)))
            x = F.relu(self.defc2_bn(self.defc2(x)))
            x = x.reshape(-1, 32, 16, 16)

            x = F.relu(self.deconv1_bn(self.deconv1(x)))
            x = F.relu(self.deconv2_bn(self.deconv2(x)))
            x = F.relu(self.deconv3_bn(self.deconv3(x)))
            x = torch.sigmoid(self.conv4(x))
        else:
            x = F.relu(self.defc1(x))
            x = F.relu(self.defc2(x))
            x = x.reshape(-1, 32, 16, 16)

            x = F.relu(self.deconv1(x))     # 32 * 16 * 16
            x = F.relu(self.deconv2(x))     # 32 * 16 * 16
            x = F.relu(self.deconv3(x))     # 32 * 32 * 32
            x = torch.sigmoid(self.conv4(x))    #3 * 32 * 32

        return x

def get_ae(encoder, decoder, x):
    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    std = torch.exp(0.5 * sigma)
    eps = torch.randn(sigma.size()).to(mu)
    z = mu + eps * std

    # decoding
    y = decoder(z)

    return y



def get_z(encoder, x):

    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    std = torch.exp(0.5 * sigma)
    eps = torch.randn(sigma.size()).to(mu)
    z = mu + eps * std

    return z

def get_loss(encoder, decoder, input, target, mode):
    batch_size = input.size(0)
    # encoding
    mu, sigma = encoder(input)
    if mode == 'vae':
        std = torch.exp(0.5 * sigma)
        eps = torch.randn(sigma.size()).to(mu)
        z = mu + eps * std
    elif mode == 'ae':
        z = mu

    # decoding
    y = decoder(z)

    y = y.view(-1, 3 * 32 * 32)
    target = target.view(-1, 3 * 32 * 32)
    #marginal_likelihood = F.binary_cross_entropy(y, target, reduction='sum') / batch_size
    marginal_likelihood = F.binary_cross_entropy(y, target) * 32 * 32
    l2_dis = torch.sum((y - target) ** 2) / batch_size
    KL_divergence = -0.5 * torch.sum(1 + sigma - torch.pow(mu, 2) - torch.exp(sigma)
                               ).sum() / batch_size
    #loss = marginal_likelihood + KL_divergence

    loss = l2_dis + KL_divergence
    return y, z, loss, marginal_likelihood, KL_divergence, l2_dis



if __name__=='__main__':
    x = torch.randn(3, 5)
    encoder = Encoder(3, 2, use_batch_norm=False)
    z_mu, z_sigma = encoder(x)
    print(z_mu, z_sigma)
