import torch
from torch import nn
from torch.nn import functional as F
import plot_utils

class Encoder(nn.Module):
    def __init__(self, input_dim, zdim, use_batch_norm=False, mode='vae'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.zdim = zdim
        self.mode = mode
        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(input_dim, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1_m = nn.Linear(16 * 5 * 5, 120)
        self.fc2_m = nn.Linear(120, 84)
        self.fc3_m = nn.Linear(84, zdim)
        self.fc1_bn1_m = nn.BatchNorm1d(120)
        self.fc2_bn2_m = nn.BatchNorm1d(84)

        self.fc1_mv = nn.Linear(16 * 5 * 5, 120)
        self.fc2_mv = nn.Linear(120, 84)
        self.fc3_mv = nn.Linear(84, zdim)
        self.fc1_bn1_mv = nn.BatchNorm1d(120)
        self.fc2_bn2_mv = nn.BatchNorm1d(84)

    def forward(self, x):
        if not self.use_batch_norm:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)

            m = F.relu(self.fc1_m(x))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            if self.mode == 'vae':
                v = F.relu(self.fc1_mv(x))
                v = F.relu(self.fc2_mv(v))
                v = self.fc3_mv(v)
            elif self.mode == 'ae':
                v = 0

            return m, v
        else:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 16 * 5 * 5)

            m = F.relu(self.fc1_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc2_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            if self.mode == 'vae':
                v = F.relu(self.fc1_bn1_mv(self.fc1_mv(x)))
                v = F.relu(self.fc2_bn2_mv(self.fc2_mv(v)))
                v = self.fc3_mv(v)
            elif self.mode == 'ae':
                v = 0
            return m, v


class Decoder(nn.Module):
    def __init__(self, zdim, output_dim, use_batch_norm=False):
        super(Decoder, self).__init__()

        self.zdim = zdim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.defc1 = nn.Linear(zdim, 84)
        self.defc2 = nn.Linear(84, 120)
        self.defc3 = nn.Linear(120, 16 * 5 * 5)
        self.deconv1 = nn.ConvTranspose2d(16, 16, 5, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = nn.ConvTranspose2d(16, 6, 3)
        self.deconv3 = nn.ConvTranspose2d(6, output_dim, 5)
        self.defc1_bn = nn.BatchNorm1d(84)
        self.defc2_bn = nn.BatchNorm1d(120)
        self.defc3_bn = nn.BatchNorm1d(16 * 5 * 5)
        self.deconv1_bn = nn.BatchNorm2d(16)
        self.deconv2_bn = nn.BatchNorm2d(6)

    def forward(self, x):
        if self.use_batch_norm:
            x = F.relu(self.defc1_bn(self.defc1(x)))
            x = F.relu(self.defc2_bn(self.defc2(x)))
            x = F.relu(self.defc3_bn(self.defc3(x)))
            x = x.reshape(-1, 16, 5, 5)

            x = F.relu(self.deconv1_bn(self.deconv1(x)))
            x = self.upsample(x)
            x = F.relu(self.deconv2_bn(self.deconv2(x)))
            x = torch.sigmoid(self.deconv3(x))
        else:
            x = F.relu(self.defc1(x))
            x = F.relu(self.defc2(x))
            x = F.relu(self.defc3(x))
            x = x.reshape(-1, 16, 5, 5)

            x = F.relu(self.deconv1(x))
            x = self.upsample(x)
            x = F.relu(self.deconv2(x))
            x = torch.sigmoid(self.deconv3(x))

        return x


def get_loss(encoder, decoder, input, target, mode):
    batch_size = input.size(0)
    # encoding
    mu, sigma = encoder(input)
    if mode == 'vae':
        z = mu + sigma * torch.randn_like(mu)
    elif mode == 'ae':
        z = mu

    # decoding
    y = decoder(z)
    y = torch.clamp(y,1e-8, 1-1e-8)

    #calculate l2_dis
    l2_dis = torch.sqrt(torch.sum((y - target)**2)) / batch_size

    # loss
    marginal_likelihood = -F.binary_cross_entropy(y, target, reduction='sum') / batch_size

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batch_size

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence, l2_dis


def get_ae(encoder, decoder, x, mode):
    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    if mode == 'vae':
        z = mu + sigma * torch.randn_like(mu)
    elif mode == 'ae':
        z = mu

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1-1e-8)

    return y

if __name__=='__main__':

    input = torch.randn(100, 3, 32, 32)
    input_dim = 3
    zdim = 12
    use_batch_norm = True
    encoder = Encoder(input_dim, zdim, use_batch_norm)      #output: num * z_dim, num * z_dim
    z_mu, z_sigma = encoder(input)
    print(z_mu.shape, z_sigma.shape)
    z = z_mu + z_sigma * torch.randn_like(z_mu)
    plot_utils.plot_t_sne(z.detach().numpy())

    output_dim = 3
    decoder = Decoder(zdim, output_dim, use_batch_norm)
    y = decoder(z)
    print(y.shape)


    l2_dis = torch.sqrt(torch.sum((y - input)**2)) / 100
    print(l2_dis)
