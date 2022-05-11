import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms


class Discriminator(nn.Module):

    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_img,
                out_channels=features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(
                in_channels=features_d*8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0
            )
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):

    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0),
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(
                in_channels=features_g*2,
                out_channels=channels_img,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.generator(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def check():
    N, in_channels, H, W, z_dim = 8, 3, 64, 64, 100
    x = torch.randn((N, in_channels, H, W))
    discriminator = Discriminator(channels_img=in_channels, features_d=8)
    initialize_weights(discriminator)
    assert discriminator(x).shape == (N, 1, 1, 1)
    generator = Generator(z_dim=z_dim, channels_img=in_channels, features_g=8)
    initialize_weights(generator)
    z = torch.randn((N, z_dim, 1, 1))
    assert generator(z).shape == (N, 3, 64, 64)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
EPOCHS = 100
FEATURES_D = FEATURES_G = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 1e-2

transformations = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5 for _ in range(CHANNELS_IMG)], std=[0.5 for _ in range(CHANNELS_IMG)])
])

dataset = datasets.MNIST(root="../datasets/", transform=transformations, download=True)
# dataset = datasets.ImageFolder(root="../datasets/celebA/", transform=transformations)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

generator = Generator(z_dim=Z_DIM, channels_img=CHANNELS_IMG, features_g=FEATURES_G).to(device=device)
critic = Discriminator(channels_img=CHANNELS_IMG, features_d=FEATURES_D).to(device=device)
initialize_weights(critic)
initialize_weights(generator)

optimizerG = optim.RMSprop(generator.parameters(), lr=LEARNING_RATE)
optimizerC = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device=device)
writer_disc = SummaryWriter(f"../tensorboard/WGAN/discriminator/")
writer_gen = SummaryWriter(f"../tensorboard/WGAN/generator/")
epoch_step = 0

for epoch in range(EPOCHS):
    epoch_loss_G = epoch_loss_C = []
    for idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device=device)

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device=device)
            fake = generator(noise)
            critic_real = critic(real).view(-1)
            critic_fake = critic(fake).view(-1)
            loss_critic = -1 * (torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizerC.step()

            epoch_loss_C.append(loss_critic.item())

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        output = critic(fake).view(-1)
        loss_gen = -1 * torch.mean(output)
        generator.zero_grad()
        loss_gen.backward()
        optimizerG.step()

        epoch_loss_G.append(loss_gen.item())

        if idx == 0:

            genLoss = sum(epoch_loss_G) / len(epoch_loss_G)
            criLoss = sum(epoch_loss_C) / len(epoch_loss_C)

            print(
                f"Epoch: {epoch + 1}/{EPOCHS} --- Loss_C: {criLoss:.3f} --- Loss_G: {genLoss:.3f}"
            )
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_gen.add_image("Generated", img_grid_fake, global_step=epoch_step)
                writer_disc.add_image("Original", img_grid_real, global_step=epoch_step)

                writer_gen.add_scalar("Loss", genLoss, global_step=epoch_step)
                writer_disc.add_scalar("Loss", criLoss, global_step=epoch_step)

                epoch_step += 1
