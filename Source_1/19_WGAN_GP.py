import os

import torch
import torch.nn as nn
import torchvision
from torch import optim
from tqdm import tqdm
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
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
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


def gradient_penalty(critic, real, fake, device):
    batch_size, C, W, H = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = epsilon * real + (1 - epsilon) * fake

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradientPenalty = torch.mean((gradient_norm - 1) ** 2)

    return gradientPenalty


def save_checkpoint(checkpoint, name="../models/checkpoint.pth.tar"):
    print("SAVING CHECKPOINT")
    if not os.path.exists("../models/"):
        os.mkdir("../models")
    torch.save(checkpoint, name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
EPOCHS = 100
FEATURES_D = FEATURES_G = 64
CRITIC_ITERATIONS = 1
LAMBDA = 10

transformations = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5 for _ in range(CHANNELS_IMG)], std=[0.5 for _ in range(CHANNELS_IMG)])
])

# dataset = datasets.MNIST(root="../datasets/", transform=transformations, download=True)
dataset = datasets.ImageFolder(root="../datasets/celebA/", transform=transformations)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

generator = Generator(z_dim=Z_DIM, channels_img=CHANNELS_IMG, features_g=FEATURES_G).to(device=device)
critic = Discriminator(channels_img=CHANNELS_IMG, features_d=FEATURES_D).to(device=device)
initialize_weights(critic)
initialize_weights(generator)

optimizerG = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
optimizerC = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device=device)
writer_disc = SummaryWriter(f"../tensorboard/WGAN_GP/discriminator/")
writer_gen = SummaryWriter(f"../tensorboard/WGAN_GP/generator/")
epoch_step = batch_step = 0

loss_critic = fake = None
NUM_BATCHES = len(loader)

for epoch in range(EPOCHS):
    batch_loop = tqdm(enumerate(loader), total=NUM_BATCHES, leave=False)
    for idx, (real, _) in batch_loop:
        real = real.to(device=device)

        for _ in range(CRITIC_ITERATIONS):
            batch_size = real.shape[0]
            noise = torch.randn((batch_size, Z_DIM, 1, 1)).to(device=device)
            fake = generator(noise)
            critic_real = critic(real).view(-1)
            critic_fake = critic(fake).view(-1)
            gp = gradient_penalty(critic, real, fake, device)
            loss_critic = -1 * (torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizerC.step()

        output = critic(fake).view(-1)
        loss_gen = -1 * torch.mean(output)
        generator.zero_grad()
        loss_gen.backward()
        optimizerG.step()

        genLoss = loss_gen.item()
        criLoss = loss_critic.item()

        writer_gen.add_scalar("Loss", genLoss, global_step=batch_step)
        writer_disc.add_scalar("Loss", criLoss, global_step=batch_step)

        batch_step += 1

        if idx % 100 == 0 and idx > 0:

            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_gen.add_image("Generated", img_grid_fake, global_step=epoch_step)
                writer_disc.add_image("Original", img_grid_real, global_step=epoch_step)

                epoch_step += 1

        batch_loop.set_description(f"Epoch: {epoch}/{EPOCHS}")
        batch_loop.set_postfix(Loss_C=criLoss, Loss_G=genLoss)

    if epoch > 0:
        checkpointG = {"state_dict": generator.state_dict(), "optimizer": optimizerG.state_dict()}
        checkpointC = {"state_dict": critic.state_dict(), "optimizer": optimizerC.state_dict()}
        save_checkpoint(checkpointG, name=f"../models/WGAN_GP_Generator_{epoch}.pth.tar")
        save_checkpoint(checkpointC, name=f"../models/WGAN_GP_Critic_{epoch}.pth.tar")
