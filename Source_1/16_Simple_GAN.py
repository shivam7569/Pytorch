import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):

    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.dicriminator = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dicriminator(x)

        return x


class Generator(nn.Module):

    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.generator(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
epochs = 50

discriminator = Discriminator(img_dim=image_dim).to(device=device)
generator = Generator(z_dim=z_dim, img_dim=image_dim).to(device=device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device=device)
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="../datasets/", transform=transformations, download=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr)
optimizer_gen = optim.Adam(generator.parameters(), lr=lr)

criterion = nn.BCELoss()

writer_disc = SummaryWriter(f"../tensorboard/SimpleGAN/discriminator/")
writer_gen = SummaryWriter(f"../tensorboard/SimpleGAN/generator/")

step = 0
batch_step = 0

for epoch in range(epochs):
    for idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device=device)

        # Train Discriminator: max[log(D(real)) + log(1 - D(G(z)))]

        noise = torch.randn(batch_size, z_dim).to(device=device)
        fake = generator(noise)
        disc_real = discriminator(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        if epoch >= 5:
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            optimizer_disc.step()

        # Train Generator min[log(1 - D(G(z)))] <-> max[log(D(G(z)))]
        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        writer_gen.add_scalar("Loss", loss_gen.item(), global_step=batch_step)
        writer_disc.add_scalar("Loss", loss_disc.item(), global_step=batch_step)

        batch_step += 1

        if idx == 0:
            print(
                f"Epoch: {epoch + 1}/{epochs} --- Loss_D: {loss_disc.item():.3f} --- Loss_G: {loss_gen.item():.3f}"
            )
            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_gen.add_image("MNIST_Generated", img_grid_fake, global_step=step)
                writer_disc.add_image("MNIST_Original", img_grid_real, global_step=step)

                step += 1
