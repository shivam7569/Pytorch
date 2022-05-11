import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


def load_image(image_path, loader):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)

    return image.to(device=DEVICE)


image_size = 356
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

original_image = load_image("../datasets/images/annahathaway.png", loader)
style_image = load_image("../datasets/styles/style5.jpg", loader)
generated = original_image.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 1e-3
alpha = 1
beta = 1e-1
model = VGG().to(device=DEVICE).eval()

optimizer = optim.Adam([generated], lr=learning_rate)
writer = SummaryWriter(f"../tensorboard/NST/")
global_step = 0

for step in range(total_steps):
    generated_features = model(generated)
    original_features = model(original_image)
    style_features = model(style_image)

    style_loss = original_loss = 0.0

    for genF, oriF, styF in zip(generated_features, original_features, style_features):
        batch_size, channels, height, width = genF.shape

        original_loss += torch.mean((genF - oriF) ** 2)

        gen_gram_matrix = genF.view(channels, height * width).mm(
            genF.view(channels, height * width).t()
        )

        sty_gram_matrix = styF.view(channels, height * width).mm(
            styF.view(channels, height * width).t()
        )

        style_loss += torch.mean((gen_gram_matrix - sty_gram_matrix) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    writer.add_scalar("Training Loss", scalar_value=total_loss.item(), global_step=global_step)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    global_step += 1

    if step % 100 == 0:
        print(f"Step: {step+1} --- Loss: {total_loss.item()}")
        if not os.path.exists("../outputs/NST/"):
            os.mkdir("../outputs/NST/")
        save_image(generated, f"../outputs/NST/Generated_Image_{step}.png")
