import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

ARCHITECTURE = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(architecture=ARCHITECTURE)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) is int:
                layers += [
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=x,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1
num_epochs = 10

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.CIFAR10(root="../datasets/", train=True, transform=transformation, download=True)
test_dataset = datasets.CIFAR10(root="../datasets/", train=False, transform=transformation, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = VGG(in_channels, num_classes).to(device=device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    epoch_loss, epoch_acc = [], []
    for idx, (batch, target) in enumerate(train_loader):
        batch = batch.to(device=device)
        target = target.to(device=device)

        scores = model(batch)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_loss.append(loss.item())
        _, preds = scores.max(dim=1)
        num_correct = (preds == target).sum()
        epoch_acc.append(num_correct / preds.size(0))

    epochLoss = sum(epoch_loss) / len(epoch_loss)
    epochAcc = sum(epoch_acc) / len(epoch_acc)
    print(f"Epoch: {epoch + 1} --- "
          f"Loss: {epochLoss}    "
          f"Acc: {epochAcc}")
