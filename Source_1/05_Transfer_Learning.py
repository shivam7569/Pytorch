import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision


class Identity(nn.Module):
    def __int__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
print(model)
model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, num_classes)
)
model.to(device=device)
print(model)

train_dataset = datasets.CIFAR10(root="../datasets/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root="../datasets/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


def check_accuracy(model, loader, log=False):

    if loader.dataset.train:
        if log:
            print("Checking accuracy on train data")
    else:
        if log:
            print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device=device), y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}")

    model.train()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for idx, (batch, target) in enumerate(train_loader):
        batch = batch.to(device=device)
        target = target.to(device=device)

        scores = model(batch)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    check_accuracy(model, train_loader)

check_accuracy(model, test_loader, True)
