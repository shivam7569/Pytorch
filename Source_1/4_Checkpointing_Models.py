# Imports
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create NN

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels*2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=input_channels*2,
            out_channels=input_channels*4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1))
        self.fc1 = nn.Linear(in_features=input_channels*4*7*7, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters

input_channels = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

# Load Data

train_dataset = datasets.MNIST(root="../datasets/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="../datasets/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Saving Model

def save_checkpoint(checkpoint, name="../models/checkpoint.pth.tar"):
    print("SAVING CHECKPOINT")
    if not os.path.exists("../models/"):
        os.mkdir("../models")
    torch.save(checkpoint, name)


# Loading Model

def load_checkpoint(checkpoint, model, optimizer):
    print("LOADING CHECKPOINT")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


# Check accuracy

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


# Initialize Network

model = CNN(input_channels, num_classes).to(device=device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

model, optimizer = load_checkpoint(torch.load("../models/checkpoint.pth.tar"), model, optimizer)

# Train Network

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
    if epoch % 2 == 0:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

check_accuracy(model, test_loader, True)
