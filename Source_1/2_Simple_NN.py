# Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create NN

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters

inputSize = 784
numClasses = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 1

# Load Data

train_dataset = datasets.MNIST(root="../datasets/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="../datasets/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network

model = NN(inputSize, numClasses).to(device=device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Train Network

for epoch in range(num_epochs):
    for idx, (batch, target) in enumerate(train_loader):
        batch = batch.reshape(batch.shape[0], -1).to(device=device)
        target = target.to(device=device)

        scores = model(batch)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


# Check accuracy

def check_accuracy(model, loader):

    if loader.dataset.train:
        print("Checking accuracy on train data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.reshape(x.shape[0], -1).to(device=device), y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}")

    model.train()


check_accuracy(model, train_loader)
check_accuracy(model, test_loader)
