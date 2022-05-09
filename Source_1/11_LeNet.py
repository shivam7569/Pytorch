import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


class LeNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=6,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x


in_channels = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 128
num_epochs = 10

transformation = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root="../datasets/", train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST(root="../datasets/", train=False, transform=transformation, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = LeNet(in_channels, num_classes).to(device=device)

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
