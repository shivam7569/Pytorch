# Imports
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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


def arrange_dir(thedir):
    dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]
    for dir_ in dirs:
        for file in os.listdir(os.path.join(thedir, dir_)):
            os.rename(os.path.join(thedir, dir_, file), os.path.join(thedir, file))
            os.rmdir(os.path.join(thedir, dir_))


# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters

input_channels = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 256
num_epochs = 5

# Load Data

transformations = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.RandomCrop((28, 28)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, ], std=[1.0, ])
])

train_dataset = datasets.MNIST(root="../datasets/", train=True, transform=transformations, download=True)
test_dataset = datasets.MNIST(root="../datasets/", train=False, transform=transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()

# For Hyperparameter Search

batch_sizes = [batch_size]  # [16, 32, 128, 1024]
lrs = [learning_rate]  # [1e-1, 1e-2, 1e-3, 1e-4]
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Train Network

for batchSize in batch_sizes:
    for lr in lrs:
        model = CNN(input_channels, num_classes).to(device=device)
        model.train()

        train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        writer = SummaryWriter(f"../tensorboard/mnist/Batch_{batchSize}_lr_{lr}/")
        step = 0

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

                img_grid = torchvision.utils.make_grid(batch)
                features = batch.reshape(batch.shape[0], -1)

                epoch_loss.append(loss.item())
                _, preds = scores.max(dim=1)
                num_correct = (preds == target).sum()
                epoch_acc.append(num_correct/preds.size(0))

                class_labels = [classes[label] for label in preds]

                writer.add_scalar("Training Loss", scalar_value=loss, global_step=step)
                writer.add_scalar("Training Acc", scalar_value=num_correct/preds.size(0), global_step=step)
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("FC1", model.fc1.weight)

                if idx == 230:
                    writer.add_embedding(features, metadata=class_labels, label_img=batch, global_step=idx)
                step += 1

            epochLoss = sum(epoch_loss)/len(epoch_loss)
            epochAcc = sum(epoch_acc)/len(epoch_acc)
            print(f"Batch_LR: {batchSize}_{lr}    "
                  f"Epoch: {epoch+1} --- "
                  f"Loss: {epochLoss}    "
                  f"Acc: {epochAcc}")
            writer.add_hparams(hparam_dict={"lr": lr, "batch_size": batchSize},
                               metric_dict={"loss": epochLoss, "acc": epochAcc})

        arrange_dir(f"../tensorboard/mnist/Batch_{batchSize}_lr_{lr}/")
