import os
import torch
import pandas as pd
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader
from skimage import io
import albumentations as A


class CustomData(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, y_label


def check_accuracy(model, loader):

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 4
num_epochs = 5

transformations = A.Compose([
    A.Resize(width=100, height=100),
    A.RandomCrop(width=90, height=90),
    A.Rotate(limit=45, p=0.6),
    A.HorizontalFlip(p=0.9),
    A.VerticalFlip(p=0.01),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.8),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=0.9),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255),
    ToTensorV2()
])

dataset = CustomData("../datasets/cats_dogs.csv", "../datasets/cats_dogs_resized/", transform=transformations)
train_dataset, test_dataset = random_split(dataset, [8, 2])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = torchvision.models.googlenet(pretrained=True).to(device=device)

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

check_accuracy(model, test_loader)
