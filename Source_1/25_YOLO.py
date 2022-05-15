import os
import pandas as pd
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from metrics.iou import intersection_over_union
from metrics.mAP import mean_average_precision
from utils.utils import *


ARCHITECTURE = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]
SEED = 420
torch.manual_seed(SEED)
torch.autograd.set_detect_anomaly(True)


class YOLOv1(nn.Module):

    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(ARCHITECTURE)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)

        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) is tuple:
                layers += [
                    self.CNNBlock(
                        in_channels=in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3]
                    )
                ]
                in_channels = x[1]
            elif type(x) is str:
                layers += [
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                ]
            elif type(x) is list:
                for _ in range(x[-1]):
                    for c_layer in x[:-1]:
                        layers += [
                            self.CNNBlock(
                                in_channels=in_channels,
                                out_channels=c_layer[1],
                                kernel_size=c_layer[0],
                                stride=c_layer[2],
                                padding=c_layer[3]
                            )
                        ]
                        in_channels = c_layer[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Linear(1024 * split_size * split_size, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5)),
        )

    class CNNBlock(nn.Module):

        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=False,
                **kwargs
            )
            self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
            self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

        def forward(self, x):
            x = self.conv(x)
            x = self.batchnorm(x)
            x = self.leakyrelu(x)

            return x


class YoloLoss(nn.Module):

    def __init__(self, s=7, b=2, c=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = s, b, c
        self.lambda_noobj, self.lambda_coord = 0.5, 5
        
    def forward(self, predictions, targets):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)

        exists_box = targets[..., 20].unsqueeze(3)

        # LOSS FOR BOX COORDINATES

        box_predictions = exists_box * (bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25])
        box_targets = exists_box * targets[..., 21:25]
        box_predictions2 = box_predictions.clone()
        box_predictions2[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions2, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # LOSS FOR OBJECT

        pred_box = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., 20:21])
        )

        # LOSS FOR NO OBJECT

        no_obj_loss_box_1 = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )
        no_obj_loss_box_2 = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )
        no_obj_loss = no_obj_loss_box_1 + no_obj_loss_box_2

        # LOSS FOR CLASS

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * targets[..., :20], end_dim=-2)
        )

        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_obj_loss + class_loss

        return loss


class VOCDataset(Dataset):
    
    def __init__(self, csv_file, img_dir, label_dir, s=7, b=2, c=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = s
        self.C = c
        self.B = b

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path, 'r') as file:
            for label in file.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace('\n', "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            img, boxes = self.transform(img, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return img, label_matrix


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


LEARNING_RATE = 2e-5
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "../models/yolo.pth.tar"
IMG_DIR = "../datasets/yolo_data/8examples/images/"
LABEL_DIR = "../datasets/yolo_data/8examples/labels/"

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train(train_loader, model, optimizer, loss_fn):

    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = loss_fn(output, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())

    print(f"Mean Loss for this epoch: {sum(mean_loss)/len(mean_loss)}")


def main():

    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "../datasets/yolo_data/8examples/8examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    test_dataset = VOCDataset(
        "../datasets/yolo_data/8examples/8examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    _ = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHS):

        """for x, y in train_loader:
            x = x.to(DEVICE)
            for idx in range(8):
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, prob_threshold=0.4, box_format="midpoint")
                plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)
                
        import sys
        sys.exit()"""

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )
        mAP = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {mAP}")

        train(train_loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn)

        if mAP > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)


if __name__ == '__main__':
    main()
