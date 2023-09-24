import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from MNIST import MNIST
from model import CNNClassifier

import os
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":

    DATA_FILE_PATH = os.path.join(
        str(Path(__file__).parent), "data", "train.csv")

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    NUM_EPOCHS = 5
    DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    SAVE_DIR = os.path.join(str(Path(__file__).parent), "runs")
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = MNIST(data_csv_path=DATA_FILE_PATH, transforms=img_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNNClassifier(in_channels=1)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        for idx, data in enumerate(tqdm(dataloader)):
            imgs, label = data

            imgs = imgs.to(DEVICE)
            labels = label.to(DEVICE)

            predictions = model(imgs)

            optimizer.zero_grad()
            loss = criterion(predictions, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            f"EPOCH {epoch + 1} / {NUM_EPOCHS} => Loss {running_loss / len(dataloader)}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))
