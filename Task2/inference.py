import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from MNIST import MNIST
from model import CNNClassifier

from pathlib import Path
import os
from tqdm import tqdm


if __name__ == "__main__":
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    MODEL_PATH = os.path.join(Path(__file__).parent, "runs", "model.pth")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MNIST(os.path.join(
        str(Path(__file__).parent), "data", "train.csv"),
        transforms=img_transforms)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = CNNClassifier(in_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    correct_predictions = 0
    total_predictions = 0
    for batch in tqdm(iter(dataloader)):
        imgs, labels = batch
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        predictions = model(imgs)

        predictions = torch.argmax(predictions, axis=1)

        correct_predictions += torch.sum(predictions == labels)
        total_predictions += len(batch)

    accuracy = correct_predictions / total_predictions
    print(f"ACCURACY: {accuracy}")
