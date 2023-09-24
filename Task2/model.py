import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(CNNClassifier, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=512, kernel_size=3, padding=1)

        # Fully connected layers
        # 7x7 image size after pooling
        self.fc1 = nn.Linear(in_features=512*7*7, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        # 10 classes for MNIST
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        batch_size = x.shape[0]
        # Apply first convolutional layer, ReLU, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolutional layer, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU
        x = F.relu(self.fc2(x))  # Apply second fully connected layer and ReLu
        x = self.fc3(x)  # Apply last connected layer
        return x


if __name__ == "__main__":
    # Create an instance of the CNN model
    model = CNNClassifier()

    img = torch.randn((1, 1, 28, 28))
    res = model(img)
    print(res.shape)
    print(res)
