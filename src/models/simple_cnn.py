import torch
import torch.nn as nn
from src.models.base_model import BaseVideoModel


class SimpleCNN(BaseVideoModel):
    def __init__(self, num_classes, num_frames=8):
        super(SimpleCNN, self).__init__(num_classes, num_frames)
        self.name = "SimpleCNN"
        self.num_frames = num_frames

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d((1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, channels, frames, height, width]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


def create_simple_cnn(num_classes, num_frames=8):
    return SimpleCNN(num_classes, num_frames)