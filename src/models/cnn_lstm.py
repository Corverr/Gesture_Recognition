import torch.nn as nn
from torchvision.models import resnet18
from src.models.base_model import BaseVideoModel


class CNNLSTM(BaseVideoModel):

    def __init__(self, num_classes, num_frames=16, hidden_size=256, num_layers=2):

        super(CNNLSTM, self).__init__(num_classes, num_frames)
        self.name = "CNN+LSTM"

        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        batch_size, channels, frames, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * frames, channels, h, w)

        features = self.cnn(x)

        features = features.view(batch_size, frames, -1)

        lstm_out, _ = self.lstm(features)

        last_out = lstm_out[:, -1, :]

        logits = self.classifier(last_out)

        return logits