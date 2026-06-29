import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from src.models.base_model import BaseVideoModel


class TemporalShiftModule(nn.Module):

    def __init__(self, n=8):
        super(TemporalShiftModule, self).__init__()
        self.n = n

    def forward(self, x):
        batch_size, channels, frames, h, w = x.shape
        fold = max(1, channels // self.n)

        out = torch.zeros_like(x)

        out[:, :fold, 1:, :, :] = x[:, :fold, :-1, :, :]
        out[:, :fold, 0, :, :] = 0

        out[:, fold:channels - fold, :, :, :] = x[:, fold:channels - fold, :, :, :]

        out[:, channels - fold:, :-1, :, :] = x[:, channels - fold:, 1:, :, :]
        out[:, channels - fold:, -1, :, :] = 0

        return out


class TSMResNet(BaseVideoModel):

    def __init__(self, num_classes, num_frames=16, model_name='resnet18',
                 shift_div=8, shift_place='block'):
        super(TSMResNet, self).__init__(num_classes, num_frames)
        self.name = f"TSM_{model_name}"
        self.shift_div = shift_div
        self.shift_place = shift_place

        if model_name == 'resnet18':
            self.base_model = resnet18(pretrained=True)
            self.feat_dim = 512
        elif model_name == 'resnet50':
            self.base_model = resnet50(pretrained=True)
            self.feat_dim = 2048
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool

        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4

        self._insert_shift_modules()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def _insert_shift_modules(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if self.shift_place == 'block':
                    block.conv1 = nn.Sequential(
                        TemporalShiftModule(self.shift_div),
                        block.conv1
                    )
                elif self.shift_place == 'residual':
                    if block.downsample is not None:
                        block.downsample = nn.Sequential(
                            TemporalShiftModule(self.shift_div),
                            block.downsample
                        )

    @staticmethod
    def _prepare_temporal_data(x):
        batch_size, channels, frames, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * frames, channels, h, w)
        return x, batch_size, frames

    def forward(self, x):
        batch_size, channels, frames, h, w = x.shape

        x, _, _ = self._prepare_temporal_data(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(batch_size, frames, -1, h // 4, w // 4)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self._forward_block_with_time(self.layer1, x)
        x = self._forward_block_with_time(self.layer2, x)
        x = self._forward_block_with_time(self.layer3, x)
        x = self._forward_block_with_time(self.layer4, x)

        x = x.mean(dim=2)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        logits = self.classifier(x)

        return logits

    @staticmethod
    def _forward_block_with_time(block, x):
        batch_size, channels, frames, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * frames, channels, h, w)

        x = block(x)

        _, channels, h, w = x.shape
        x = x.view(batch_size, frames, channels, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x


def TSM(num_classes, num_frames=16):
    return TSMResNet(num_classes, num_frames, model_name='resnet18')