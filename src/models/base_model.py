import torch.nn as nn


class BaseVideoModel(nn.Module):

    def __init__(self, num_classes, num_frames=16):

        super(BaseVideoModel, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.name = "BaseModel"  # Будет переопределено в дочерних классах

    def forward(self, x):

        raise NotImplementedError("Каждая модель должна реализовать метод forward()")

    def get_name(self):
        return self.name

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())