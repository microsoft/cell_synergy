import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class ResNet50Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

    def forward(self, x_tuple):
        x, _ = x_tuple if isinstance(x_tuple, tuple) else (x_tuple, None)
        return self.backbone(x)

    @property
    def transform(self):
        """Standard image transformation for UNI model."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
