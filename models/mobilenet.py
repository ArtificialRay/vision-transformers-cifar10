from torchvision import models
import torch

def get_mobilenet_model(pretrained=True, num_classes=1000):
    """Returns a MobileNetV2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes for the final classification layer

    Returns:
        model (torch.nn.Module): MobileNetV2 model
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Modify the classifier to match the number of classes
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    
    return model