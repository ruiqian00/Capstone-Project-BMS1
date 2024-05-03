

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import copy
from torch.optim import Adam
import timm  



def create_model(model_name, num_classes, use_pretrained=True):
    if model_name == 'resnet101':
        model = models.resnet101(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'swin_transformer':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=use_pretrained, num_classes=num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=use_pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=use_pretrained)
        # handle auxiliary output
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if model.AuxLogits is not None:
            num_aux_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_aux_ftrs, num_classes)
    else:
        raise ValueError('Invalid model name')
    return model

