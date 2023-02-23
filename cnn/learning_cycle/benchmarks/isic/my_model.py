
import timm 
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

_NORM_AND_SIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [224, 224]]

def set_model (model_name, num_class):

    model = timm.create_model(model_name, pretrained=True, num_classes=num_class)

    return model


