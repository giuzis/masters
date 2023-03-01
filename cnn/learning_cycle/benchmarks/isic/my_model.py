
import timm 
from timm.data import resolve_data_config
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

_NORM_AND_SIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [224, 224]]

def set_model (model_name, num_class, input_size=224):

    model = None

    if "efficientnet" in model_name:
        model_name = "tf_" + model_name
        model = timm.create_model(model_name, pretrained=True, num_classes=num_class, in_chans=3)
        config = resolve_data_config({}, model=model)
        config['input_size'] = (3, input_size, input_size)
        model = timm.create_model(model_name, pretrained=True, num_classes=num_class, in_chans=3, **config)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_class)


    return model


