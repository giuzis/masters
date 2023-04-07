
import timm 
from timm.data import resolve_data_config
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch import nn
from torchinfo import summary

def set_model (model_name, num_class, dropout_prob = 0.0, train_classifier_only = False):

    model = timm.create_model(model_name, pretrained=True)

    if train_classifier_only:
        print('Freezing all layers except classifier')
        for param in model.parameters():
            param.requires_grad = False

    if 'efficientnet' in model_name or 'densenet121' in model_name:
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(model.classifier.in_features, num_class)
        )

    elif 'resnest101e' in model_name or 'seresnext101_32x8d' in model_name or 'resnext101_32x8d' in model_name or 'resnest50d' in model_name:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(model.fc.in_features, num_class)
        )

    elif 'vgg19' in model_name:
        model.head.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(model.head.fc.in_features, num_class)
        )

    elif 'pnasnet5large' in model_name:
        model.last_linear = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(model.last_linear.in_features, num_class)
        )
        
    summary(model, input_size=(1, 3, 224, 224), verbose=0, col_names=['input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds', 'trainable'], row_settings=["var_names"])
    
    return model


