
import timm 
from timm.data import resolve_data_config
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch import nn

_NORM_AND_SIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [224, 224]]

def set_model (model_name, num_class, input_size=224):

    model = timm.create_model(model_name, pretrained=True, num_classes=num_class)

    # dropout_prob = 0.5 # probabilidade de dropout

    # if model_name.startswith('efficientnet'):
    #     # encontre a última camada linear do modelo
    #     num_ftrs = model.classifier.in_features 

    #     # adicione o dropout à camada linear e crie uma nova camada linear com o mesmo número de saídas
    #     model.classifier = nn.Sequential(
    #         nn.Dropout(p=dropout_prob),
    #         nn.Linear(num_ftrs, model.num_classes)
    #     )

    # else:
    #     # encontre a última camada linear do modelo
    #     num_ftrs = model.fc.in_features 

    #     # adicione o dropout à camada linear e crie uma nova camada linear com o mesmo número de saídas
    #     model.fc = nn.Sequential(
    #         nn.Dropout(p=dropout_prob),
    #         nn.Linear(num_ftrs, model.num_classes)
    #     )
        

    return model


