from .resnet import ResNet
from .mobilenet import MobileNetV1, MobileNetV2
from .mix6 import Mix6Net

MODELS = {
    'resnet': ResNet,
    'mobilenetv1': MobileNetV1,
    'mobilenetv2': MobileNetV2,
    'mix6': Mix6Net,
}


def build_model(model_type, **kwargs):
    return MODELS[model_type](**kwargs)
