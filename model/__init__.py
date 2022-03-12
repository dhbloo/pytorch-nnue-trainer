from .resnet import ResNet
from .mobilenet import MobileNetV1, MobileNetV2
from .mix6 import Mix6Net
from .patnet import PatNetv0, PatNetv1, PatNNUEv1

MODELS = {
    'resnet': ResNet,
    'mobilenetv1': MobileNetV1,
    'mobilenetv2': MobileNetV2,
    'mix6': Mix6Net,
    'patnetv0': PatNetv0,
    'patnetv1': PatNetv1,
    'patnnuev1': PatNNUEv1,
}


def build_model(model_type, **kwargs):
    return MODELS[model_type](**kwargs)
