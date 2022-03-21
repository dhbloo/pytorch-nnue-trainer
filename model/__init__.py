from .resnet import ResNet
from .mobilenet import MobileNetV1, MobileNetV2
from .mix6 import Mix6Net
from .patnet import PatNetBaseline, PatNetv1, PatNetv2, PatNNUEv1
from .linear import LinearModel

MODELS = {
    'resnet': ResNet,
    'mobilenetv1': MobileNetV1,
    'mobilenetv2': MobileNetV2,
    'mix6': Mix6Net,
    'patnetbaseline': PatNetBaseline,
    'patnetv1': PatNetv1,
    'patnetv2': PatNetv2,
    'patnnuev1': PatNNUEv1,
    'linear': LinearModel,
}


def build_model(model_type, **kwargs):
    return MODELS[model_type](**kwargs)
