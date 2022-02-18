from .resnet import ResNet
from .mix6 import Mix6Net

MODELS = {
    'resnet': ResNet,
    'mix6': Mix6Net,
}


def build_model(model_type, **kwargs):
    return MODELS[model_type](**kwargs)
