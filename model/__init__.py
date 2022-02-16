from .resnet import ResNet

MODELS = {'resnet': ResNet}


def build_model(model_type, **kwargs):
    return MODELS[model_type](**kwargs)
