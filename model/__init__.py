from torch.nn import Module
from utils.misc_utils import Registry, import_submodules

MODELS = Registry("model")
import_submodules(__name__, recursive=False)


def build_model(model_type, **kwargs) -> Module:
    assert model_type in MODELS, f"Unknown model type: {model_type}"
    return MODELS[model_type](**kwargs)
