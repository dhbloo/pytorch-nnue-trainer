from utils.misc_utils import Register, import_submodules

MODELS = Register("model")
import_submodules(__name__, recursive=False)


def build_model(model_type, **kwargs):
    assert model_type in MODELS, f"Unknown model type: {model_type}"
    return MODELS[model_type](**kwargs)
