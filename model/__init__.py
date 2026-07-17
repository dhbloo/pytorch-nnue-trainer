"""Model registry and the model output convention.

Registered models take a ``data`` dict as input and return a ``dict[str, Tensor | dict]``.
Standard keys for the value-policy supervised paradigm:

    value       Tensor            value head output (required)
    policy      Tensor            policy head output (required)
    board_mask  Tensor, optional  valid-cell mask for variable board sizes
    aux_losses  dict, optional    {name: Tensor | (predefined_loss_type, inputs)},
                                  consumed by the loss orchestrator
    aux_outputs dict, optional    {name: Tensor}, logged only

Rules:
  * Omit absent keys entirely; never put ``None`` values in the output.
  * The key set must be constant per model instance (decided in ``__init__``,
    never by per-batch data) — unstable key sets force torch.compile to
    re-specialize one graph per key set.
  * Future training paradigms define their own keys; loss functions index the
    keys they require (missing keys fail fast with KeyError) and use ``.get()``
    for optional ones.
  * ``forward_debug_print`` mirrors ``forward``'s return convention.
"""

from torch.nn import Module
from utils.misc_utils import Registry, import_submodules

MODELS = Registry("model")
import_submodules(__name__, recursive=False)


def build_model(model_type, **kwargs) -> Module:
    assert model_type in MODELS, f"Unknown model type: {model_type}"
    return MODELS[model_type](**kwargs)
