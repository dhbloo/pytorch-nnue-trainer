"""Low-overhead exponential moving averages for model parameters."""

from collections import defaultdict

import torch


def _shadow_dtype(parameter: torch.Tensor):
    if parameter.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return parameter.dtype


class ModelEMA:
    """Keep an EMA shadow of a model's trainable floating-point parameters.

    The shadow stays on the model device and updates through one foreach call
    per device/dtype group. Buffers are intentionally read from the live model
    only when a checkpoint is built: EMA is a weight average, while running
    statistics, internally updated codebooks, and other persistent state retain
    their normal checkpoint value.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        state_dict: dict[str, torch.Tensor] | None = None,
    ):
        live_state = model.state_dict()
        parameters_by_name = dict(
            model.named_parameters(remove_duplicate=False)
        )
        if state_dict is not None:
            if list(state_dict) != list(live_state):
                raise RuntimeError("EMA checkpoint keys differ from the model")
            for name, value in state_dict.items():
                reference = live_state[name]
                parameter = parameters_by_name.get(name)
                expected_dtype = (
                    _shadow_dtype(parameter)
                    if parameter is not None and parameter.requires_grad
                    else reference.dtype
                )
                if value.shape != reference.shape or value.dtype != expected_dtype:
                    raise RuntimeError(
                        f"EMA checkpoint tensor {name!r} differs from the model"
                    )

        shadows_by_parameter = {}
        self._shadows_by_name = {}
        groups = defaultdict(lambda: ([], []))
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if not parameter.requires_grad or not (
                parameter.is_floating_point() or parameter.is_complex()
            ):
                continue
            identity = id(parameter)
            shadow = shadows_by_parameter.get(identity)
            if shadow is None:
                source = parameter if state_dict is None else state_dict[name]
                shadow = source.detach().to(
                    device=parameter.device,
                    dtype=_shadow_dtype(parameter),
                    copy=True,
                )
                shadows_by_parameter[identity] = shadow
                shadows, parameters = groups[
                    (parameter.device, shadow.dtype, parameter.dtype)
                ]
                shadows.append(shadow)
                parameters.append(parameter)
            self._shadows_by_name[name] = shadow

        self.decay = decay
        self._groups = tuple(groups.values())

    @torch.no_grad()
    def update(self):
        """Blend the current live parameters into the shadow in place."""
        weight = 1.0 - self.decay
        for shadows, parameters in self._groups:
            if shadows[0].dtype == parameters[0].dtype:
                torch._foreach_lerp_(shadows, parameters, weight)
            else:
                torch._foreach_mul_(shadows, self.decay)
                torch._foreach_add_(shadows, parameters, alpha=weight)

    def state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Return a loadable model state with EMA parameters and live buffers."""
        state = model.state_dict()
        for name, shadow in self._shadows_by_name.items():
            state[name] = shadow
        return state
