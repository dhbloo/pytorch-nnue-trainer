"""Helpers for applying model-scoped compiler configuration."""


def model_inductor_config(model) -> dict:
    """Return a detached copy of the model's optional Inductor hints."""
    return dict(getattr(model, "inductor_config", {}))


def with_inductor_options(compile_kwargs: dict, settings: dict | None) -> dict:
    """Merge model defaults into lazy-safe ``torch.compile`` options.

    PyTorch does not allow ``mode`` and ``options`` together, so an active mode
    is expanded to its option preset first.  Explicit options supplied by the
    caller win over model defaults, which in turn win over the mode preset.
    Hints are ignored for non-Inductor backends, where their names are invalid.
    """
    kwargs = dict(compile_kwargs)
    if not settings or kwargs.get("backend", "inductor") != "inductor":
        return kwargs

    mode = kwargs.pop("mode", None)
    if mode and mode != "default":
        from torch._inductor import list_mode_options

        options = dict(list_mode_options(mode, kwargs.get("dynamic")))
    else:
        options = {}
    options.update(settings)
    options.update(kwargs.get("options") or {})
    kwargs["options"] = options
    return kwargs
