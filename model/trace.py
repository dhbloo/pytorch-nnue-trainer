"""Eager-only tensor tracing for model diagnostics."""

from contextlib import contextmanager
from threading import Lock
from types import MethodType
from weakref import WeakSet

import torch.nn as nn


_active_trace_modules = WeakSet()
_active_trace_modules_lock = Lock()


def print_tensor_trace(name, tensor):
    """Print one observed tensor without affecting the real forward."""
    print(f"{name}:\n{tensor.detach().cpu()}")


class TraceableModule:
    def _trace(self, name, tensor):
        pass


class TraceableModel(TraceableModule, nn.Module):
    def forward_debug_print(self, data):
        with model_trace(self, print_tensor_trace):
            return self(data)


@contextmanager
def model_trace(model, callback):
    """Temporarily route trace events from one model to ``callback``."""
    modules = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(module, TraceableModule)
    ]

    # The lock protects admission only; distinct model traces execute
    # independently and concurrently after this short critical section.
    with _active_trace_modules_lock:
        if any(module in _active_trace_modules for _, module in modules):
            raise RuntimeError("model instance is already being traced")
        _active_trace_modules.update(module for _, module in modules)

    previous = []
    try:
        for path, module in modules:
            had_override = "_trace" in module.__dict__
            old_override = module.__dict__.get("_trace")

            def emit(self, name, tensor, *, _path=path):
                qualified_name = f"{_path}.{name}" if _path else name
                callback(qualified_name, tensor)

            previous.append((module, had_override, old_override))
            module.__dict__["_trace"] = MethodType(emit, module)
        yield model
    finally:
        for module, had_override, old_override in reversed(previous):
            if had_override:
                module.__dict__["_trace"] = old_override
            else:
                module.__dict__.pop("_trace", None)
        with _active_trace_modules_lock:
            for _, module in modules:
                _active_trace_modules.discard(module)
