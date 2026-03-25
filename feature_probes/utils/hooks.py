"""Hook utilities for intercepting model activations."""

import contextlib
import functools
from typing import Callable, List, Tuple

import torch.nn as nn


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[nn.Module, Callable]],
    module_forward_hooks: List[Tuple[nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.
    
    Args:
        module_forward_pre_hooks: List of (module, hook_fn) pairs for pre-hooks
        module_forward_hooks: List of (module, hook_fn) pairs for forward hooks
        **kwargs: Additional arguments passed to hook functions
    """
    handles = []
    try:
        # Register pre-hooks
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        
        # Register forward hooks
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        
        yield
    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()