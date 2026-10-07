__all__ = ["CLASS_NAMES", "FoldModel"]


def __getattr__(name):
    # Raw-log workers need the parser, not a separate PyTorch/CUDA runtime.
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
