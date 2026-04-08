"""MagZero package."""

__all__ = ["Magzero", "__version__"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "Magzero":
        try:
            from .model import Magzero as _Magzero
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Magzero requires optional runtime dependencies such as 'nearedge'. "
                "Install the research dependencies before importing the model."
            ) from exc
        return _Magzero
    raise AttributeError(f"module 'magzero' has no attribute {name!r}")
