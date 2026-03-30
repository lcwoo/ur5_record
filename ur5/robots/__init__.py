"""Robot interfaces and bridges."""

__all__ = ["UR5RTDEBridge"]


def __getattr__(name):
    """Lazily import heavy modules to avoid runpy re-import warnings."""
    if name == "UR5RTDEBridge":
        from ur5.robots.ur5_bridge import UR5RTDEBridge
        return UR5RTDEBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
