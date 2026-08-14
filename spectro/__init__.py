__version__ = "2.0.0"

__all__ = ["main", "__version__"]


def __getattr__(name):
    # keep `import spectro` cheap, cli drags in numpy and friends
    if name == "main":
        from .cli import main
        return main
    raise AttributeError(name)
