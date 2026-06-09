__version__ = "2.0.0"

from .cli import main

main_wrapper = main

__all__ = ["main", "main_wrapper", "__version__"]
