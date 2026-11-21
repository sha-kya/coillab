"""CoilLab: design and analysis of inductive (search-coil) magnetic sensors."""

from .coils import Amplifier, CircularCoil, Coil, CoilReport, PCBSquareSpiral, SquareCoil, eng

__version__ = "0.1.0"

__all__ = [
    "Amplifier",
    "CircularCoil",
    "Coil",
    "CoilReport",
    "PCBSquareSpiral",
    "SquareCoil",
    "eng",
    "__version__",
]
