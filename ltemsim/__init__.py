"""ltemsim — steady-state drift-diffusion + EL2 trap model of a pumped, biased SI-GaAs gap
and the near-field (tip) THz emission signal."""
from .params import Params, DEFAULT, SOURCES
from .grid import Grid
from .physics import Material, Scale
from .solver import Simulation, Solution

__all__ = ["Params", "DEFAULT", "SOURCES", "Grid", "Material", "Scale", "Simulation", "Solution"]
__version__ = "0.1.0"
