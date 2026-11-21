"""Physical constants and material properties (SI units)."""

import math

MU0 = 4e-7 * math.pi  # vacuum permeability, H/m (classical value; CODATA differs by ~1e-10)
EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m
K_BOLTZMANN = 1.380649e-23  # J/K

# Annealed copper
CU_RESISTIVITY_20C = 1.724e-8  # ohm*m at 20 C
CU_TEMP_COEFF = 0.00393  # 1/K near 20 C
CU_DENSITY = 8960.0  # kg/m^3

# Standard PCB copper weights: 1 oz/ft^2 ~ 35 um
OZ_COPPER_THICKNESS = 35e-6  # m


def copper_resistivity(temp_c: float = 20.0) -> float:
    """Copper resistivity at ``temp_c`` degrees Celsius (linear model)."""
    return CU_RESISTIVITY_20C * (1.0 + CU_TEMP_COEFF * (temp_c - 20.0))


def awg_diameter(gauge: int) -> float:
    """Bare conductor diameter in metres for an American Wire Gauge number."""
    return 0.127e-3 * 92.0 ** ((36 - gauge) / 39.0)
