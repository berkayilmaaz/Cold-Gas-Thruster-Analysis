"""
Gas properties for cold gas thruster analysis.

Ideal gas model with thermodynamic properties derived from
molecular structure (diatomic for air/N2, monatomic for He/Ar).
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class GasProperties:
    """
    Thermodynamic properties of an ideal gas.

    For a diatomic ideal gas (air, N2):
        gamma = 7/5 = 1.4
        R = R_universal / M_molar

    Frozen dataclass — immutable after creation, which is
    physically correct: gas properties don't change mid-calculation.
    """
    name: str
    gamma: float          # cp/cv, dimensionless
    R: float              # specific gas constant [J/(kg·K)]
    molar_mass: float     # [kg/mol]
    mu_ref: float         # reference dynamic viscosity [Pa·s] at T_ref
    T_ref: float = 300.0  # reference temperature for viscosity [K]

    @property
    def cp(self) -> float:
        """Specific heat at constant pressure [J/(kg·K)]"""
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        """Specific heat at constant volume [J/(kg·K)]"""
        return self.R / (self.gamma - 1.0)

    def density(self, p: float, T: float) -> float:
        """Ideal gas: rho = P / (R·T)"""
        return p / (self.R * T)

    def sound_speed(self, T: float) -> float:
        """a = sqrt(gamma · R · T)"""
        return np.sqrt(self.gamma * self.R * T)

    def viscosity(self, T: float) -> float:
        """
        Sutherland's law approximation for viscosity scaling.
        mu(T) = mu_ref · (T/T_ref)^(3/2) · (T_ref + S) / (T + S)

        For air, S ≈ 110.4 K.
        Falls back to power-law if you don't want the complexity.
        """
        S = 110.4  # Sutherland constant for air [K]
        return self.mu_ref * (T / self.T_ref)**1.5 * (self.T_ref + S) / (T + S)


# Pre-defined gases commonly used in cold gas thrusters
AIR = GasProperties(
    name="Air",
    gamma=1.4,
    R=287.05,          # R_universal / 28.97e-3
    molar_mass=28.97e-3,
    mu_ref=1.81e-5,
)

NITROGEN = GasProperties(
    name="N₂",
    gamma=1.4,
    R=296.8,
    molar_mass=28.01e-3,
    mu_ref=1.76e-5,
)

HELIUM = GasProperties(
    name="He",
    gamma=5.0 / 3.0,   # monatomic
    R=2077.0,
    molar_mass=4.003e-3,
    mu_ref=1.96e-5,
)

CO2 = GasProperties(
    name="CO₂",
    gamma=1.29,         # triatomic, lower gamma
    R=188.9,
    molar_mass=44.01e-3,
    mu_ref=1.47e-5,
)
