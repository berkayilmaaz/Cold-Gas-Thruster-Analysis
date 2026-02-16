"""
Isentropic (adiabatic + reversible) flow relations.

All relations derive from combining:
  - Conservation of energy (1st law, adiabatic: h + v²/2 = h0)
  - Ideal gas equation of state (P = rho·R·T)
  - Isentropic process (P/rho^gamma = const)

Reference: Anderson, "Modern Compressible Flow", Ch. 3
"""

import numpy as np
from scipy.optimize import brentq
from .gas import GasProperties


class IsentropicFlow:
    """
    Compressible isentropic flow calculator for a given gas.

    All "ratio" methods return the stagnation-to-static ratio
    as a function of Mach number.
    """

    def __init__(self, gas: GasProperties):
        self.gas = gas
        self.g = gas.gamma

    def _gm1_half(self) -> float:
        """(gamma - 1) / 2, appears everywhere in compressible flow"""
        return (self.g - 1.0) / 2.0

    # ── Stagnation-to-static ratios ──────────────────────────────

    def temperature_ratio(self, M: float) -> float:
        """T0/T = 1 + (γ-1)/2 · M²"""
        return 1.0 + self._gm1_half() * M**2

    def pressure_ratio(self, M: float) -> float:
        """P0/P = [1 + (γ-1)/2 · M²]^(γ/(γ-1))"""
        return self.temperature_ratio(M) ** (self.g / (self.g - 1.0))

    def density_ratio(self, M: float) -> float:
        """rho0/rho = [1 + (γ-1)/2 · M²]^(1/(γ-1))"""
        return self.temperature_ratio(M) ** (1.0 / (self.g - 1.0))

    # ── Critical (sonic, M=1) ratios ─────────────────────────────

    def critical_pressure_ratio(self) -> float:
        """
        P*/P0 — pressure at the throat when flow is choked.
        For air: (2/2.4)^3.5 ≈ 0.5283
        """
        return (2.0 / (self.g + 1.0)) ** (self.g / (self.g - 1.0))

    def critical_temperature_ratio(self) -> float:
        """T*/T0 = 2/(γ+1). For air: 0.8333"""
        return 2.0 / (self.g + 1.0)

    # ── Inverse: Mach from ratios ────────────────────────────────

    def mach_from_pressure_ratio(self, p0: float, p: float) -> float:
        """
        Given P0 and P (static), find Mach number.

        M = sqrt{ 2/(γ-1) · [(P0/P)^((γ-1)/γ) - 1] }
        """
        ratio = p0 / p
        exponent = (self.g - 1.0) / self.g
        return np.sqrt(2.0 / (self.g - 1.0) * (ratio**exponent - 1.0))

    def mach_from_area_ratio(self, area_ratio: float, supersonic: bool = True) -> float:
        """
        Invert the area-Mach relation numerically.

        The A/A* equation has two solutions for every area_ratio > 1:
        one subsonic, one supersonic. The `supersonic` flag picks which.

        Uses Brent's method on [1+eps, 30] for supersonic,
        [0.01, 1-eps] for subsonic.
        """
        def residual(M):
            return self.area_ratio(M) - area_ratio

        if supersonic:
            return brentq(residual, 1.001, 50.0)
        else:
            return brentq(residual, 0.01, 0.999)

    # ── Area-Mach relation ───────────────────────────────────────

    def area_ratio(self, M: float) -> float:
        """
        A/A* as a function of Mach.

        A/A* = (1/M) · [ 2/(γ+1) · (1 + (γ-1)/2 · M²) ]^((γ+1)/(2(γ-1)))
        """
        term = (2.0 / (self.g + 1.0)) * (1.0 + self._gm1_half() * M**2)
        exp = (self.g + 1.0) / (2.0 * (self.g - 1.0))
        return (1.0 / M) * term**exp

    # ── Mass flow parameter ──────────────────────────────────────

    def choked_mdot_per_area(self, p0: float, T0: float) -> float:
        """
        Maximum mass flux through a choked throat [kg/(s·m²)].

        ṁ/A* = P0 · sqrt(γ/(R·T0)) · [2/(γ+1)]^((γ+1)/(2(γ-1)))

        This is the fundamental constraint: once the throat is choked,
        increasing downstream pressure won't increase mass flow.
        Only raising P0 or lowering T0 helps.
        """
        coeff = np.sqrt(self.g / (self.gas.R * T0))
        exp = (self.g + 1.0) / (2.0 * (self.g - 1.0))
        factor = (2.0 / (self.g + 1.0))**exp
        return p0 * coeff * factor

    # ── Static conditions at a given Mach ────────────────────────

    def static_temperature(self, T0: float, M: float) -> float:
        return T0 / self.temperature_ratio(M)

    def static_pressure(self, p0: float, M: float) -> float:
        return p0 / self.pressure_ratio(M)

    def velocity(self, T0: float, M: float) -> float:
        """Flow velocity at a given Mach: v = M · a = M · sqrt(γ·R·T)"""
        T = self.static_temperature(T0, M)
        return M * self.gas.sound_speed(T)
