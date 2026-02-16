"""
Pipe flow analysis for high-pressure compressible gas.

Physics:
  - Darcy-Weisbach: ΔP = f · (L/D) · (ρv²/2)
  - Colebrook-White: implicit friction factor for turbulent flow
  - Compressibility: density varies with pressure along the pipe,
    solved iteratively with average density approximation.

Limitations:
  - Isothermal assumption (no heat transfer modeling)
  - Average density approach breaks down for Mach > 0.3
    (Fanno flow model would be more appropriate there)
  - No minor losses (fittings, valves, bends) — add separately
"""

from dataclasses import dataclass
import numpy as np
from .gas import GasProperties, AIR


# Pipe roughness values [m]
ROUGHNESS = {
    "commercial_steel": 0.045e-3,
    "drawn_tubing":     0.0015e-3,
    "stainless_steel":  0.015e-3,
    "galvanized_steel": 0.15e-3,
    "cast_iron":        0.26e-3,
    "concrete":         1.0e-3,
    "smooth":           0.0,
}


@dataclass
class PipeGeometry:
    """Physical pipe dimensions and material."""
    length: float           # [m]
    inner_diameter: float   # [m]
    roughness: float = 0.045e-3  # default: commercial steel [m]

    @property
    def area(self) -> float:
        return np.pi * (self.inner_diameter / 2.0)**2

    @property
    def relative_roughness(self) -> float:
        return self.roughness / self.inner_diameter


@dataclass
class PipeFlowResult:
    """Complete pipe flow analysis output."""
    dp: float           # pressure drop [Pa]
    p_out: float        # outlet pressure [Pa]
    velocity: float     # bulk velocity [m/s]
    reynolds: float
    friction_factor: float
    mach: float         # flow Mach number in pipe
    density_avg: float  # average density [kg/m³]
    regime: str         # "laminar", "transitional", "turbulent"
    loss_fraction: float  # dp / p_in


class PipeFlow:
    """
    Pressure loss calculator for compressible pipe flow.

    The core iteration loop:
      1. Guess outlet pressure (start with p_out = p_in)
      2. Compute average density from average pressure
      3. Get velocity from continuity (mdot = rho·A·v)
      4. Compute Reynolds → friction factor → pressure drop
      5. Update outlet pressure, repeat until converged

    This is the standard industrial approach for moderate Mach
    numbers (M < 0.3). Beyond that, use Fanno flow tables.
    """

    def __init__(self, gas: GasProperties = AIR):
        self.gas = gas

    @staticmethod
    def colebrook(Re: float, eps_over_D: float,
                  tol: float = 1e-8, max_iter: int = 50) -> float:
        """
        Solve Colebrook-White equation iteratively:

        1/√f = -2·log₁₀(ε/(3.7D) + 2.51/(Re·√f))

        Initial guess: Swamee-Jain explicit approximation.
        """
        if Re < 2300:
            return 64.0 / Re

        a = eps_over_D / 3.7
        # Swamee-Jain starting point
        f = 0.25 / (np.log10(a + 5.74 / Re**0.9))**2

        for _ in range(max_iter):
            rhs = -2.0 * np.log10(a + 2.51 / (Re * np.sqrt(f)))
            f_new = 1.0 / rhs**2
            if abs(f_new - f) < tol:
                return f_new
            f = f_new

        return f  # return best estimate even if not fully converged

    def analyze(self, pipe: PipeGeometry, mdot: float,
                p_in: float, T: float) -> PipeFlowResult:
        """
        Compute pressure drop for given mass flow and inlet conditions.

        Parameters:
            pipe: PipeGeometry instance
            mdot: mass flow rate [kg/s]
            p_in: inlet pressure [Pa]
            T: gas temperature [K] (isothermal assumption)
        """
        D = pipe.inner_diameter
        A = pipe.area
        eps_D = pipe.relative_roughness

        p_out = p_in  # initial guess: no loss

        for _ in range(50):
            p_avg = (p_in + p_out) / 2.0
            rho = self.gas.density(p_avg, T)

            v = mdot / (rho * A)
            mu = self.gas.viscosity(T)
            Re = rho * v * D / mu

            if Re < 1.0:
                # negligible flow
                return PipeFlowResult(
                    dp=0, p_out=p_in, velocity=0, reynolds=0,
                    friction_factor=0, mach=0, density_avg=rho,
                    regime="no flow", loss_fraction=0,
                )

            f = self.colebrook(Re, eps_D)

            dp = f * (pipe.length / D) * (rho * v**2 / 2.0)
            p_out_new = p_in - dp

            if p_out_new <= 0:
                p_out_new = p_in * 0.01  # pipe way too restrictive

            if abs(p_out_new - p_out) < 1.0:  # converged within 1 Pa
                break

            p_out = p_out_new

        a_sound = self.gas.sound_speed(T)
        mach = v / a_sound

        if Re < 2300:
            regime = "laminar"
        elif Re < 4000:
            regime = "transitional"
        else:
            regime = "turbulent"

        return PipeFlowResult(
            dp=dp,
            p_out=p_out_new,
            velocity=v,
            reynolds=Re,
            friction_factor=f,
            mach=mach,
            density_avg=rho,
            regime=regime,
            loss_fraction=dp / p_in,
        )

    def sweep_diameter(self, length: float, roughness: float,
                       mdot: float, p_in: float, T: float,
                       d_range: tuple[float, float] = (3e-3, 25e-3),
                       n_points: int = 80) -> tuple[np.ndarray, np.ndarray]:
        """
        Pressure loss as a function of pipe diameter.
        Useful for selecting the right pipe size.
        """
        diameters = np.linspace(d_range[0], d_range[1], n_points)
        losses = np.zeros(n_points)

        for i, d in enumerate(diameters):
            pipe = PipeGeometry(length=length, inner_diameter=d, roughness=roughness)
            result = self.analyze(pipe, mdot, p_in, T)
            losses[i] = result.dp

        return diameters, losses
