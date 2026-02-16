"""
Converging-diverging (CD) nozzle design and performance analysis.

Design approach:
  1. Specify desired thrust + operating conditions
  2. Isentropic relations give ideal throat/exit sizing
  3. Discharge coefficient (Cd) accounts for viscous losses

The nozzle is the core component — everything else in the
pneumatic system exists to feed it the right conditions.
"""

from dataclasses import dataclass, field
import numpy as np
from .gas import GasProperties, AIR
from .isentropic import IsentropicFlow

G0 = 9.80665  # [m/s²]


@dataclass
class NozzleGeometry:
    """Physical dimensions of a CD nozzle."""
    d_throat: float   # throat diameter [m]
    d_exit: float     # exit diameter [m]
    d_inlet: float    # inlet (convergent entry) diameter [m]
    half_angle_conv: float = 30.0   # convergent half-angle [deg]
    half_angle_div: float = 15.0    # divergent half-angle [deg]

    @property
    def a_throat(self) -> float:
        return np.pi * (self.d_throat / 2.0)**2

    @property
    def a_exit(self) -> float:
        return np.pi * (self.d_exit / 2.0)**2

    @property
    def area_ratio(self) -> float:
        return self.a_exit / self.a_throat

    @property
    def length_convergent(self) -> float:
        """Axial length of converging section [m]"""
        r_diff = (self.d_inlet - self.d_throat) / 2.0
        return r_diff / np.tan(np.radians(self.half_angle_conv))

    @property
    def length_divergent(self) -> float:
        """Axial length of diverging section [m]"""
        r_diff = (self.d_exit - self.d_throat) / 2.0
        return r_diff / np.tan(np.radians(self.half_angle_div))

    def profile(self, n_points: int = 100):
        """
        Generate 2D meridional contour points.
        Returns (x, r_upper, r_lower) arrays in [m].

        Convergent: parabolic contraction (smooth acceleration)
        Divergent:  conical expansion (simple, predictable)
        """
        r_t = self.d_throat / 2.0
        r_e = self.d_exit / 2.0
        r_i = self.d_inlet / 2.0

        L_c = self.length_convergent
        L_d = self.length_divergent

        x_conv = np.linspace(-L_c, 0, n_points // 2)
        r_conv = r_t + (r_i - r_t) * (x_conv / (-L_c))**2

        x_div = np.linspace(0, L_d, n_points // 2)
        r_div = r_t + (r_e - r_t) * (x_div / L_d)

        x = np.concatenate([x_conv, x_div])
        r = np.concatenate([r_conv, r_div])
        return x, r, -r


@dataclass
class NozzlePerformance:
    """Results from nozzle analysis — what you actually care about."""
    thrust: float         # [N]
    mdot: float           # [kg/s]
    isp: float            # [s]
    exit_mach: float
    exit_velocity: float  # [m/s]
    exit_temperature: float  # [K]
    exit_pressure: float  # [Pa]
    geometry: NozzleGeometry
    is_fully_expanded: bool


class Nozzle:
    """
    CD nozzle designer and analyzer.

    Two modes of operation:
      - design(): "I need X Newtons, what throat do I need?"
      - analyze(): "I have this throat, what thrust do I get?"
    """

    def __init__(self, gas: GasProperties = AIR, Cd: float = 1.0):
        """
        Cd: discharge coefficient (0 < Cd <= 1).
        Accounts for boundary layer blockage at the throat.
        Ideal: 1.0. Typical cold gas: 0.90–0.97.
        Well-machined small nozzle: ~0.85–0.92.
        """
        self.gas = gas
        self.flow = IsentropicFlow(gas)
        self.Cd = Cd

    def is_choked(self, p0: float, pa: float) -> bool:
        """Check if the pressure ratio is sufficient for choked flow."""
        p_crit = p0 * self.flow.critical_pressure_ratio()
        return pa < p_crit

    def design(self, p0: float, T0: float, pa: float,
               target_thrust: float) -> NozzlePerformance:
        """
        Size a nozzle for a target thrust.

        Assumes fully-expanded operation (Pe = Pa) for max efficiency.
        This means the exit pressure matches ambient — no pressure thrust
        wasted, and no flow separation risk.

        Parameters:
            p0: stagnation (chamber) pressure [Pa]
            T0: stagnation temperature [K]
            pa: ambient pressure [Pa]
            target_thrust: desired thrust [N]
        """
        if not self.is_choked(p0, pa):
            raise ValueError(
                f"Pressure ratio P0/Pa = {p0/pa:.1f} is insufficient for "
                f"choked flow. Need P0/Pa > {1/self.flow.critical_pressure_ratio():.2f}"
            )

        # fully expanded: Pe = Pa
        Me = self.flow.mach_from_pressure_ratio(p0, pa)
        Te = self.flow.static_temperature(T0, Me)
        Ve = self.flow.velocity(T0, Me)

        # F = mdot·Ve  (pressure thrust = 0 when fully expanded)
        mdot_ideal = target_thrust / Ve
        mdot_real = mdot_ideal / self.Cd  # need more mass flow to compensate losses

        # throat sizing from choked mass flow
        mdot_per_area = self.flow.choked_mdot_per_area(p0, T0)
        a_throat = mdot_real / mdot_per_area
        d_throat = 2.0 * np.sqrt(a_throat / np.pi)

        # exit sizing from area-Mach relation
        ar = self.flow.area_ratio(Me)
        a_exit = ar * a_throat
        d_exit = 2.0 * np.sqrt(a_exit / np.pi)

        # inlet: typically 2-2.5x throat diameter
        d_inlet = d_throat * 2.2

        geom = NozzleGeometry(
            d_throat=d_throat,
            d_exit=d_exit,
            d_inlet=d_inlet,
        )

        return NozzlePerformance(
            thrust=target_thrust,
            mdot=mdot_real,
            isp=Ve / G0,
            exit_mach=Me,
            exit_velocity=Ve,
            exit_temperature=Te,
            exit_pressure=pa,
            geometry=geom,
            is_fully_expanded=True,
        )

    def analyze(self, geometry: NozzleGeometry, p0: float, T0: float,
                pa: float) -> NozzlePerformance:
        """
        Predict performance of an existing nozzle.

        Given fixed geometry, compute thrust at specified conditions.
        Handles both fully-expanded and off-design cases.
        """
        if not self.is_choked(p0, pa):
            raise ValueError("Flow not choked at these conditions.")

        a_throat = geometry.a_throat

        # mass flow through choked throat
        mdot = self.Cd * a_throat * self.flow.choked_mdot_per_area(p0, T0)

        # exit Mach from area ratio (supersonic solution)
        ar = geometry.area_ratio
        Me = self.flow.mach_from_area_ratio(ar, supersonic=True)

        # exit conditions
        Te = self.flow.static_temperature(T0, Me)
        Ve = self.flow.velocity(T0, Me)
        Pe = self.flow.static_pressure(p0, Me)

        # general thrust equation: F = mdot·Ve + (Pe - Pa)·Ae
        a_exit = geometry.a_exit
        thrust = mdot * Ve + (Pe - pa) * a_exit

        fully_expanded = abs(Pe - pa) / pa < 0.01

        return NozzlePerformance(
            thrust=thrust,
            mdot=mdot,
            isp=thrust / (mdot * G0),
            exit_mach=Me,
            exit_velocity=Ve,
            exit_temperature=Te,
            exit_pressure=Pe,
            geometry=geometry,
            is_fully_expanded=fully_expanded,
        )

    def thrust_curve(self, geometry: NozzleGeometry, T0: float, pa: float,
                     p_range: tuple[float, float] = (5e5, 350e5),
                     n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Thrust as a function of chamber pressure for a fixed nozzle.
        Returns (pressures, thrusts) arrays.
        """
        pressures = np.linspace(p_range[0], p_range[1], n_points)
        thrusts = np.zeros(n_points)

        for i, p0 in enumerate(pressures):
            try:
                perf = self.analyze(geometry, p0, T0, pa)
                thrusts[i] = perf.thrust
            except ValueError:
                thrusts[i] = 0.0

        return pressures, thrusts
