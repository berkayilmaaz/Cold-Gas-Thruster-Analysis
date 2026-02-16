"""
Cold Gas Thruster system model.

Integrates nozzle, pipe, and gas modules into a complete
propulsion system. Handles the full chain:

  Tank (P_tank) → Regulator (P_reg) → Pipe → Nozzle → Atmosphere

This is the object you'd use in a flight simulation or
system-level trade study.
"""

from dataclasses import dataclass
import numpy as np
from .gas import GasProperties, AIR
from .nozzle import Nozzle, NozzleGeometry, NozzlePerformance
from .pipe import PipeFlow, PipeGeometry, PipeFlowResult
from .isentropic import IsentropicFlow


G0 = 9.80665


@dataclass
class ThrusterConfig:
    """
    Complete thruster system specification.

    Mirrors a real cold gas thruster setup:
    tank → regulator → feed line → nozzle
    """
    # operating conditions
    p_chamber: float      # nozzle inlet pressure [Pa] (after regulator)
    T0: float             # gas temperature [K]
    p_ambient: float      # ambient pressure [Pa]

    # nozzle
    Cd: float = 1.0       # nozzle discharge coefficient

    # feed line (optional — set to None to skip pipe analysis)
    pipe: PipeGeometry | None = None


@dataclass
class SystemAnalysis:
    """Full system analysis results."""
    nozzle: NozzlePerformance
    pipe: PipeFlowResult | None
    effective_chamber_pressure: float  # after pipe losses [Pa]
    thrust_with_losses: float          # thrust accounting for pipe loss [N]
    system_efficiency: float           # actual/ideal thrust ratio


class ColdGasThruster:
    """
    Top-level system model for a cold gas thruster.

    Usage:
        thruster = ColdGasThruster(gas=AIR)
        config = ThrusterConfig(p_chamber=40e5, T0=298, p_ambient=89875)

        # design mode: size a nozzle for target thrust
        result = thruster.design(config, target_thrust=83.0)

        # analysis mode: predict performance of existing hardware
        geom = NozzleGeometry(d_throat=0.0038, d_exit=0.0084, d_inlet=0.009)
        result = thruster.analyze(config, geom)
    """

    def __init__(self, gas: GasProperties = AIR):
        self.gas = gas
        self.nozzle_calc = Nozzle(gas)
        self.pipe_calc = PipeFlow(gas)
        self.isentropic = IsentropicFlow(gas)

    def design(self, config: ThrusterConfig,
               target_thrust: float) -> SystemAnalysis:
        """
        Design a nozzle and evaluate the complete system.

        If pipe geometry is specified, accounts for feed line losses
        by iterating: design nozzle → compute pipe loss → re-design
        with effective chamber pressure.
        """
        self.nozzle_calc.Cd = config.Cd
        p_eff = config.p_chamber

        pipe_result = None

        if config.pipe is not None:
            # first pass: estimate mdot without pipe loss
            nozzle_result = self.nozzle_calc.design(
                p_eff, config.T0, config.p_ambient, target_thrust
            )

            # compute pipe loss with that mdot
            pipe_result = self.pipe_calc.analyze(
                config.pipe, nozzle_result.mdot, config.p_chamber, config.T0
            )
            p_eff = pipe_result.p_out

            # second pass: redesign with effective pressure
            nozzle_result = self.nozzle_calc.design(
                p_eff, config.T0, config.p_ambient, target_thrust
            )

            # update pipe loss with corrected mdot
            pipe_result = self.pipe_calc.analyze(
                config.pipe, nozzle_result.mdot, config.p_chamber, config.T0
            )
            p_eff = pipe_result.p_out
        else:
            nozzle_result = self.nozzle_calc.design(
                p_eff, config.T0, config.p_ambient, target_thrust
            )

        # evaluate thrust at effective pressure
        nozzle_at_eff = self.nozzle_calc.design(
            p_eff, config.T0, config.p_ambient, target_thrust
        )

        # ideal thrust (no pipe loss, Cd=1)
        ideal_nozzle = Nozzle(self.gas, Cd=1.0)
        ideal_result = ideal_nozzle.design(
            config.p_chamber, config.T0, config.p_ambient, target_thrust
        )

        efficiency = nozzle_at_eff.thrust / ideal_result.thrust if ideal_result.thrust > 0 else 0

        return SystemAnalysis(
            nozzle=nozzle_result,
            pipe=pipe_result,
            effective_chamber_pressure=p_eff,
            thrust_with_losses=nozzle_at_eff.thrust,
            system_efficiency=efficiency,
        )

    def analyze(self, config: ThrusterConfig,
                geometry: NozzleGeometry) -> SystemAnalysis:
        """Predict performance of an existing nozzle + pipe system."""
        self.nozzle_calc.Cd = config.Cd
        p_eff = config.p_chamber

        pipe_result = None

        if config.pipe is not None:
            # estimate mdot from nozzle at nominal pressure
            nozzle_nominal = self.nozzle_calc.analyze(
                geometry, config.p_chamber, config.T0, config.p_ambient
            )

            pipe_result = self.pipe_calc.analyze(
                config.pipe, nozzle_nominal.mdot, config.p_chamber, config.T0
            )
            p_eff = pipe_result.p_out

        nozzle_result = self.nozzle_calc.analyze(
            geometry, p_eff, config.T0, config.p_ambient
        )

        # ideal comparison
        ideal_nozzle = Nozzle(self.gas, Cd=1.0)
        ideal_result = ideal_nozzle.analyze(
            geometry, config.p_chamber, config.T0, config.p_ambient
        )

        efficiency = nozzle_result.thrust / ideal_result.thrust if ideal_result.thrust > 0 else 0

        return SystemAnalysis(
            nozzle=nozzle_result,
            pipe=pipe_result,
            effective_chamber_pressure=p_eff,
            thrust_with_losses=nozzle_result.thrust,
            system_efficiency=efficiency,
        )

    def tank_blowdown(self, config: ThrusterConfig,
                      geometry: NozzleGeometry,
                      V_tank: float, p_tank_initial: float,
                      p_regulated: float | None = None,
                      dt: float = 0.01,
                      t_max: float = 30.0) -> dict:
        """
        Simulate tank blowdown over time.

        Two modes:
          - Regulated: constant chamber pressure until tank drops below regulator setpoint
          - Unregulated: chamber pressure = tank pressure (direct feed)

        Parameters:
            V_tank: tank volume [m³]
            p_tank_initial: initial tank pressure [Pa]
            p_regulated: regulator output pressure [Pa], None for blowdown
            dt: time step [s]
            t_max: maximum simulation time [s]

        Returns dict with time-series arrays.
        """
        self.nozzle_calc.Cd = config.Cd
        T0 = config.T0
        pa = config.p_ambient

        n_steps = int(t_max / dt) + 1
        t = np.zeros(n_steps)
        p_tank = np.zeros(n_steps)
        thrust = np.zeros(n_steps)
        mdot_arr = np.zeros(n_steps)
        mass_gas = np.zeros(n_steps)

        # initial gas mass in tank (ideal gas)
        rho_init = self.gas.density(p_tank_initial, T0)
        m_gas = rho_init * V_tank

        p_tank[0] = p_tank_initial
        mass_gas[0] = m_gas

        for i in range(1, n_steps):
            t[i] = i * dt

            # current tank pressure from remaining mass
            rho_current = m_gas / V_tank
            p_current = rho_current * self.gas.R * T0

            if p_current <= pa:
                # tank depleted
                p_tank[i:] = pa
                break

            p_tank[i] = p_current

            # chamber pressure: regulated or direct
            if p_regulated is not None:
                p_chamber = min(p_current, p_regulated)
            else:
                p_chamber = p_current

            if not self.nozzle_calc.is_choked(p_chamber, pa):
                break

            try:
                perf = self.nozzle_calc.analyze(geometry, p_chamber, T0, pa)
                thrust[i] = perf.thrust
                mdot_arr[i] = perf.mdot
            except ValueError:
                break

            # deplete gas
            m_gas -= perf.mdot * dt
            mass_gas[i] = max(m_gas, 0)

            if m_gas <= 0:
                break

        # trim to actual simulation length
        mask = t <= t[i] if i < n_steps - 1 else slice(None)

        return {
            "t": t[mask],
            "p_tank": p_tank[mask],
            "thrust": thrust[mask],
            "mdot": mdot_arr[mask],
            "mass_gas": mass_gas[mask],
            "total_impulse": np.trapezoid(thrust[mask], t[mask]),
            "burn_time": t[np.max(np.where(thrust > 0))] if np.any(thrust > 0) else 0,
        }
