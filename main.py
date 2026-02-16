#!/usr/bin/env python3
"""
Pneumatic Thruster Analysis — Main entry point.

Runs a complete analysis for a MARSIS-like cold gas thruster system:
  1. Nozzle design for target thrust
  2. Thrust estimation for existing hardware
  3. Pipe loss analysis
  4. Tank blowdown simulation
  5. Validation against analytical limits

Usage:
    python main.py                  # full analysis + plots
    python main.py --no-plots       # numbers only
    python main.py --validate       # run validation suite
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pneumatic_analysis import (
    AIR, NITROGEN, HELIUM,
    IsentropicFlow, Nozzle, NozzleGeometry,
    PipeFlow, PipeGeometry, ROUGHNESS,
    ColdGasThruster, ThrusterConfig,
)

G0 = 9.80665
SEPARATOR = "═" * 65


def print_header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run_nozzle_design():
    """Tab 1 equivalent: design nozzles for target thrust values."""
    print_header("NOZZLE DESIGN — CD Nozzle Sizing")

    # operating conditions (MARSIS-like)
    p0 = 40e5       # 40 bar (after regulator)
    T0 = 298.0      # K
    pa = 89875.0     # ~0.899 bar (1 km altitude)

    nozzle = Nozzle(gas=AIR, Cd=1.0)  # ideal first

    targets = [
        ("Main thruster", 83.0),
        ("RCS thruster",  17.0),
        ("Low thrust",    10.0),
    ]

    for name, F in targets:
        perf = nozzle.design(p0, T0, pa, F)
        g = perf.geometry

        print(f"\n  {name} — F = {F:.0f} N")
        print(f"  {'─'*40}")
        print(f"  ṁ          = {perf.mdot*1e3:>10.3f} g/s")
        print(f"  D* (throat) = {g.d_throat*1e3:>10.3f} mm")
        print(f"  De (exit)   = {g.d_exit*1e3:>10.3f} mm")
        print(f"  Ae/A*       = {g.area_ratio:>10.3f}")
        print(f"  Me (exit)   = {perf.exit_mach:>10.4f}")
        print(f"  Ve          = {perf.exit_velocity:>10.1f} m/s")
        print(f"  Te          = {perf.exit_temperature:>10.1f} K")
        print(f"  Isp         = {perf.isp:>10.1f} s")

    # effect of discharge coefficient
    print(f"\n  {'─'*50}")
    print(f"  Discharge coefficient sensitivity (83N main thruster):")
    for Cd in [1.0, 0.95, 0.90, 0.85]:
        nozzle_real = Nozzle(gas=AIR, Cd=Cd)
        perf = nozzle_real.design(p0, T0, pa, 83.0)
        print(f"    Cd={Cd:.2f}: D*={perf.geometry.d_throat*1e3:.3f} mm, "
              f"ṁ={perf.mdot*1e3:.2f} g/s, Isp={perf.isp:.1f} s")


def run_thrust_estimation():
    """Tab 2 equivalent: predict thrust from existing nozzle geometry."""
    print_header("THRUST ESTIMATION — Existing Nozzle Performance")

    nozzle = Nozzle(gas=AIR, Cd=0.95)  # realistic Cd
    T0 = 298.0
    pa = 89875.0

    # define a nozzle geometry (MARSIS main-thruster-like)
    geom = NozzleGeometry(
        d_throat=3.8e-3,
        d_exit=8.4e-3,
        d_inlet=8.0e-3,
    )

    print(f"\n  Nozzle: D*={geom.d_throat*1e3:.1f} mm, "
          f"De={geom.d_exit*1e3:.1f} mm, Ae/A*={geom.area_ratio:.2f}")

    pressures = [10, 20, 30, 40, 50, 80, 100, 150, 200, 300]
    print(f"\n  {'P0 [bar]':>10} {'F [N]':>10} {'ṁ [g/s]':>10} "
          f"{'Me':>8} {'Pe [bar]':>10} {'Expanded?':>10}")
    print(f"  {'─'*62}")

    for p in pressures:
        try:
            perf = nozzle.analyze(geom, p * 1e5, T0, pa)
            exp = "✓" if perf.is_fully_expanded else "✗"
            print(f"  {p:>10} {perf.thrust:>10.2f} {perf.mdot*1e3:>10.3f} "
                  f"{perf.exit_mach:>8.3f} {perf.exit_pressure/1e5:>10.4f} {exp:>10}")
        except ValueError:
            print(f"  {p:>10} {'— not choked —':>42}")


def run_pipe_analysis():
    """Tab 3 equivalent: feed line pressure loss."""
    print_header("PIPE LOSS — Feed Line Analysis")

    pipe_calc = PipeFlow(gas=AIR)

    # baseline config
    pipe = PipeGeometry(
        length=1.0,
        inner_diameter=9.5e-3,  # 3/8" tubing
        roughness=ROUGHNESS["commercial_steel"],
    )

    mdot = 0.140     # kg/s (main thruster)
    p_in = 40e5      # 40 bar
    T0 = 298.0

    result = pipe_calc.analyze(pipe, mdot, p_in, T0)
    loss_pct = result.loss_fraction * 100

    print(f"\n  Pipe: L={pipe.length}m, D={pipe.inner_diameter*1e3:.1f}mm, "
          f"ε={pipe.roughness*1e3:.3f}mm")
    print(f"  Flow: ṁ={mdot:.3f} kg/s, P_in={p_in/1e5:.0f} bar, T={T0:.0f} K")
    print(f"\n  {'─'*40}")
    print(f"  ΔP           = {result.dp/1e5:.4f} bar")
    print(f"  P_out        = {result.p_out/1e5:.4f} bar")
    print(f"  Loss         = {loss_pct:.3f}%")
    print(f"  Re           = {result.reynolds:.0f}  ({result.regime})")
    print(f"  f (Darcy)    = {result.friction_factor:.6f}")
    print(f"  v (bulk)     = {result.velocity:.2f} m/s")
    print(f"  Mach (pipe)  = {result.mach:.4f}")

    if loss_pct > 10:
        print(f"\n  ⚠ CRITICAL: Loss exceeds 10% — increase pipe diameter or reduce length!")
    elif result.mach > 0.3:
        print(f"\n  ⚠ WARNING: Pipe Mach > 0.3 — compressibility effects significant.")
        print(f"             Consider Fanno flow model for better accuracy.")
    else:
        print(f"\n  ✓ Loss acceptable.")

    # pipe diameter sweep
    print(f"\n  Diameter sensitivity (L={pipe.length}m, ṁ={mdot} kg/s):")
    for d in [4, 6, 8, 9.5, 12, 16, 20]:
        p = PipeGeometry(length=1.0, inner_diameter=d * 1e-3,
                         roughness=ROUGHNESS["commercial_steel"])
        r = pipe_calc.analyze(p, mdot, p_in, T0)
        flag = "⚠" if r.loss_fraction > 0.10 else " "
        print(f"    D={d:>5.1f} mm → ΔP={r.dp/1e5:>8.3f} bar "
              f"({r.loss_fraction*100:>6.3f}%) Re={r.reynolds:>9.0f}  {flag}")


def run_blowdown():
    """Simulate tank blowdown — regulated and unregulated modes."""
    print_header("TANK BLOWDOWN SIMULATION")

    thruster = ColdGasThruster(gas=AIR)

    geom = NozzleGeometry(
        d_throat=3.8e-3, d_exit=8.4e-3, d_inlet=8.0e-3,
    )

    config = ThrusterConfig(
        p_chamber=40e5,
        T0=298.0,
        p_ambient=89875.0,
        Cd=0.95,
    )

    V_tank = 9e-3  # 9 liters

    # regulated mode
    bd_reg = thruster.tank_blowdown(
        config, geom,
        V_tank=V_tank,
        p_tank_initial=300e5,
        p_regulated=40e5,
        dt=0.01,
    )

    print(f"\n  Tank: {V_tank*1e3:.0f} L at {300:.0f} bar")
    print(f"  Nozzle: D*={geom.d_throat*1e3:.1f} mm (Cd={config.Cd})")

    print(f"\n  REGULATED (40 bar):")
    print(f"    Burn time       = {bd_reg['burn_time']:.2f} s")
    print(f"    Total impulse   = {bd_reg['total_impulse']:.1f} N·s")
    print(f"    Gas consumed    = {(bd_reg['mass_gas'][0] - bd_reg['mass_gas'][-1])*1e3:.1f} g")

    # unregulated blowdown
    bd_unreg = thruster.tank_blowdown(
        config, geom,
        V_tank=V_tank,
        p_tank_initial=300e5,
        p_regulated=None,
        dt=0.01,
    )

    print(f"\n  UNREGULATED (blowdown from 300 bar):")
    print(f"    Burn time       = {bd_unreg['burn_time']:.2f} s")
    print(f"    Total impulse   = {bd_unreg['total_impulse']:.1f} N·s")
    print(f"    Peak thrust     = {max(bd_unreg['thrust']):.1f} N")
    print(f"    Gas consumed    = {(bd_unreg['mass_gas'][0] - bd_unreg['mass_gas'][-1])*1e3:.1f} g")


def run_validation():
    """
    Validate physics against known analytical solutions.

    NOT comparing to KTR — comparing to textbook results
    and limiting cases where we know the exact answer.
    """
    print_header("VALIDATION — Analytical Checks")

    flow = IsentropicFlow(AIR)
    passed = 0
    total = 0

    def check(name, computed, expected, tol=1e-4):
        nonlocal passed, total
        total += 1
        err = abs(computed - expected) / abs(expected) if expected != 0 else abs(computed)
        ok = err < tol
        if ok:
            passed += 1
        status = "✓" if ok else "✗"
        print(f"  {status}  {name:.<45} {computed:>12.6f}  (expected {expected:.6f}, err={err:.2e})")
        return ok

    print("\n  Isentropic flow (air, γ=1.4):")
    print(f"  {'─'*70}")

    # critical ratios — these are textbook constants
    check("P*/P0", flow.critical_pressure_ratio(), 0.528282)
    check("T*/T0", flow.critical_temperature_ratio(), 0.833333)

    # Mach 1: all ratios should equal their critical values
    check("P0/P at M=1", flow.pressure_ratio(1.0), 1.0/0.528282, tol=1e-3)
    check("T0/T at M=1", flow.temperature_ratio(1.0), 1.2)
    check("A/A* at M=1", flow.area_ratio(1.0), 1.0)

    # Mach 2: standard table values (Anderson, Table A.1)
    check("P0/P at M=2", flow.pressure_ratio(2.0), 7.82445, tol=1e-3)
    check("T0/T at M=2", flow.temperature_ratio(2.0), 1.8)
    check("A/A* at M=2", flow.area_ratio(2.0), 1.6875)

    # Mach 3: standard table
    check("P0/P at M=3", flow.pressure_ratio(3.0), 36.7327, tol=1e-3)
    check("T0/T at M=3", flow.temperature_ratio(3.0), 2.8)
    check("A/A* at M=3", flow.area_ratio(3.0), 4.23457, tol=1e-3)

    # inverse: Mach from pressure ratio
    check("M from P0/P=7.824", flow.mach_from_pressure_ratio(7.82445, 1.0), 2.0, tol=1e-3)

    # Mach from area ratio (supersonic branch)
    check("M from A/A*=1.6875 (sup)", flow.mach_from_area_ratio(1.6875, supersonic=True), 2.0, tol=1e-3)
    check("M from A/A*=1.6875 (sub)", flow.mach_from_area_ratio(1.6875, supersonic=False), 0.37217, tol=1e-2)

    # mass flow check: choked throat with known conditions
    # for a 1 cm² throat at 10 bar, 300K air
    a_throat = 1e-4  # m²
    p0 = 10e5
    T0 = 300.0
    mdot_per_a = flow.choked_mdot_per_area(p0, T0)
    mdot = mdot_per_a * a_throat
    # analytical: mdot = A*P0*sqrt(γ/(R*T0)) * (2/(γ+1))^((γ+1)/(2(γ-1)))
    expected_mdot = a_throat * p0 * np.sqrt(1.4 / (287.05 * 300)) * (2/2.4)**3.0
    check("ṁ at A*=1cm², P0=10bar", mdot, expected_mdot, tol=1e-4)

    # pipe flow: laminar Poiseuille
    print(f"\n  Pipe flow:")
    print(f"  {'─'*70}")
    from pneumatic_analysis.pipe import PipeFlow
    pf = PipeFlow(AIR)
    f_lam = pf.colebrook(1000, 0.001)  # Re=1000, laminar
    check("f (laminar, Re=1000)", f_lam, 64.0/1000.0)

    # turbulent smooth pipe at Re=100000 (Blasius: f ≈ 0.316/Re^0.25)
    f_turb = pf.colebrook(100000, 0.0)
    f_blasius = 0.316 / 100000**0.25
    check("f (smooth turbulent, Re=1e5) ≈ Blasius", f_turb, f_blasius, tol=0.05)

    # helium: different gamma, same framework should work
    print(f"\n  Multi-gas (Helium, γ=5/3):")
    print(f"  {'─'*70}")
    flow_he = IsentropicFlow(HELIUM)
    check("He P*/P0", flow_he.critical_pressure_ratio(), (2/2.6667)**(2.5), tol=1e-3)
    check("He T*/T0", flow_he.critical_temperature_ratio(), 2.0/2.6667, tol=1e-3)

    print(f"\n  Result: {passed}/{total} checks passed.")
    if passed == total:
        print("  All validations passed ✓")
    else:
        print(f"  ⚠ {total - passed} check(s) failed!")

    return passed == total


def generate_plots(output_dir: str = "output"):
    """Generate all analysis plots to files."""
    from pneumatic_analysis.plots import (
        plot_nozzle_profile, plot_thrust_vs_pressure,
        plot_loss_vs_diameter, plot_blowdown,
    )
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(output_dir, exist_ok=True)
    print_header(f"GENERATING PLOTS → {output_dir}/")

    nozzle = Nozzle(gas=AIR, Cd=0.95)
    p0 = 40e5
    T0 = 298.0
    pa = 89875.0

    # 1: nozzle profile
    perf = nozzle.design(p0, T0, pa, 83.0)
    fig = plot_nozzle_profile(perf.geometry)
    fig.savefig(f"{output_dir}/nozzle_profile.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ nozzle_profile.png")

    # 2: thrust vs pressure
    geom = NozzleGeometry(d_throat=3.8e-3, d_exit=8.4e-3, d_inlet=8.0e-3)
    pressures, thrusts = nozzle.thrust_curve(geom, T0, pa)
    refs = {"Regulator (40 bar)": 40e5, "Tank (300 bar)": 300e5}
    fig = plot_thrust_vs_pressure(pressures, thrusts, reference_lines=refs)
    fig.savefig(f"{output_dir}/thrust_curve.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ thrust_curve.png")

    # 3: pipe loss vs diameter
    pipe_calc = PipeFlow(gas=AIR)
    diameters, losses = pipe_calc.sweep_diameter(
        length=1.0, roughness=ROUGHNESS["commercial_steel"],
        mdot=0.140, p_in=40e5, T=298.0,
    )
    fig = plot_loss_vs_diameter(diameters, losses, 40e5)
    fig.savefig(f"{output_dir}/pipe_loss.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ pipe_loss.png")

    # 4: blowdown
    thruster = ColdGasThruster(gas=AIR)
    config = ThrusterConfig(p_chamber=40e5, T0=298.0, p_ambient=89875.0, Cd=0.95)
    bd = thruster.tank_blowdown(
        config, geom, V_tank=9e-3, p_tank_initial=300e5, p_regulated=40e5,
    )
    fig = plot_blowdown(bd)
    fig.savefig(f"{output_dir}/blowdown.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ blowdown.png")

    print(f"\n  All plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Pneumatic thruster design & analysis tool"
    )
    parser.add_argument("--validate", action="store_true",
                        help="Run validation suite only")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    if args.validate:
        success = run_validation()
        sys.exit(0 if success else 1)

    run_nozzle_design()
    run_thrust_estimation()
    run_pipe_analysis()
    run_blowdown()
    run_validation()

    if not args.no_plots:
        generate_plots()

    print(f"\n{SEPARATOR}")
    print("  Done.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
