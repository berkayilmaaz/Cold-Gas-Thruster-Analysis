<p align="center">
  <img src="brky-logo.png" alt="brky.ai" height="64" style="border-radius:8px">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="marsis.png" alt="MARSIS" height="72">
</p>

<h3 align="center">Cold Gas Thruster Analysis</h3>

<p align="center">
  <strong>brky.ai × MARSIS</strong> — Marmara University Defence Systems Society<br>
  <a href="https://projects.brky.ai/thruster/">🔗 Live Demo</a> · <a href="https://brky.ai">brky.ai</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/numpy-≥1.24-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scipy-≥1.11-8CAAE6?logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## Overview

An engineering toolkit for designing and analyzing cold gas thruster systems, developed for the MARSIS rocketry team at Marmara University. Covers the full pneumatic chain from pressurized tank to nozzle exit.

```
Tank (P_tank) → Regulator (P_reg) → Feed Line → CD Nozzle → Atmosphere
```

The companion web interface is live at **[projects.brky.ai/thruster](https://projects.brky.ai/thruster/)** — an interactive single-page application where you can adjust parameters, run simulations, and visualize results without installing anything.

## Features

- **Nozzle Design & Analysis** — Converging-diverging nozzle sizing for target thrust, or performance prediction from existing geometry. Isentropic flow relations with discharge coefficient correction.
- **Pipe Flow** — Darcy-Weisbach pressure loss with iterative Colebrook-White friction factor. Diameter sweep for feed line selection.
- **Tank Blowdown** — Time-domain simulation of regulated and unregulated blowdown modes. Tracks thrust, mass flow, and tank pressure over time.
- **Multi-Gas Support** — Air, N₂, He, CO₂ with proper thermodynamic properties (Sutherland viscosity, ideal gas EOS).
- **Validation Suite** — Automated checks against Anderson's compressible flow tables, Blasius correlation, and analytical limiting cases.

## Project Structure

```
pneumatic_analysis/
├──assets
|   ├──brky-logo.png
|   ├──marsis.png
├── __init__.py          # package exports
├── gas.py               # ideal gas properties, pre-defined gases
├── isentropic.py        # isentropic compressible flow relations
├── nozzle.py            # CD nozzle design & off-design analysis
├── pipe.py              # Darcy-Weisbach pipe loss, Colebrook solver
├── thruster.py          # system-level model, tank blowdown
├── plots.py             # matplotlib dark-theme visualization
├── main.py              # CLI entry point, full analysis pipeline
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/berkayilmaaz/pneumatic-thruster-analysis.git
cd pneumatic-thruster-analysis

pip install -r requirements.txt

python main.py                # full analysis + plots
python main.py --no-plots     # numbers only
python main.py --validate     # run validation suite
```

## Usage

### Nozzle Design

```python
from pneumatic_analysis import AIR, Nozzle

nozzle = Nozzle(gas=AIR, Cd=0.95)

perf = nozzle.design(
    p0=40e5,            # chamber pressure [Pa]
    T0=298.0,           # stagnation temperature [K]
    pa=89875.0,         # ambient pressure [Pa]
    target_thrust=83.0  # desired thrust [N]
)

print(f"Throat: {perf.geometry.d_throat*1e3:.2f} mm")
print(f"Exit:   {perf.geometry.d_exit*1e3:.2f} mm")
print(f"Isp:    {perf.isp:.1f} s")
print(f"ṁ:      {perf.mdot*1e3:.2f} g/s")
```

### Existing Nozzle Performance

```python
from pneumatic_analysis import Nozzle, NozzleGeometry, AIR

nozzle = Nozzle(gas=AIR, Cd=0.95)
geom = NozzleGeometry(d_throat=3.8e-3, d_exit=8.4e-3, d_inlet=8.0e-3)

perf = nozzle.analyze(geom, p0=40e5, T0=298.0, pa=89875.0)
print(f"Thrust: {perf.thrust:.2f} N")
```

### Feed Line Loss

```python
from pneumatic_analysis import PipeFlow, PipeGeometry, ROUGHNESS, AIR

pipe = PipeGeometry(
    length=1.0,
    inner_diameter=9.5e-3,
    roughness=ROUGHNESS["commercial_steel"]
)

result = PipeFlow(AIR).analyze(pipe, mdot=0.140, p_in=40e5, T=298.0)
print(f"ΔP: {result.dp/1e5:.4f} bar ({result.loss_fraction*100:.3f}%)")
```

### Tank Blowdown

```python
from pneumatic_analysis import ColdGasThruster, ThrusterConfig, NozzleGeometry, AIR

thruster = ColdGasThruster(gas=AIR)
config = ThrusterConfig(p_chamber=40e5, T0=298.0, p_ambient=89875.0, Cd=0.95)
geom = NozzleGeometry(d_throat=3.8e-3, d_exit=8.4e-3, d_inlet=8.0e-3)

blowdown = thruster.tank_blowdown(
    config, geom,
    V_tank=9e-3,
    p_tank_initial=300e5,
    p_regulated=40e5
)

print(f"Burn time: {blowdown['burn_time']:.2f} s")
print(f"Total impulse: {blowdown['total_impulse']:.1f} N·s")
```

## Physics

The analysis rests on three pillars:

| Module | Governing Equations | Assumptions |
|--------|-------------------|-------------|
| **Isentropic Flow** | Energy conservation + ideal gas EOS + $P/\rho^\gamma = \text{const}$ | Adiabatic, reversible, calorically perfect gas |
| **Nozzle** | $F = \dot{m}V_e + (P_e - P_a)A_e$, choked mass flow | Quasi-1D, steady, isentropic core flow |
| **Pipe Flow** | Darcy-Weisbach $\Delta P = f \cdot \frac{L}{D} \cdot \frac{\rho v^2}{2}$ | Isothermal, fully developed, incompressible iteration |

Key relations used throughout:

$$\frac{T_0}{T} = 1 + \frac{\gamma - 1}{2} M^2 \qquad \frac{P_0}{P} = \left(\frac{T_0}{T}\right)^{\gamma/(\gamma-1)}$$

$$\dot{m}_{\max} = A^* P_0 \sqrt{\frac{\gamma}{R T_0}} \left(\frac{2}{\gamma+1}\right)^{(\gamma+1)/(2(\gamma-1))}$$

## Validation

The `--validate` flag checks computed values against textbook data (Anderson, *Modern Compressible Flow*) and analytical limits. All isentropic ratios, area-Mach inversions, and friction factor correlations are tested.

```
$ python main.py --validate

  ✓  P*/P0 ............................ 0.528282
  ✓  T*/T0 ............................ 0.833333
  ✓  P0/P at M=2 ...................... 7.824450
  ✓  A/A* at M=3 ...................... 4.234568
  ...
  Result: 16/16 checks passed.
```

## Web Interface

The interactive web version at **[projects.brky.ai/thruster](https://projects.brky.ai/thruster/)** mirrors the Python toolkit with client-side JavaScript. It features:

- Real-time nozzle sizing with adjustable parameters
- Thrust curve visualization across chamber pressures
- Feed line diameter sensitivity analysis
- Tank blowdown simulation with regulated/unregulated modes
- Bilingual support (TR/EN)

Built as a single-file HTML application following the [brky.ai design system](https://brky.ai).

## Author

**Berkay Yılmaz** — [brky.ai](https://brky.ai) · [LinkedIn](https://linkedin.com/in/berkayilmaaz)

Developed for the **MARSIS** (Marmara University Defence Systems Society) rocketry team, TUSAŞ LiftUp Program.

---

<p align="center">
  <sub>brky.ai × MARSIS — Marmara University</sub>
</p>
