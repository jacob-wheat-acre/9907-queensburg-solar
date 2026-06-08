"""
9907 Queensburg St – Solar Power Quality Dashboard
Streamlit app for the OpenDSS model of the DIA solar site.

Data source: analytical circuit computation using DSS model parameters.
If `opendssdirect` is installed, power flow results come from live DSS runs.
"""

import warnings
warnings.filterwarnings("ignore")

import subprocess
import sys
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="9907 Queensburg – Power Quality",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DSS_FILE = Path(__file__).parent / "9907_queensburg_solar.dss"

# ── System constants (from DSS model) ─────────────────────────────────────────
SYS = dict(
    xcel_kv=24.9, dia7_kv=0.48, mv_kv=13.8, site_kv=0.63,
    t1_kva=2500, t2_kva=2350, t3_kva=2520,
    line_mi=1.94, inv_kw=750, total_kw=2250,
    mvasc=500,           # Xcel fault duty at substation
    xhl_pct=5.73,        # transformer leakage reactance %
    r_pct=0.50,          # transformer total resistance %
    # Customer 13.8 kV feeder — 1,000 kcmil Al, 15 kV class XLPE underground
    r1_per_mi=0.063,     # Ω/mi  (was 0.306 for overhead ACSR)
    x1_per_mi=0.040,     # Ω/mi  (was 0.627)
    c1_nf_per_mi=380,    # nF/mi (was 5.60 — underground is ~68× higher)
    # Xcel 24.9 kV feeder — 1,000 kcmil Al, 25 kV class XLPE underground, 2.0 mi
    xcel_line_mi=2.0,
    xcel_r1_per_mi=0.063,
    xcel_x1_per_mi=0.050,
    xcel_c1_nf_per_mi=290,  # nF/mi, 25 kV class (thicker insulation → less C)
)
BASE_KVA   = 2500.0
IEEE519_V  = 3.0    # % individual voltage harmonic at PCC
IEEE519_THD= 5.0    # % total voltage THD at PCC
DG_I_LIMIT = 10.0  # % individual current harmonic (DG Tech Manual)
HARM_ORDERS= [1, 3, 5, 7, 9, 11, 13]

SCENARIOS = {
    "100%  –  2,250 kW":  dict(pct=100, n_inv=3, kw=2250),
    "75%   –  1,688 kW":  dict(pct=75,  n_inv=3, kw=1688),
    "50%   –  1,125 kW":  dict(pct=50,  n_inv=3, kw=1125),
    "25%   –    563 kW":  dict(pct=25,  n_inv=3, kw=563),
    "No inverters (idle)":dict(pct=0,   n_inv=0, kw=0),
}

# ── Circuit impedance functions (all in per-unit, BASE_KVA base) ──────────────
def xfmr_z(kva_rated, h=1):
    """Transformer series impedance in p.u. at harmonic h."""
    s = BASE_KVA / kva_rated
    return complex(SYS["r_pct"] / 100 * s, SYS["xhl_pct"] / 100 * s * h)

def line_z(h=1):
    """Line series impedance in p.u. at harmonic h."""
    zb = (SYS["mv_kv"] ** 2 * 1e3) / BASE_KVA
    R  = SYS["r1_per_mi"] * SYS["line_mi"]
    X  = SYS["x1_per_mi"] * SYS["line_mi"] * h
    return complex(R / zb, X / zb)

def line_yc(h=1):
    """Line shunt capacitive admittance in p.u. at harmonic h."""
    C  = SYS["c1_nf_per_mi"] * SYS["line_mi"] * 1e-9  # F
    B  = 2 * np.pi * 60 * h * C
    yb = BASE_KVA / (SYS["mv_kv"] ** 2 * 1e3)
    return complex(0, B / yb)

def source_z(h=1):
    """Xcel substation Thevenin impedance in p.u. at harmonic h."""
    return complex(0, BASE_KVA / (SYS["mvasc"] * 1e3) * h)

def xcel_cable_z(h=1):
    """Xcel 24.9 kV underground cable series impedance in p.u. at harmonic h."""
    zb = (SYS["xcel_kv"] ** 2 * 1e3) / BASE_KVA
    R  = SYS["xcel_r1_per_mi"] * SYS["xcel_line_mi"]
    X  = SYS["xcel_x1_per_mi"] * SYS["xcel_line_mi"] * h
    return complex(R / zb, X / zb)

def xcel_cable_yc(h=1):
    """Xcel 24.9 kV cable shunt capacitive admittance in p.u. at harmonic h."""
    C  = SYS["xcel_c1_nf_per_mi"] * SYS["xcel_line_mi"] * 1e-9
    B  = 2 * np.pi * 60 * h * C
    yb = BASE_KVA / (SYS["xcel_kv"] ** 2 * 1e3)
    return complex(0, B / yb)

def z_at_bus480(h):
    """
    Driving-point impedance at Bus_480V at harmonic h (p.u.).
    Two parallel paths from Bus_480V:
      • Toward source: T1 + Xcel cable (pi model) + substation Thevenin
      • Toward load:   T2 + customer cable (pi model) + T3
    Both cable sections are modeled as pi sections (shunt C/2 at each end).
    """
    # Source path — Xcel cable pi section terminating at substation Thevenin Z
    Yc_xcel  = xcel_cable_yc(h)
    Z_xcel_L = xcel_cable_z(h)
    Y_sub    = Yc_xcel / 2 + 1.0 / source_z(h)
    Z_xcel_thru = Z_xcel_L + 1.0 / Y_sub
    Y_xcel_send = Yc_xcel / 2 + 1.0 / Z_xcel_thru
    Z_src_path  = xfmr_z(SYS["t1_kva"], h) + 1.0 / Y_xcel_send

    # Load path — customer cable pi section terminating at T3
    Z_T2 = xfmr_z(SYS["t2_kva"], h)
    Z_T3 = xfmr_z(SYS["t3_kva"], h)
    Z_L  = line_z(h)
    Yc   = line_yc(h)
    Y_recv = Yc / 2 + 1.0 / Z_T3
    Z_thru = Z_L + 1.0 / Y_recv
    Y_send = Yc / 2 + 1.0 / Z_thru
    Z_load_path = Z_T2 + 1.0 / Y_send

    return 1.0 / (1.0 / Z_src_path + 1.0 / Z_load_path)

def z_base_480():
    return (SYS["dia7_kv"] ** 2 * 1e3) / BASE_KVA  # Ω

# ── OpenDSS via subprocess ────────────────────────────────────────────────────
# libdss_capi.dylib raises SIGILL in Streamlit worker threads on ARM64 macOS,
# killing the whole process. Running DSS in a child process isolates any crash.

_DSS_WORKER = """\
import sys, json, warnings
warnings.filterwarnings("ignore")
pct   = float(sys.argv[1])
n_inv = int(sys.argv[2])
dss_file = sys.argv[3]
import opendssdirect as d
d.run_command("Clear")
d.run_command("Redirect " + dss_file)
for i in range(1, 4):
    en  = "true" if i <= n_inv else "false"
    irr = "{:.3f}".format(pct / 100)
    d.run_command("PVSystem.Inverter" + str(i) + ".enabled=" + en)
    d.run_command("PVSystem.Inverter" + str(i) + ".irradiance=" + irr)
d.run_command("Set Mode=Snapshot")
d.run_command("Solve")
buses    = list(d.Circuit.AllBusNames())
vpu_all  = list(d.Circuit.AllBusMagPu())
# AllBusMagPu returns one value per node (phase); average phases per bus
n = len(buses)
stride = len(vpu_all) // n if n else 3
vpu = []
for i in range(n):
    vals = [v for v in vpu_all[i*stride:(i+1)*stride] if v > 0.01]
    vpu.append(round(sum(vals)/len(vals), 5) if vals else 0.0)
rows  = []
idx   = d.Transformers.First()
while idx > 0:
    nm  = d.Transformers.Name()
    kva = d.Transformers.kVA()
    d.Circuit.SetActiveElement("Transformer." + nm)
    p   = list(d.CktElement.Powers())
    kw  = sum(p[i] for i in range(0, min(len(p), 6), 2))
    kvar = sum(p[i] for i in range(1, min(len(p), 6), 2))
    ka  = (kw**2 + kvar**2) ** 0.5
    rows.append(dict(Transformer=nm, kVA_rated=kva, kW=round(kw,1),
                     kVAR=round(kvar,1), kVA=round(ka,1),
                     Loading_pct=round(100*ka/kva,1) if kva>0 else 0))
    idx = d.Transformers.Next()
print(json.dumps(dict(buses=buses, vpu=vpu, xfmrs=rows,
                      ok=bool(d.Solution.Converged()))))
"""


@st.cache_resource
def _dss_available() -> bool:
    """Check if opendssdirect works in a subprocess (absorbs any SIGILL crash)."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import opendssdirect as d; d.run_command('New Circuit.Probe'); print('ok')"],
            capture_output=True, text=True, timeout=8,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False


def run_dss(pct, n_inv):
    """Run a snapshot power flow in a child process; return dict or None."""
    if not _dss_available():
        return None
    try:
        r = subprocess.run(
            [sys.executable, "-c", _DSS_WORKER,
             str(pct), str(n_inv), str(DSS_FILE)],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "DSS exited non-zero"
            return dict(error=tail)
        return json.loads(r.stdout)
    except Exception as e:
        return dict(error=str(e))

# ── Analytical estimates for power flow ───────────────────────────────────────
def analytical_voltages(pct):
    """Approximate bus voltages (pu). Solar generation slightly raises voltage."""
    scale = pct / 100.0
    return {
        "SourceBus\n24.9 kV":    1.000,
        "Bus_480V\n0.48 kV":     0.997 + 0.003 * scale,
        "Bus_13800\n(send)":     0.996 + 0.004 * scale,
        "Bus_13800\n(recv)":     0.992 + 0.007 * scale,
        "Bus_630V\n0.63 kV":     0.988 + 0.010 * scale,
    }

def analytical_xfmr_loading(kw):
    """Approximate transformer loading %."""
    return [
        dict(Name="T1  Xcel Service\n2,500 kVA",  Loading=round(100*kw*0.92/SYS["t1_kva"],1)),
        dict(Name="T2  Step-Up\n2,350 kVA",        Loading=round(100*kw*0.95/SYS["t2_kva"],1)),
        dict(Name="T3  Step-Down\n2,520 kVA",      Loading=round(100*kw/SYS["t3_kva"],1)),
    ]

# ── Harmonic data ─────────────────────────────────────────────────────────────
# Inverter injection: realistic PWM inverter, % of fundamental current
INJECTED = {1:100, 3:0.5, 5:4.0, 7:2.0, 9:0.3, 11:1.0, 13:0.5}

# Transfer through circuit: inverter current referred to 0.48kV bus
# Turns-ratio chain: 0.63kV → 13.8kV (T3) → 0.48kV (T2) → net factor ≈ 1.31
XFER_RATIO = (SYS["site_kv"] / SYS["mv_kv"]) * (SYS["mv_kv"] / SYS["dia7_kv"])  # ≈ 1.31

# Harmonic voltage at Bus_480V per unit of harmonic current (from impedance scan)
# V_harm = I_harm × |Z_480V(h)|  →  shown in the impedance tab

# T2 magnetizing harmonic content (% of magnetizing current, which is 0.5% of rated)
XFMR_MAG = {1:100, 3:35, 5:15, 7:7, 9:2, 11:1, 13:0.5}

def harm_current_amps(pct):
    """
    Returns {h: amps} at Bus_480V for the given production level.
    Inverter injection referred through transformer turns ratios.
    """
    if pct == 0:
        # Only T2 magnetizing current (Figure 8 in report)
        i_mag_rated = SYS["t2_kva"] / (np.sqrt(3) * SYS["dia7_kv"]) * 0.005  # 0.5% imag
        return {h: i_mag_rated * XFMR_MAG[h] / 100 for h in HARM_ORDERS}
    i_fund_480 = SYS["total_kw"] * (pct/100) / (np.sqrt(3) * SYS["dia7_kv"])
    result = {}
    for h in HARM_ORDERS:
        inv_amps_per_inv = SYS["inv_kw"] * (pct/100) / (np.sqrt(3) * SYS["site_kv"])
        harm_inv = 3 * inv_amps_per_inv * INJECTED[h] / 100
        harm_at_480 = harm_inv * XFER_RATIO
        result[h] = harm_at_480
    return result

def harm_pct_of_fund(pct):
    """Returns {h: % of fundamental} at Bus_480V."""
    amps = harm_current_amps(pct)
    fund = amps[1] if amps[1] > 0 else 1
    return {h: 100 * amps[h] / fund for h in HARM_ORDERS}

# ── One-line diagram ──────────────────────────────────────────────────────────
def draw_oneline(n_inv_active=3):
    BG   = "#0e1117"
    WIRE = "#8899bb"
    RED  = "#e05c5c"
    BLUE = "#4a9eff"
    GOLD = "#f5c542"
    PURP = "#bb88ff"
    GRN  = "#4fc978"
    WHT  = "#e8e8e8"
    DIM  = "#556677"
    TEAL = "#4dd0c9"

    W, H = 19.5, 6.5
    Y = 3.2

    fig, ax = plt.subplots(figsize=(W, H), dpi=130)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    def hline(x0, x1, y=Y, color=WIRE, lw=2.2):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="round")

    def vline(x, y0, y1, color=WIRE, lw=2.2):
        ax.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="round")

    def bus_bar(x, y_ctr, label, kv_str, color=BLUE):
        vline(x, y_ctr - 0.6, y_ctr + 0.6, color=color, lw=6)
        ax.text(x, y_ctr + 0.76, label, ha="center", va="bottom",
                fontsize=9.5, color=WHT, fontweight="bold")
        ax.text(x, y_ctr - 0.78, kv_str, ha="center", va="top",
                fontsize=9, color=color)

    def xfmr_tri(x, y, name, detail, color=RED):
        tri = plt.Polygon(
            [[x - 0.50, y - 0.44], [x + 0.50, y - 0.44], [x, y + 0.54]],
            closed=True, color=color, zorder=4, alpha=0.88)
        ax.add_patch(tri)
        ax.text(x, y + 0.68, name, ha="center", va="bottom",
                fontsize=9, color=WHT, fontweight="bold")
        ax.text(x, y - 0.60, detail, ha="center", va="top",
                fontsize=8, color=WIRE, linespacing=1.4)

    def inv_box(x, y, label, active=True):
        c = GOLD if active else DIM
        fc = "#252540" if active else "#1a1a2e"
        r = plt.Rectangle((x - 0.60, y - 0.34), 1.20, 0.68,
                           linewidth=1.8, edgecolor=c,
                           facecolor=fc, zorder=4)
        ax.add_patch(r)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9.5, color=c, fontweight="bold", zorder=5)

    def der_circle(x, y, active=True):
        c = GRN if active else DIM
        ax.add_patch(plt.Circle((x, y), 0.35, color=c, zorder=4))
        ax.text(x, y, "DER", ha="center", va="center",
                fontsize=8.5, color="#0e1117", fontweight="bold", zorder=5)

    def cable_section(x0, x1, label, stub_xs, color=WIRE):
        hline(x0, x1, color=color)
        xm = (x0 + x1) / 2
        ax.text(xm, Y + 0.28, f"  {label}  ",
                ha="center", fontsize=8.5, color=color, style="italic",
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none"))
        for xc in stub_xs:
            vline(xc, Y - 0.60, Y, lw=1.4, color=color)
            ax.plot([xc - 0.18, xc + 0.18], [Y - 0.60, Y - 0.60], color=color, lw=2.5)
            ax.plot([xc - 0.11, xc + 0.11], [Y - 0.80, Y - 0.80], color=color, lw=2.5)

    # Key x positions
    X_SUB  = 1.20   # Substation 24.9 kV bus (Xcel)
    X_T1   = 4.20   # T1 center
    X_480  = 5.55   # DIA7 0.48 kV bus
    X_T2   = 7.00   # T2 center
    X_13S  = 8.35   # 13.8 kV send bus
    X_13R  = 12.55  # 13.8 kV receive bus
    X_T3   = 13.95  # T3 center
    X_63   = 15.25  # 0.63 kV inverter bus

    # ── Xcel substation source
    ax.text(0.52, Y, "Xcel\nSubstation\n500 MVA sc", ha="center", va="center",
            fontsize=8.5, color=WHT,
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec=TEAL, lw=1.5))
    hline(0.90, X_SUB, color=TEAL)
    bus_bar(X_SUB, Y, "Substation Bus", "24.9 kV", color=TEAL)

    # ── Xcel 2.0-mile underground cable (teal = Xcel-owned)
    cable_section(X_SUB, X_T1 - 0.50,
                  "2.0 mi · 24.9 kV · 1,000 kcmil Al XLPE Underground",
                  stub_xs=[1.70, 2.45, 3.20], color=TEAL)

    # ── T1: Xcel service transformer
    xfmr_tri(X_T1, Y, "T1 – Xcel Service",
             "2,500 kVA · 24.9 / 0.48 kV\nWye–Wye")
    hline(X_T1 + 0.50, X_480)

    # ── 0.48 kV bus
    bus_bar(X_480, Y, "DIA7 Switchgear", "0.48 kV")

    # Zig-Zag grounding transformer (below bus)
    vline(X_480, Y - 0.6, Y - 1.25)
    zz = plt.Rectangle((X_480 - 0.60, Y - 1.90), 1.20, 0.65,
                        linewidth=1.3, edgecolor=PURP,
                        facecolor="#1a1a2e", zorder=4)
    ax.add_patch(zz)
    ax.text(X_480, Y - 1.575, "Zig-Zag · 250 kVA\nGrounding Xfmr",
            ha="center", va="center", fontsize=8.5, color=PURP, fontweight="bold",
            linespacing=1.35, zorder=5)
    vline(X_480, Y - 1.90, Y - 2.20)
    for k, hw in enumerate([0.24, 0.16, 0.08]):
        yg = Y - 2.20 - k * 0.15
        ax.plot([X_480 - hw, X_480 + hw], [yg, yg], color=PURP, lw=2.0)

    # ── T2: customer step-up
    hline(X_480, X_T2 - 0.50)
    xfmr_tri(X_T2, Y, "T2 – Step-Up  ⚠",
             "2,350 kVA · 0.48 / 13.8 kV\nWye–Wye  (no delta trap)",
             color="#e09030")
    hline(X_T2 + 0.50, X_13S)

    # ── 13.8 kV send bus
    bus_bar(X_13S, Y, "13.8 kV  Send", "13.8 kV")

    # ── 1.94-mile customer underground feeder
    cable_section(X_13S, X_13R,
                  "1.94 mi · 13.8 kV · 1,000 kcmil Al XLPE Underground",
                  stub_xs=[9.10, 10.45, 11.80])

    # ── 13.8 kV receive bus
    bus_bar(X_13R, Y, "13.8 kV  Receive", "13.8 kV")

    # ── T3: site step-down
    hline(X_13R, X_T3 - 0.50)
    xfmr_tri(X_T3, Y, "T3 – Site Step-Down",
             "2,520 kVA · 13.8 / 0.63 kV\nDelta–Wye")
    hline(X_T3 + 0.50, X_63)

    # ── 0.63 kV inverter bus
    bus_bar(X_63, Y, "Inverter Bus", "0.63 kV")

    # ── Three inverters
    inv_ys = [Y + 1.45, Y, Y - 1.45]
    for i, iy in enumerate(inv_ys):
        active = (i < n_inv_active)
        if iy != Y:
            vline(X_63, min(Y, iy), max(Y, iy))
        hline(X_63, X_63 + 0.85, y=iy)
        inv_box(X_63 + 1.50, iy, f"INV {i+1}\n750 kW", active=active)
        hline(X_63 + 2.10, X_63 + 2.70, y=iy)
        der_circle(X_63 + 3.15, iy, active=active)

    # ── Billing CT annotation
    ax.annotate(
        "Billing CT\n(measurement point\nin Xcel report)",
        xy=(X_480, Y + 0.6),
        xytext=(4.50, H - 0.55),
        fontsize=8.5, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.35", fc="#1a1a2e", ec=GOLD, lw=1.0),
    )

    plt.tight_layout(pad=0.3)
    return fig

# ── Sequence networks ─────────────────────────────────────────────────────────
def draw_sequence_networks():
    """
    Three-panel figure: positive, negative, and zero-sequence networks.
    All impedances in per-unit on 2500 kVA base.
    Xcel 2.0-mile XLPE underground cable (Z_xcbl) shown explicitly between
    SubstationBus and SourceBus. Shunt elements connect main rail to ground.
    """
    BG   = "#0e1117"
    WIRE = "#8899bb"
    RED  = "#e05c5c"
    GOLD = "#f5c542"
    PURP = "#bb88ff"
    GRN  = "#4fc978"
    BLUE = "#4a9eff"
    ORG  = "#e09030"
    TEAL = "#4dd0c9"
    WHT  = "#e8e8e8"
    DIM  = "#445566"

    # per-unit impedances on 2500 kVA base
    z1_src  = complex(0,           2500 / (500 * 1e3))
    z0_src  = complex(0, 3 * 2500 / (400 * 1e3) - 2 * 2500 / (500 * 1e3))
    z1_t1   = complex(0.00500, 0.05730)
    z1_t2   = complex(0.00500 * 2500/2350, 0.05730 * 2500/2350)
    Zb3     = 13.8**2 * 1e3 / 2500
    z1_line = complex(SYS["r1_per_mi"] * SYS["line_mi"] / Zb3,
                      SYS["x1_per_mi"] * SYS["line_mi"] / Zb3)
    z0_line = complex(0.11 * SYS["line_mi"] / Zb3, 0.17 * SYS["line_mi"] / Zb3)
    z1_t3   = complex(0.00500 * 2500/2520, 0.05730 * 2500/2520)
    z0_zz   = complex(0.00500 * 2500/250,  0.05000 * 2500/250)
    Zb_xcel = SYS["xcel_kv"]**2 * 1e3 / 2500
    z1_xcbl = complex(SYS["xcel_r1_per_mi"] * SYS["xcel_line_mi"] / Zb_xcel,
                      SYS["xcel_x1_per_mi"] * SYS["xcel_line_mi"] / Zb_xcel)
    z0_xcbl = z1_xcbl  # concentric neutral: Z₀ ≈ Z₁

    def fz(z):
        return f"{z.real:.4f}+j{z.imag:.4f}"

    fig, axes = plt.subplots(3, 1, figsize=(18.5, 13.5), dpi=110)
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.38, top=0.97, bottom=0.02, left=0.01, right=0.99)

    cfgs = [
        ("Positive-Sequence Network  (Z₁)  — series path, voltage source drives current through circuit",  GOLD, "1"),
        ("Negative-Sequence Network  (Z₂)  — same series impedances; no voltage source (passive network)", PURP, "2"),
        ("Zero-Sequence Network  (Z₀)  — shunt elements to ground are the key feature (see annotation)",   RED,  "0"),
    ]

    for ax, (title, tc, seq) in zip(axes, cfgs):
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.set_xlim(0, 18.5)
        ax.set_ylim(0, 4.8)
        ax.set_title(title, color=tc, fontsize=10.5, fontweight="bold", loc="left", pad=6)

        YM = 3.3
        YR = 0.6

        ax.plot([0.1, 18.4], [YR, YR], color="#334455", lw=2.0, zorder=1)
        ax.text(0.12, YR - 0.18, "REF / Ground (V=0)", fontsize=8.5,
                color="#667788", va="top", fontweight="bold")

        def dot(x, y, c=WIRE, r=0.07):
            ax.add_patch(plt.Circle((x, y), r, color=c, zorder=6))

        def wire(x0, x1, y=YM, lw=2.0):
            ax.plot([x0, x1], [y, y], color=WIRE, lw=lw, solid_capstyle="round")

        def vwire(x, y0, y1, c=WIRE, lw=2.0):
            ax.plot([x, x], [y0, y1], color=c, lw=lw, solid_capstyle="round")

        def ground_hash(x, c=WIRE):
            for k, hw in enumerate([0.13, 0.09, 0.05]):
                ax.plot([x - hw, x + hw], [YR - k * 0.10, YR - k * 0.10], color=c, lw=1.6)

        def bus_bar(x, name, kv="", color=BLUE):
            ax.plot([x, x], [YM - 0.45, YM + 0.45], color=color, lw=6.5, zorder=3)
            ax.text(x, YM + 0.60, name, ha="center", va="bottom",
                    fontsize=9, color=WHT, fontweight="bold")
            if kv:
                ax.text(x, YM - 0.60, kv, ha="center", va="top", fontsize=8.5, color=color)

        def zbox(x1, x2, label, val, c=WIRE):
            xm = (x1 + x2) / 2
            bw = (x2 - x1) * 0.58
            bh = 0.52
            wire(x1, xm - bw/2)
            wire(xm + bw/2, x2)
            r = plt.Rectangle((xm - bw/2, YM - bh/2), bw, bh,
                               ec=c, fc="#12122a", lw=1.6, zorder=4)
            ax.add_patch(r)
            ax.text(xm, YM + 0.09, label, ha="center", va="center",
                    fontsize=9, color=c, fontweight="bold", zorder=5)
            ax.text(xm, YM - 0.23, val, ha="center", va="center",
                    fontsize=7, color="#9999bb", zorder=5)

        def shunt_box(x, label, val, c=GRN):
            ym_box = (YM + YR) / 2
            bw = 1.20; bh = 0.46
            vwire(x, YM, ym_box + bh/2, c=c)
            r = plt.Rectangle((x - bw/2, ym_box - bh/2), bw, bh,
                               ec=c, fc="#12122a", lw=1.5, zorder=4)
            ax.add_patch(r)
            ax.text(x, ym_box + 0.08, label, ha="center", va="center",
                    fontsize=9, color=c, fontweight="bold", zorder=5)
            ax.text(x, ym_box - 0.22, val, ha="center", va="center",
                    fontsize=7, color="#9999bb", zorder=5)
            vwire(x, ym_box - bh/2, YR, c=c)
            ground_hash(x, c)
            dot(x, YM, c=c, r=0.08)

        def open_sym(x1, x2, lbl):
            xm = (x1 + x2) / 2
            g = 0.32
            wire(x1, xm - g)
            wire(xm + g, x2)
            ax.plot([xm - g, xm - g], [YM - 0.28, YM + 0.28], color=RED, lw=2.5)
            ax.plot([xm + g, xm + g], [YM - 0.28, YM + 0.28], color=RED, lw=2.5)
            ax.text(xm, YM + 0.48, lbl, ha="center", va="bottom",
                    fontsize=8.5, color=RED, style="italic", fontweight="bold")

        def vsource(x):
            ym = (YM + YR) / 2
            ax.add_patch(plt.Circle((x, ym), 0.34, ec=GOLD, fc="#12122a", lw=2.0, zorder=4))
            ax.text(x, ym + 0.06, "V₁", ha="center", va="center",
                    fontsize=10, color=GOLD, fontweight="bold", zorder=5)
            ax.text(x, ym - 0.20, "∠0°", ha="center", va="center",
                    fontsize=7.5, color=GOLD, zorder=5)
            vwire(x, ym + 0.34, YM, c=GOLD)
            dot(x, YM, c=GOLD, r=0.08)
            vwire(x, YR, ym - 0.34, c=GOLD)
            ground_hash(x, GOLD)

        XS     = 0.72   # Thevenin source / shunt
        XSTBUS = 2.20   # Substation Bus 24.9 kV
        XSRC   = 4.50   # SourceBus (T1 primary, after Xcel cable)
        X48    = 7.40   # Bus_480V
        X13S   = 10.90  # 13.8 kV Send
        X13R   = 14.10  # 13.8 kV Recv
        X63    = 16.60  # Bus_630V
        XINV   = 18.10  # Inverter

        if seq in ("1", "2"):
            if seq == "1":
                vsource(XS)
                wire(XS, XSTBUS)
                zbox(XS, XSTBUS, "Z₁_src", f"j{z1_src.imag:.4f}", GOLD)
            else:
                wire(XS, XSTBUS)
                shunt_box(XS + 0.60, "Z₂_src", f"j{z1_src.imag:.4f}", PURP)
                ax.text(XS + 0.60, YM + 0.62, "source Z₂\n(shunt to ground)",
                        ha="center", va="bottom", fontsize=7.5, color=PURP, style="italic")

            bus_bar(XSTBUS, "Substation Bus", "24.9 kV", color=TEAL)
            zbox(XSTBUS, XSRC,
                 "Z₁_xcbl" if seq == "1" else "Z₂_xcbl", fz(z1_xcbl), TEAL)
            bus_bar(XSRC, "SourceBus", "24.9 kV")
            zbox(XSRC, X48, "Z₁_T1" if seq == "1" else "Z₂_T1", fz(z1_t1), RED)
            bus_bar(X48, "Bus_480V", "0.48 kV")
            ax.text(X48, YM - 0.76, "Zig-Zag: inactive\n(open in Z₁, Z₂)",
                    ha="center", va="top", fontsize=8, color=DIM, style="italic")
            zbox(X48, X13S, "Z₁_T2" if seq == "1" else "Z₂_T2", fz(z1_t2), ORG)
            bus_bar(X13S, "13.8 kV  Send", "13.8 kV")
            zbox(X13S, X13R, "Z₁_line" if seq == "1" else "Z₂_line", fz(z1_line), WIRE)
            bus_bar(X13R, "13.8 kV  Recv", "13.8 kV")
            zbox(X13R, X63, "Z₁_T3" if seq == "1" else "Z₂_T3", fz(z1_t3), GRN)
            bus_bar(X63, "Bus_630V", "0.63 kV")
            wire(X63, XINV - 0.52)
            if seq == "1":
                ax.add_patch(plt.Circle((XINV - 0.26, YM), 0.30,
                                        ec=GOLD, fc="#12122a", lw=1.6, zorder=4))
                ax.text(XINV - 0.26, YM + 0.06, "I₁", ha="center", va="center",
                        fontsize=9.5, color=GOLD, fontweight="bold", zorder=5)
                ax.text(XINV - 0.26, YM - 0.20, "INV", ha="center", va="center",
                        fontsize=7, color=GOLD, zorder=5)
            else:
                ax.text(XINV - 0.50, YM, "  open  \n(INV)", ha="center", va="center",
                        fontsize=8.5, color=DIM,
                        bbox=dict(boxstyle="round,pad=0.28", fc="#12122a", ec=DIM, lw=1.0))

        else:  # zero sequence
            shunt_box(XS, "Z₀_src", f"j{z0_src.imag:.4f}", GOLD)
            ax.text(XS, YM + 0.62, "utility grounding\npath (higher Z)",
                    ha="center", va="bottom", fontsize=7.5, color=GOLD, style="italic")
            wire(XS, XSTBUS)
            bus_bar(XSTBUS, "Substation Bus", "24.9 kV", color=TEAL)
            zbox(XSTBUS, XSRC, "Z₀_xcbl  (≈Z₁)", fz(z0_xcbl), TEAL)
            bus_bar(XSRC, "SourceBus", "24.9 kV")

            zbox(XSRC, X48, "Z₀_T1  (Yg–Yg)", fz(z1_t1), RED)
            bus_bar(X48, "Bus_480V", "0.48 kV")

            x_zz = X48 + 0.80
            wire(X48, x_zz)
            dot(x_zz, YM, c=PURP)
            shunt_box(x_zz, "Z₀_ZZ  (Zig-Zag)", f"{z0_zz.real:.2f}+j{z0_zz.imag:.2f}", PURP)
            ax.text(x_zz, YM + 0.62, "preferred return path\n(designed for this!)",
                    ha="center", va="bottom", fontsize=7.5, color=PURP,
                    fontweight="bold", style="italic")

            x48r = X48 + 1.65
            wire(x_zz, x48r)
            zbox(x48r, X13S, "Z₀_T2  (Yg–Yg)", fz(z1_t2), ORG)
            bus_bar(X13S, "13.8 kV  Send", "13.8 kV")

            zbox(X13S, X13R, "Z₀_line  (concentric neutral)", fz(z0_line), WIRE)
            bus_bar(X13R, "13.8 kV  Recv", "13.8 kV")

            open_sym(X13R, X63, "OPEN — Δ primary blocks I₀")
            ax.text(X63, YM, "Bus_630V\n(isolated)", ha="center", va="center",
                    fontsize=8.5, color=DIM,
                    bbox=dict(boxstyle="round,pad=0.30", fc="#12122a", ec=DIM, lw=1.0))
            vwire(X63, YM - 0.44, YR + 0.60, c=DIM)
            ax.text(X63, YR + 0.54, "Z₀_T3→REF\n(Δ traps I₀, doesn't pass through)",
                    ha="center", va="top", fontsize=7.5, color=GRN, style="italic")

    return fig


# ── Plotly theme helper ───────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=50, b=60),
)

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("⚡ 9907 Queensburg St – Solar Power Quality")
st.caption(
    "DIA Solar Site · 2,250 kW AC · Three 750 kW Inverters · "
    "13.8 kV Private Distribution · Xcel Energy Secondary-Metered Service"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario")
    scen_label = st.selectbox("Solar production level", list(SCENARIOS.keys()))
    scen = SCENARIOS[scen_label]

    st.divider()
    st.subheader("System")
    st.metric("Total rated capacity",  "2,250 kW AC")
    st.metric("Inverters",             "3 × 750 kW")
    st.metric("Site transformer (T3)", "2,520 kVA  Δ–Y")
    st.metric("Step-up transformer (T2)", "2,350 kVA  Y–Y")
    st.metric("Feeder",                "1.94 mi · 13.8 kV")
    st.metric("Service transformer (T1)", "2,500 kVA  Y–Y")

    st.divider()
    st.caption("Applicable Standards")
    st.caption(f"IEEE 519 – Individual voltage harmonic ≤ {IEEE519_V}%")
    st.caption(f"IEEE 519 – Voltage THD ≤ {IEEE519_THD}%")
    st.caption(f"DG Tech Manual – Individual current harmonic ≤ {DG_I_LIMIT}%")

    if _dss_available():
        st.success("OpenDSS available – power flow is live")
    else:
        st.info("OpenDSS unavailable – using analytical estimates\n\n"
                "*(libdss_capi.dylib SIGILL in Streamlit threads on ARM64)*")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📐 One-Line Diagram",
    "⚡ Power Flow",
    "〜 Harmonic Spectrum",
    "📈 Impedance Scan",
    "🔄 Sequence Networks",
    "✅ Recommendations",
])

# ═══════════════ TAB 1 – One-Line ═════════════════════════════════════════════
with t1:
    fig_ol = draw_oneline(n_inv_active=scen["n_inv"])
    st.pyplot(fig_ol, width="stretch")
    plt.close(fig_ol)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**T1 – Xcel Service**")
        st.markdown("24.9 kV → 0.48 kV · 2,500 kVA · Wye–Wye")
        st.markdown("Secondary-metered; Xcel provides 0.48 kV directly.")
    with c2:
        st.markdown("**T2 – Customer Step-Up** ⚠️")
        st.markdown("0.48 kV → 13.8 kV · 2,350 kVA · **Wye–Wye**")
        st.markdown(
            "No delta winding — harmonics from the inverters have no local "
            "trap and propagate back through T1 to the billing CT. "
            "A Wye–Wye path also allows zero-sequence interaction with the Zig-Zag."
        )
    with c3:
        st.markdown("**T3 – Site Step-Down**")
        st.markdown("13.8 kV → 0.63 kV · 2,520 kVA · Delta–Wye")
        st.markdown(
            "Delta primary provides some harmonic trapping on the 13.8 kV side, "
            "but triplen currents that flow through T2 (Wye–Wye) bypass this."
        )

    st.warning(
        "**Root cause summary:** Three transformer leakage reactances in series between "
        "the inverters and the Xcel grid create high harmonic impedance — it rises "
        "roughly linearly with harmonic order. At 300 Hz (5th harmonic), the path is "
        "~5× more resistive than at 60 Hz. The Wye–Wye step-up (T2) provides no "
        "delta winding to circulate and damp harmonic currents locally."
    )

# ═══════════════ TAB 2 – Power Flow ══════════════════════════════════════════
with t2:
    st.subheader(f"Power Flow — {scen_label}")

    pf = run_dss(scen["pct"], scen["n_inv"])

    if pf and "error" not in pf:
        # ── Live OpenDSS results
        st.success("OpenDSS converged" if pf.get("ok") else "⚠️ Did not converge")
        bus_df = (pd.DataFrame({"Bus": pf["buses"], "V (pu)": pf["vpu"]})
                  .query("`V (pu)` > 0.05"))
        col1, col2 = st.columns(2)
        with col1:
            colors = ["#e05c5c" if v < 0.95 or v > 1.05 else "#4a9eff"
                      for v in bus_df["V (pu)"]]
            fig = go.Figure(go.Bar(x=bus_df["Bus"], y=bus_df["V (pu)"],
                                   marker_color=colors))
            fig.add_hline(y=0.95, line_dash="dash", line_color="#ffaa00")
            fig.add_hline(y=1.05, line_dash="dash", line_color="#ffaa00")
            fig.update_layout(title="Bus Voltages (p.u.)", height=370,
                              xaxis_tickangle=-40, **LAYOUT_BASE)
            st.plotly_chart(fig, width="stretch")
        with col2:
            if pf["xfmrs"]:
                xf = pd.DataFrame(pf["xfmrs"])
                colors = ["#e05c5c" if l > 100 else "#f5c542" if l > 80 else "#4fc978"
                          for l in xf["Loading_pct"]]
                fig = go.Figure(go.Bar(x=xf["Transformer"], y=xf["Loading_pct"],
                                       marker_color=colors,
                                       text=[f"{l:.1f}%" for l in xf["Loading_pct"]],
                                       textposition="outside"))
                fig.add_hline(y=100, line_dash="dash", line_color="#e05c5c")
                fig.update_layout(title="Transformer Loading (%)", height=370,
                                  **LAYOUT_BASE)
                st.plotly_chart(fig, width="stretch")
    else:
        # ── Analytical estimates
        if pf and "error" in pf:
            st.warning(f"OpenDSS: {pf['error']} — using analytical estimates.")
        else:
            st.info("Analytical estimates (install `opendssdirect` for live DSS results).")

        vd = analytical_voltages(scen["pct"])
        col1, col2 = st.columns(2)
        with col1:
            colors = ["#e05c5c" if v < 0.95 or v > 1.05 else "#4a9eff"
                      for v in vd.values()]
            fig = go.Figure(go.Bar(
                x=list(vd.keys()), y=list(vd.values()),
                marker_color=colors,
                text=[f"{v:.4f}" for v in vd.values()],
                textposition="outside",
            ))
            fig.add_hline(y=0.95, line_dash="dash", line_color="#ffaa00",
                          annotation_text="0.95 pu")
            fig.add_hline(y=1.05, line_dash="dash", line_color="#ffaa00",
                          annotation_text="1.05 pu")
            fig.update_layout(title="Bus Voltages — Estimated (p.u.)",
                              yaxis=dict(range=[0.93, 1.08]), height=370,
                              xaxis_tickangle=-20, **LAYOUT_BASE)
            st.plotly_chart(fig, width="stretch")

        with col2:
            xf = analytical_xfmr_loading(scen["kw"])
            colors = ["#e05c5c" if r["Loading"] > 100
                      else "#f5c542" if r["Loading"] > 80 else "#4fc978"
                      for r in xf]
            fig = go.Figure(go.Bar(
                x=[r["Name"] for r in xf],
                y=[r["Loading"] for r in xf],
                marker_color=colors,
                text=[f"{r['Loading']:.1f}%" for r in xf],
                textposition="outside",
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="#e05c5c",
                          annotation_text="Rated")
            fig.update_layout(title="Transformer Loading — Estimated (%)",
                              yaxis=dict(range=[0, 115]), height=370,
                              xaxis_tickangle=-10, **LAYOUT_BASE)
            st.plotly_chart(fig, width="stretch")

    # ── Key metrics row
    st.divider()
    i_fund = scen["kw"] / (np.sqrt(3) * SYS["dia7_kv"]) if scen["kw"] > 0 else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Solar output",           f"{scen['kw']:,} kW")
    m2.metric("Active inverters",        str(scen["n_inv"]))
    m3.metric("Fundamental current\nat 0.48 kV CT", f"{i_fund:.0f} A")
    m4.metric("T2 step-up loading",
              f"{100 * scen['kw'] * 0.95 / SYS['t2_kva']:.1f} %")

# ═══════════════ TAB 3 – Harmonic Spectrum ═══════════════════════════════════
with t3:
    # ── Resonant amplification (from impedance model) ─────────────────────────
    Z1_mag = abs(z_at_bus480(1))
    amp_factors = {h: abs(z_at_bus480(h)) / Z1_mag for h in HARM_ORDERS}

    # Harmonic pct at inverter output (what is injected)
    pct_injected = {h: INJECTED[h] for h in HARM_ORDERS}   # % of fundamental

    # Estimated harmonic pct at the CT after resonant amplification
    pct_at_ct = {h: pct_injected[h] * amp_factors[h] for h in HARM_ORDERS}

    # Xcel report measured values at CT (100% production) from the power quality report
    XCEL_MEASURED = {3: 2.0, 5: 47.0, 7: 8.0, 9: 1.0, 11: 3.0, 13: 1.5}

    st.subheader("Harmonic Current Spectrum — Two-Step View")
    st.markdown(
        """
The DG Tech Manual's 10% limit applies to harmonic currents **at the point of common coupling** (the billing CT).
There are two distinct values to track:

| | |
|---|---|
| **Injected** | What the inverters actually put into the circuit (typically 2–5% at h=5 for modern inverters) |
| **At CT (amplified)** | What the CT measures after those currents travel through the high-impedance transformer chain |

The stacked transformers (T1 + T2 + T3, each 5.73% reactance) present an impedance that rises with harmonic order.
At h=5 (300 Hz), the transformer reactances are **5× higher** than at 60 Hz, so the same injected current
produces 5× more harmonic voltage — and the CT sees a correspondingly amplified current.
        """
    )

    pct = scen["pct"]
    hs = [h for h in HARM_ORDERS if h > 1]
    labels_h = [f"h={h} ({h*60} Hz)" for h in hs]
    colors_inj = ["#4a9eff"] * len(hs)
    colors_ct  = ["#e05c5c" if pct_at_ct[h] > DG_I_LIMIT else "#f5c542" for h in hs]

    fig_h = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"① Harmonic current injected by inverters  ({scen_label})",
            f"② Harmonic current estimated at 0.48 kV billing CT",
        ),
        y_title="Harmonic current  (% of 60 Hz fundamental current I₁)",
    )
    fig_h.add_trace(go.Bar(
        x=labels_h,
        y=[pct_injected[h] for h in hs],
        marker_color=colors_inj,
        text=[f"{pct_injected[h]:.1f}%" for h in hs],
        textposition="outside",
        name="Injected",
    ), row=1, col=1)
    fig_h.add_hline(y=DG_I_LIMIT, line_dash="dash", line_color="#e05c5c",
                    annotation_text="DG Tech Manual limit: 10% of I₁",
                    annotation_position="top right",
                    row=1, col=1)

    fig_h.add_trace(go.Bar(
        x=labels_h,
        y=[pct_at_ct[h] for h in hs],
        marker_color=colors_ct,
        text=[f"{pct_at_ct[h]:.1f}%" for h in hs],
        textposition="outside",
        name="At CT",
    ), row=1, col=2)
    fig_h.add_hline(y=DG_I_LIMIT, line_dash="dash", line_color="#e05c5c",
                    annotation_text="DG Tech Manual limit: 10% of I₁",
                    annotation_position="top right",
                    row=1, col=2)

    fig_h.update_layout(height=450, showlegend=False, **LAYOUT_BASE)
    fig_h.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_h, width="stretch")

    # ── Amplification factor cards ────────────────────────────────────────────
    st.markdown("**Resonant amplification factors** (from impedance model — ratio of Z at harmonic to Z at 60 Hz):")
    cols_af = st.columns(len(hs))
    for col, h in zip(cols_af, hs):
        af = amp_factors[h]
        col.metric(
            f"h={h}  ({h*60} Hz)",
            f"{af:.1f}×",
            delta=f"→ {pct_injected[h]*af:.0f}% at CT" if pct > 0 else None,
            delta_color="inverse",
        )

    # ── Cross-scenario summary table ──────────────────────────────────────────
    st.divider()
    st.subheader("Summary: Injected vs. At CT Across All Scenarios")

    rows = []
    for lbl, s in SCENARIOS.items():
        ph_inj = {h: INJECTED[h] for h in hs} if s["pct"] > 0 else {h: 0.0 for h in hs}
        ph_ct  = {h: ph_inj[h] * amp_factors[h] for h in hs}
        thd_ct = np.sqrt(sum(ph_ct[h]**2 for h in hs))
        rows.append(dict(
            Scenario=lbl,
            kW=s["kw"],
            **{f"h{h} inj.": round(ph_inj[h], 1) for h in hs},
            **{f"h{h} CT": round(ph_ct[h], 1) for h in hs},
            **{f"h{h}_ok": ph_ct[h] <= DG_I_LIMIT for h in hs},
            **{"THD@CT": round(thd_ct, 1)},
        ))

    # Xcel report row — use None for unknown injected values (renders as blank)
    rows.append(dict(
        Scenario="Xcel measured (100%, report)",
        kW=2250,
        **{f"h{h} inj.": None for h in hs},
        **{f"h{h} CT": float(XCEL_MEASURED.get(h, float("nan"))) for h in hs},
        **{f"h{h}_ok": XCEL_MEASURED.get(h, 0) <= DG_I_LIMIT for h in hs},
        **{"THD@CT": round(np.sqrt(sum(v**2 for v in XCEL_MEASURED.values())), 1)},
    ))

    df_sum = pd.DataFrame(rows)
    ok_cols  = [c for c in df_sum.columns if c.endswith("_ok")]
    show_cols = [c for c in df_sum.columns if not c.endswith("_ok")]

    def highlight_violations(row):
        styles = [""] * len(row)
        for i, col in enumerate(row.index):
            if " CT" in col or col == "THD@CT":
                val = row[col]
                if isinstance(val, (int, float)) and val > DG_I_LIMIT:
                    styles[i] = "background-color:#3a1010; color:#ff8888"
        return styles

    st.dataframe(
        df_sum[show_cols].style.apply(highlight_violations, axis=1),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "**Injected (%)**: harmonic current as % of fundamental at inverter terminals (modern inverter spec, ~4% at h=5).  "
        "**At CT (%)**: estimated after amplification through the T1+T2+T3 transformer impedance chain.  "
        "**Xcel measured**: values digitized from the November 2023 power quality report; h=5 was 47% at CT — "
        "nearly 5× the DG Tech Manual's 10% individual harmonic limit."
    )

# ═══════════════ TAB 4 – Impedance Scan ══════════════════════════════════════
with t4:
    st.subheader("Driving-Point Impedance at Bus_480V  vs  Harmonic Order")
    st.markdown(
        "Injecting 1 A at Bus_480V and measuring the resulting voltage gives the "
        "impedance **Z(h)** at that bus. Larger Z → more harmonic voltage for the "
        "same injected current. A **resonant peak** (sharp spike) would indicate a "
        "classic LC resonance; a **monotonically rising** curve indicates inductive "
        "dominance from the stacked transformers."
    )

    h_arr  = np.arange(1, 26)
    z_pu   = [z_at_bus480(h) for h in h_arr]
    z_ohm  = [abs(z) * z_base_480() for z in z_pu]
    z_ang  = [np.degrees(np.angle(z)) for z in z_pu]

    bar_c  = ["#e05c5c" if h == 5 else "#f5c542" if h in [3,7,9] else "#4a9eff"
               for h in h_arr]

    fig_z = make_subplots(
        rows=2, cols=1,
        subplot_titles=("|Z| at Bus_480V (Ω)", "Impedance Angle (degrees)"),
        vertical_spacing=0.15,
        row_heights=[0.65, 0.35],
    )

    fig_z.add_trace(go.Bar(
        x=[f"h={h}" for h in h_arr],
        y=z_ohm,
        marker_color=bar_c,
        name="|Z| (Ω)",
        hovertemplate="h=%{x}<br>|Z| = %{y:.5f} Ω<extra></extra>",
    ), row=1, col=1)

    fig_z.add_trace(go.Scatter(
        x=[f"h={h}" for h in h_arr],
        y=z_ang,
        mode="lines+markers",
        line=dict(color="#bb88ff", width=2),
        marker=dict(size=5),
        name="Angle (°)",
    ), row=2, col=1)

    fig_z.add_hline(y=90, line_dash="dot", line_color="#555577",
                    annotation_text="Pure inductive (+90°)", row=2, col=1)
    fig_z.add_hline(y=-90, line_dash="dot", line_color="#555577",
                    annotation_text="Pure capacitive (−90°)", row=2, col=1)

    fig_z.update_layout(height=560, showlegend=False, **LAYOUT_BASE)
    fig_z.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_z, width="stretch")

    # ── Key metrics
    z1 = z_ohm[0]; z5 = z_ohm[4]
    peak_h = int(h_arr[np.argmax(z_ohm)])
    peak_z = max(z_ohm)

    m1, m2, m3 = st.columns(3)
    m1.metric("|Z| at h=1  (60 Hz)",  f"{z1:.5f} Ω")
    m2.metric("|Z| at h=5  (300 Hz)", f"{z5:.5f} Ω",
              delta=f"{z5/z1:.1f}× fundamental", delta_color="inverse")
    m3.metric(f"Peak |Z| at h={peak_h}  ({peak_h*60} Hz)", f"{peak_z:.5f} Ω")

    st.divider()
    st.subheader("What This Means")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Why impedance rises with harmonic order**")
        st.markdown(
            "Each transformer's leakage reactance scales linearly with frequency: "
            "`X(h) = h × X₆₀`. Three transformers in series means the total "
            "inductive path between the inverter and the Xcel grid is "
            "approximately `h × (X_T1 + X_T2 + X_T3 + X_line)`. "
            "At h=5 this is ~5× the fundamental impedance, causing "
            "5× more harmonic *voltage* distortion at Bus_480V for the same "
            "harmonic *current* injection."
        )
    with col2:
        st.markdown("**What a resonance would look like**")
        st.markdown(
            "A classic parallel LC resonance shows as a sharp spike in |Z| "
            "at one specific harmonic order, where inductive reactance "
            "exactly cancels capacitive susceptance. "
            "The model now uses **underground XLPE cable** throughout: "
            "380 nF/mi for the 13.8 kV customer feeder (~737 nF total) "
            "and 290 nF/mi for the Xcel 24.9 kV section (~580 nF total). "
            "That is ~68× more capacitance than the overhead ACSR assumption. "
            "The resonant peak this creates falls at h ≈ 35–42 — well above the "
            "harmonic orders of concern (h=5, 7). At h=5 the capacitance instead "
            "provides a parallel return path that slightly *reduces* |Z| compared "
            "to the overhead model. Power-factor correction capacitor banks (not "
            "present here) or much shorter cable runs would push resonance down "
            "into the h=5–13 danger zone."
        )

    st.info(
        "**To force a resonance in the model:** add a capacitor bank at Bus_480V "
        "or Bus_630V with `New Capacitor.PFC bus1=Bus_480V kv=0.48 kvar=500`, "
        "then sweep harmonic order with the `Isource.ZProbe` in Scenario 5 of "
        "the DSS file. Watch for a spike in the monitor output near h=5."
    )

# ═══════════════ TAB 5 – Sequence Networks ═══════════════════════════════════
with t5:
    st.subheader("Symmetrical Components — Sequence Networks")
    st.markdown(
        """
**What are symmetrical components?**
Any unbalanced three-phase quantity (voltage, current) can be decomposed into three
balanced sets called *sequences*. This decomposition is powerful because each sequence
sees a different circuit — the sequence networks. You solve three simple networks instead
of one complex unbalanced one.

| Sequence | Symbol | What it represents |
|---|---|---|
| **Positive (Z₁)** | +, 1 | Normal balanced ABC-rotation currents; this is what every meter reads in normal operation |
| **Negative (Z₂)** | −, 2 | Reverse-rotation (ACB) currents; appear under unbalanced faults or unbalanced loads |
| **Zero (Z₀)** | 0 | All three phases in phase with each other; these are **ground currents** — they only exist if a neutral path is available |

For transformers, the winding **connection** determines whether zero-sequence current can
pass through. This is the most important thing to understand about the diagram below:

- **Wye-grounded / Wye-grounded** (T1, T2): zero sequence **passes through** — the neutral
  provides a return path on both sides.
- **Delta / Wye-grounded** (T3): the delta side is an **open circuit** to zero sequence.
  Current cannot enter or leave through a delta winding. The delta *does* allow zero-sequence
  current to *circulate inside* it (as a closed loop), but that current never leaves — the delta traps it.
- **Zig-zag grounding transformer**: specifically designed to be a **low-impedance path for
  zero-sequence current**. In positive and negative sequence it is essentially invisible.
  In zero sequence it is the preferred route back to ground.
        """
    )

    fig_seq = draw_sequence_networks()
    st.pyplot(fig_seq, width="stretch")
    plt.close(fig_seq)

    # ── Per-unit impedance table ───────────────────────────────────────────────
    st.divider()
    st.subheader("Sequence Impedances — Per-Unit on 2,500 kVA Base")

    Zb3_tab = 13.8**2 * 1e3 / 2500
    _z1_src  = complex(0,       2500 / (500 * 1e3))
    _z0_src  = complex(0, 3*2500/(400*1e3) - 2*2500/(500*1e3))
    _z1_t1   = complex(0.00500, 0.05730)
    _z1_t2   = complex(0.00500 * 2500/2350, 0.05730 * 2500/2350)
    _z1_line = complex(SYS["r1_per_mi"]*SYS["line_mi"]/Zb3_tab,
                       SYS["x1_per_mi"]*SYS["line_mi"]/Zb3_tab)
    _z0_line = complex(0.11*SYS["line_mi"]/Zb3_tab, 0.17*SYS["line_mi"]/Zb3_tab)
    _z1_t3   = complex(0.00500 * 2500/2520, 0.05730 * 2500/2520)
    _z0_zz   = complex(0.00500 * 2500/250,  0.05000 * 2500/250)

    def _fzpu(z):
        return f"{z.real:.4f}+j{z.imag:.4f}"

    seq_table = pd.DataFrame([
        dict(Element="Xcel source",
             Connection="3φ grounded Wye",
             Z1_pu=_fzpu(_z1_src),
             Z2_pu="= Z₁",
             Z0_pu=_fzpu(_z0_src),
             Z0_note="Z₀ > Z₁ (asymmetric fault duty)"),
        dict(Element="T1 – Xcel service\n2,500 kVA",
             Connection="Wye-g / Wye-g",
             Z1_pu=_fzpu(_z1_t1),
             Z2_pu="= Z₁",
             Z0_pu="= Z₁",
             Z0_note="Passes I₀ — both neutrals grounded"),
        dict(Element="Zig-Zag\n250 kVA",
             Connection="Zig-Zag (0.48 kV)",
             Z1_pu="≈ open",
             Z2_pu="≈ open",
             Z0_pu=_fzpu(_z0_zz),
             Z0_note="Primary purpose: low-Z I₀ return path"),
        dict(Element="T2 – Step-up  ⚠\n2,350 kVA",
             Connection="Wye-g / Wye-g",
             Z1_pu=_fzpu(_z1_t2),
             Z2_pu="= Z₁",
             Z0_pu="= Z₁",
             Z0_note="Passes I₀ — both neutrals grounded"),
        dict(Element="Feeder\n1.94 mi XLPE UG",
             Connection="Underground, concentric neutral",
             Z1_pu=_fzpu(_z1_line),
             Z2_pu="= Z₁",
             Z0_pu=_fzpu(_z0_line),
             Z0_note="Z₀ ≈ Z₁ (concentric neutral; no earth return penalty)"),
        dict(Element="T3 – Site step-down\n2,520 kVA",
             Connection="Delta / Wye-g",
             Z1_pu=_fzpu(_z1_t3),
             Z2_pu="= Z₁",
             Z0_pu="OPEN (13.8 kV side)",
             Z0_note="Δ blocks I₀; 0.63 kV wye neutral trapped in Δ"),
        dict(Element="Inverters\n3 × 750 kW",
             Connection="0.63 kV PV systems",
             Z1_pu="current source",
             Z2_pu="open",
             Z0_pu="open",
             Z0_note="Grid-following inverters: no I₀ contribution"),
    ])
    seq_table.columns = ["Element", "Connection", "Z₁ (pu)", "Z₂ (pu)", "Z₀ (pu)", "Zero-seq note"]
    st.dataframe(seq_table, hide_index=True, width="stretch")

    # ── Key takeaways ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("What Each Sequence Network Tells You")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Positive Sequence (Z₁)**")
        st.markdown(
            "This is the network for normal operations and three-phase faults. "
            "The inverters appear as current sources injecting into Bus_630V. "
            "Current flows: INV → T3 → feeder → T2 → T1 → Xcel grid. "
            "All three transformers add reactance in series — this is why harmonics "
            "are amplified (X scales with frequency, so Z₁ at h=5 is 5× Z₁ at h=1)."
        )
        st.info("Three-phase fault current is determined by Z₁ alone.")

    with c2:
        st.markdown("**Negative Sequence (Z₂)**")
        st.markdown(
            "Identical impedances to Z₁ (no rotating machines), but no voltage source. "
            "Negative-sequence currents appear during line-to-line or single-line-to-ground "
            "faults. Since Z₂ = Z₁ here, the Thevenin equivalent for an SLG fault has "
            "Z₁ in series with Z₂ in series with Z₀ — the zero-sequence path determines "
            "how the three networks interconnect. "
            "Inverters do not contribute negative-sequence current (they trip on unbalance)."
        )
        st.info("Z₂ = Z₁ when there are no synchronous generators.")

    with c3:
        st.markdown("**Zero Sequence (Z₀) — the unusual one**")
        st.markdown(
            "The topology changes completely. The Wye-Wye path through T2 means a "
            "13.8 kV ground fault sends zero-sequence current back through T2 to the "
            "0.48 kV bus. From there it splits two ways: "
            "(1) through the **Zig-Zag** (low Z — preferred path), and "
            "(2) through T1 back to Xcel (higher Z). "
            "The **T3 delta** completely blocks zero-sequence current from the solar side — "
            "the inverters cannot contribute to ground faults on the 13.8 kV feeder. "
            "This is good for protection, but the Wye-Wye T2 (no delta) means the "
            "0.48 kV utility metering bus IS exposed to 13.8 kV ground fault current."
        )
        st.warning(
            "**T2 Wye-Wye creates two distinct problems:** "
            "(1) Ground faults on the 13.8 kV feeder send zero-sequence current back through "
            "T2 to the 0.48 kV billing bus — changing T2 to **Wye-Delta** (HV side delta) would "
            "block this. "
            "(2) h=5 and h=7 are positive/negative sequence and pass through ANY winding "
            "configuration — the winding change does not fix the harmonic amplification problem. "
            "A passive harmonic filter is required for that."
        )

# ═══════════════ TAB 6 – Recommendations ══════════════════════════════════════
with t6:
    st.header("Recommendations — 9907 Queensburg St Solar")
    st.markdown(
        "Based on the November 2023 power quality study and circuit impedance analysis. "
        "Recommendations are ordered by priority."
    )

    # ── Measurement context ────────────────────────────────────────────────────
    st.subheader("What the PQ Study Found")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("h5 Voltage (Vcn avg)", "4.3 Vac", "1.55% of fundamental",
              delta_color="off")
    m2.metric("Voltage THD at DIA7 bus", "2.0% avg / 2.4% max",
              "IEEE 519 limit: 5.0%", delta_color="off")
    m3.metric("Dominant harmonic", "h=5 (300 Hz)", "as predicted by model",
              delta_color="off")
    m4.metric("Measurement point", "0.48 kV DIA7 billing bus",
              "lowest-Z point in circuit", delta_color="off")

    st.info(
        "**Interpretation:** Voltage THD at the billing meter is within IEEE 519 limits (2.4% max "
        "vs. 5.0% limit). However, the DIA7 bus is the lowest-impedance point in the circuit — "
        "immediately downstream of T1 and the utility source. Harmonic voltage rises with "
        "distance from the source. The 13.8 kV bus (between T2 and the feeder) was not "
        "monitored and is expected to show significantly higher distortion. Equipment stress "
        "on T2 and connected loads on the 13.8 kV side is the primary concern, not billing-point "
        "IEEE 519 compliance."
    )

    st.divider()

    # ── Recommendation 1 ──────────────────────────────────────────────────────
    with st.expander("🔴 Priority 1 — Measure harmonics at the 13.8 kV bus", expanded=True):
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("**Action:** Install a temporary power quality monitor on the 13.8 kV bus "
                        "between T2 and the customer's MV feeder (or at T2 13.8 kV terminals).")
            st.markdown("**Why this matters:**")
            st.markdown(
                "- The 2023 study measured only at the 0.48 kV billing point — the best-case "
                "location for harmonic voltage because it's closest to the utility source\n"
                "- The harmonic impedance model predicts **~5× amplification** of h=5 current "
                "between the source and the 13.8 kV bus due to T1 + T2 + T3 leakage reactances "
                "stacking in series at 300 Hz\n"
                "- Until 13.8 kV measurements are taken, the actual severity of the problem "
                "is unknown; a filter may or may not be required\n"
                "- This measurement also establishes a pre-mitigation baseline"
            )
            st.markdown("**IEEE 519-2022 voltage THD limits at 13.8 kV:** 5.0% (1–69 kV)")
        with col_r:
            st.markdown("**Estimated cost:** Low (meter rental + labor)")
            st.markdown("**Timeline:** Can be completed in 1–2 weeks")
            st.markdown("**Owner:** Customer + Xcel engineering")
            st.success("Do this first — it determines whether mitigation is required "
                       "and what size filter to specify.")

    # ── Recommendation 2 ──────────────────────────────────────────────────────
    with st.expander("🔴 Priority 2 — Install passive 5th harmonic filter at DIA7 0.48 kV bus",
                     expanded=True):
        col_l, col_r = st.columns([3, 2])

        # Compute filter sizing from SYS dict
        _h_tune = 4.7          # detuned below h=5 to avoid capacitor overload
        _q_filter_kvar = 500   # rough sizing; exact value from 13.8 kV measurements

        with col_l:
            st.markdown("**Action:** Install a detuned passive harmonic filter "
                        f"(tuned to h≈{_h_tune}, i.e., ~{_h_tune*60:.0f} Hz) at the 0.48 kV "
                        "DIA7 bus, sized after 13.8 kV measurements confirm the harmonic "
                        "current magnitude.")
            st.markdown("**How it works:**")
            st.markdown(
                "- A passive filter is a series LC circuit: capacitor bank + series reactor\n"
                "- Tuned to h=4.7 (not exactly h=5) to present near-zero impedance at h=5 "
                "without risk of capacitor resonance with the system at exactly 300 Hz\n"
                "- Harmonic current from the inverters finds a low-impedance path to ground "
                "through the filter instead of circulating through T2 and the 13.8 kV feeder\n"
                "- The amplification mechanism is broken: Z_480(h=5) drops from ~5× to near zero\n"
                "- Filter also provides fundamental-frequency VAR support (improves power factor)"
            )
            st.markdown("**What this fixes:** h=5 (and to a lesser extent h=7) harmonic amplification")
            st.markdown("**What this does not fix:** Zero-sequence fault current exposure on the "
                        "0.48 kV bus — that requires the T2 winding change (Priority 3)")
        with col_r:
            st.markdown(f"**Rough filter sizing:** ~{_q_filter_kvar} kVAR at 0.48 kV")
            st.markdown("**Estimated cost:** $30,000–$80,000 installed")
            st.markdown("**Timeline:** 3–6 months (design, procure, install)")
            st.markdown("**Owner:** Customer (per Xcel DG Tech Manual §3.2)")
            st.warning("Size after 13.8 kV measurements. Over-sizing the filter creates its own "
                       "resonance risk.")

    # ── Recommendation 3 ──────────────────────────────────────────────────────
    with st.expander("🟡 Priority 3 — Change T2 to Wye-Delta (0.48 kV Wye / 13.8 kV Delta)",
                     expanded=False):
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("**Action:** Replace or rewind T2 (0.48/13.8 kV, currently Yg-Yg) so the "
                        "13.8 kV winding is Delta-connected. The 0.48 kV winding stays Wye-grounded "
                        "to preserve the utility metering connection.")
            st.markdown("**What this fixes — zero-sequence fault isolation:**")
            st.markdown(
                "- A single-line-to-ground fault on the 13.8 kV feeder currently sends "
                "zero-sequence current back through T2 to the 0.48 kV billing bus\n"
                "- A delta winding on the 13.8 kV side circulates that zero-sequence current "
                "internally in the delta and blocks it from passing through to the 0.48 kV side\n"
                "- This protects the billing CT, metering equipment, and T1 from participating "
                "in ground faults on the customer's 13.8 kV feeder"
            )
            st.markdown("**What this does NOT fix:**")
            st.markdown(
                "- h=5 and h=7 are **positive and negative sequence** — they pass through "
                "any winding configuration. The delta only traps zero-sequence (triplen) harmonics. "
                "The harmonic amplification problem requires a filter regardless of winding."
            )
            st.markdown("**Note:** The Xcel DG Tech Manual (§10, §17) does not require a specific "
                        "winding configuration for step-up transformers beyond effective grounding. "
                        "Yg-Yg is listed as 'common.' This is a gap in the standard.")
        with col_r:
            st.markdown("**Estimated cost:** $50,000–$150,000 (transformer replacement)")
            st.markdown("**Timeline:** 6–12 months")
            st.markdown("**Owner:** Customer")
            st.info("Lower urgency than the filter — the zero-sequence exposure exists today "
                    "and has not caused a documented incident. Prioritize filter first.")

    # ── Recommendation 4 ──────────────────────────────────────────────────────
    with st.expander("🟢 Priority 4 — Flag DG Tech Manual gap to Xcel interconnection engineering",
                     expanded=False):
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("**Action:** Submit a written finding to Xcel interconnection engineering "
                        "that the DG Tech Manual does not address transformer winding configuration "
                        "for harmonic amplification management in multi-transformer solar "
                        "interconnections.")
            st.markdown("**The gap:**")
            st.markdown(
                "- The DG Tech Manual evaluates transformer windings only for grounding/zero-sequence "
                "adequacy (§10, §17). It calls Yg-Yg 'common' for step-up transformers.\n"
                "- There is no guidance on how stacked transformer leakage reactances amplify "
                "inverter harmonic currents, or how winding configuration (or underground cable "
                "capacitance) affects resonant frequency\n"
                "- An interconnection engineer following the manual to the letter would design "
                "exactly the 9907 Queensburg circuit and be fully compliant — and still produce "
                "a harmonic amplification problem that the manual never flagged\n"
                "- The manual does require IEEE 519 / IEEE 1547 compliance (§3.2), but gives no "
                "guidance on how to achieve it for this circuit topology"
            )
            st.markdown("**Why this matters beyond this site:** Any future solar DG with "
                        "a similar three-transformer topology (step-up → customer MV feeder → "
                        "another step-up) faces the same risk.")
        with col_r:
            st.markdown("**Estimated cost:** Engineering time only")
            st.markdown("**Timeline:** Can be done immediately")
            st.markdown("**Owner:** Xcel protection/power quality engineering")
            st.success("Low effort, high leverage for future projects.")

    st.divider()
    st.subheader("Policy & Rate Design Implications")
    st.markdown(
        "The 9907 Queensburg topology was driven by rate economics, not technical necessity. "
        "Understanding the financial incentive is required to design a regulatory response."
    )

    # ── Rate differential analysis ─────────────────────────────────────────────
    with st.expander("💰 Rate Differential: Net Metering vs. Community Solar (Schedule NM vs. SRCS)",
                     expanded=True):
        # Key tariff constants (PSCo Electric Tariff, effective Nov 2025)
        SRCS_SG        = 0.07891   # $/kWh — SRCS 2026 credit for SG customers
        AHIC_2025      = 0.06511   # $/kWh — AHIC excess energy cashout for SG (2025)
        AHIC_2024      = 0.07597
        AHIC_2023      = 0.07437
        SG_DIST_D      = 9.36      # $/kW-month — distribution demand charge
        SG_GT_D_SUM    = 15.14     # $/kW-month — G&T demand, summer (Jun-Sep)
        SG_GT_D_WIN    = 9.11      # $/kW-month — G&T demand, winter (Oct-May)
        PLANT_KW       = 2250
        CF             = 0.25      # Colorado capacity factor
        ANN_MWH        = PLANT_KW * CF * 8760 / 1000  # MWh/year

        srcs_rev = ANN_MWH * 1000 * SRCS_SG

        # ── THIS SITE: primarily export, ~300 kVA original load ──
        # PQ study (page 9): avg net power = -18 kW (exporting), avg import current ~27A at 0.48 kV
        # Original service transformer was 300 kVA → building own load ≈ 0-50 kW max
        # Building peak demand occurs at night (no solar) → solar cannot reduce billing demand peak
        # Therefore: demand reduction ≈ $0 at this billing meter
        nm_ahic_site   = ANN_MWH * 1000 * AHIC_2025   # essentially all production cashed at AHIC
        nm_site_gap    = nm_ahic_site - srcs_rev        # negative = NM earns LESS than SRCS here

        # ── OS-NM SCENARIO: credits applied to a high-demand portfolio meter elsewhere ──
        # Under Schedule OS-NM, NM credits from this plant can offset bills at other
        # customer-owned meters. If applied to a large commercial load with high demand:
        osmn_demand_kw = 2000     # kW of demand reduction at the benefiting meter
        osmn_energy_mwh = ANN_MWH * 0.80  # 80% of production offsets load at the other meter
        osmn_excess_mwh = ANN_MWH * 0.20
        osmn_energy_val = osmn_energy_mwh * 1000 * SRCS_SG   # offset at full retail energy
        osmn_demand_val = (osmn_demand_kw * (SG_DIST_D + SG_GT_D_SUM) * 4 +
                           osmn_demand_kw * (SG_DIST_D + SG_GT_D_WIN) * 8)
        osmn_ahic_val   = osmn_excess_mwh * 1000 * AHIC_2025
        osmn_total      = osmn_energy_val + osmn_demand_val + osmn_ahic_val
        osmn_gap        = osmn_total - srcs_rev

        st.markdown(
            f"**Annual generation estimate:** {ANN_MWH:,.0f} MWh "
            f"({PLANT_KW:,} kW × {CF:.0%} capacity factor × 8,760 hr) &nbsp;|&nbsp; "
            f"**SRCS baseline (SG, 2026):** ${SRCS_SG:.5f}/kWh → **${srcs_rev:,.0f}/year**"
        )

        st.markdown("---")
        st.markdown("#### At the 9907 Queensburg billing meter (DIA7 — Denver International Airport)")
        st.markdown(
            "This site is on Denver International Airport property. 'DIA7' is one of DIA's "
            "many Xcel billing meters — this one originally served a small equipment outbuilding "
            "with ~**300 kVA** of service. DIA consumes roughly **250,000–350,000 MWh/year** "
            "across 30+ billing meters (terminal, concourses A/B/C/E, people mover, baggage, etc.). "
            "This 2,250 kW plant generates ~4,900 MWh/year — under 2% of DIA's total consumption.\n\n"
            "The PQ study confirms near-zero net import at this meter (avg -18 kW, exporting). "
            "The billing demand peak at DIA7 occurs at night when solar is off — "
            "solar **cannot reduce the demand peak at this meter**. "
            "Demand savings at DIA7 alone are effectively zero."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("NM year-end AHIC cashout (2025)", f"${nm_ahic_site:,.0f}/yr",
                    f"${AHIC_2025:.5f}/kWh", delta_color="off")
        col2.metric("SRCS revenue", f"${srcs_rev:,.0f}/yr",
                    f"${SRCS_SG:.5f}/kWh", delta_color="off")
        col3.metric("NM vs. SRCS at this meter", f"${nm_site_gap:,.0f}/yr",
                    "NM earns LESS than SRCS here", delta_color="inverse")

        st.error(
            f"At the 9907 Queensburg billing meter in isolation, **NM earns ~${abs(nm_site_gap):,.0f}/year "
            f"LESS than SRCS** (AHIC ${AHIC_2025:.5f} vs. SRCS ${SRCS_SG:.5f}/kWh). "
            "The rate arbitrage that motivated this topology does NOT materialize at this meter alone. "
            "The financial logic only works if NM credits are applied to a high-demand load "
            "at a different location under **Schedule OS-NM (Off-Site Net Metering)**."
        )

        st.markdown("---")
        st.markdown("#### Where the money actually is — DIA's meter portfolio via OS-NM (Schedule OS-NM)")
        st.markdown(
            "Schedule OS-NM allows NM credits from this plant to offset bills at **other DIA "
            "billing meters** — the main terminal, concourses, people mover, baggage halls. "
            "These meters have massive, continuous demand loads. DIA designates which meters "
            "receive the credits, targeting the highest-demand accounts for maximum value.\n\n"
            "If applied to a high-demand DIA concourse or terminal meter:"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**OS-NM applied to high-demand portfolio meter**")
            st.markdown(f"*({osmn_demand_kw:,} kW demand reduction, 80% self-consumed at benefiting meter)*")
            st.markdown(
                f"- Energy offset at full retail: ${osmn_energy_val:,.0f}/yr\n"
                f"- Demand reduction: ${osmn_demand_val:,.0f}/yr\n"
                f"- Residual AHIC cashout: ${osmn_ahic_val:,.0f}/yr\n"
                f"- **OS-NM total: ${osmn_total:,.0f}/yr**\n"
                f"- **Advantage over SRCS: +${osmn_gap:,.0f}/yr**"
            )
            st.success(f"OS-NM premium over SRCS: +${osmn_gap:,.0f}/yr")
        with col_b:
            st.markdown("**Why demand charges dominate:**")
            summer_d = osmn_demand_kw * (SG_DIST_D + SG_GT_D_SUM) * 4
            winter_d = osmn_demand_kw * (SG_DIST_D + SG_GT_D_WIN) * 8
            st.markdown(
                f"At a large commercial building, reducing billing demand by "
                f"{osmn_demand_kw:,} kW is worth:\n"
                f"- Summer (4 mo): {osmn_demand_kw:,} × ${SG_DIST_D+SG_GT_D_SUM:.2f}/kW "
                f"× 4 = **${summer_d:,.0f}**\n"
                f"- Winter (8 mo): {osmn_demand_kw:,} × ${SG_DIST_D+SG_GT_D_WIN:.2f}/kW "
                f"× 8 = **${winter_d:,.0f}**\n"
                f"- **Total demand savings: ${summer_d+winter_d:,.0f}/yr**\n\n"
                f"This demand value is **invisible under SRCS** — CSG pays only $/kWh. "
                f"It is only captured through NM credits applied via OS-NM."
            )

        st.markdown("---")
        st.markdown(
            "**Rate sources:** PSCo Electric Tariff (COLO. PUC No. 8 Electric), Schedule SG (Sheet 43), "
            "Schedule SRCS (Sheet 114G, Ninth Revised), Schedule NM (Sheet 112), "
            "Schedule OS-NM (Sheet 123). "
            "Demand: distribution $9.36/kW, G&T summer $15.14/kW, G&T winter $9.11/kW. "
            "AHIC for SG excess generation: $0.07437 (2023), $0.07597 (2024), $0.06511 (2025). "
            "Rates effective November 2025."
        )
        st.warning(
            "**The T1 upgrade is the policy trigger that was missed.** "
            "Xcel upgraded the DIA7 service transformer from ~300 kVA to 2,500 kVA as part of "
            "the interconnection approval — a 8× capacity increase to accommodate DIA's solar export. "
            "That upgrade, from a transformer sized for a small airport outbuilding to one sized for "
            "2,250 kW of export through a three-transformer stack, should have been a mandatory "
            "trigger for a harmonic study. It was not, because the DG Tech Manual contains no "
            "such requirement. Any future service upgrade of this magnitude for a multi-transformer "
            "DG topology should trigger the harmonic study requirement proposed below. "
            "DIA is likely not the only large institutional customer (airport, university, hospital) "
            "pursuing this same OS-NM strategy at other sites."
        )

    # ── Draft DG Tech Manual language ─────────────────────────────────────────
    with st.expander("📋 Proposed DG Tech Manual Update — Draft Language", expanded=False):
        st.markdown("**Proposed addition to Section 3.2 (Harmonics) of the DG Tech Manual, "
                    "following the existing IEEE 519 / IEEE 1547 compliance requirement:**")
        st.code(
            "3.2.1  Harmonic Study Requirement — Multi-Transformer Interconnections\n\n"
            "For any Distributed Generation (DG) facility rated greater than 500 kW AC that\n"
            "interconnects through more than two (2) power transformers in series between the\n"
            "Company's distribution source and the generation inverters or terminals, the Customer\n"
            "shall submit a harmonic impedance study as part of the Interconnection Application.\n\n"
            "The study shall:\n"
            "  (a) Calculate the driving-point harmonic impedance at the Point of Common Coupling\n"
            "      (PCC) for harmonic orders h=3 through h=13, using a pi-section model for\n"
            "      underground cable segments longer than 500 feet;\n"
            "  (b) Demonstrate that voltage THD at the PCC does not exceed the limits of\n"
            "      IEEE 519-2022 Table 2 at full rated generation output;\n"
            "  (c) Demonstrate compliance at each intermediate transformer bus, not only at\n"
            "      the billing meter;\n"
            "  (d) If any intermediate bus exceeds IEEE 519 limits, the Customer shall propose\n"
            "      a mitigation plan (harmonic filter, active filter, or equivalent) prior to\n"
            "      interconnection approval.\n\n"
            "The Company may require the Customer to install a power quality monitor at the PCC\n"
            "and at intermediate buses for a minimum of 30 days following commercial operation,\n"
            "at the Customer's expense, to verify compliance.\n\n"
            "Note on winding configuration: Grounded-Wye/Grounded-Wye step-up transformers\n"
            "(Section 10.2.a) are acceptable for grounding purposes but do not provide harmonic\n"
            "filtering. h=5 and h=7 harmonics (positive and negative sequence) pass through any\n"
            "winding configuration. A delta winding only blocks zero-sequence (triplen) harmonics.\n"
            "Harmonic mitigation requires a filter, not a winding configuration change.",
            language="text"
        )
        st.markdown("**Rationale:** The current DG Tech Manual requires IEEE 519 compliance "
                    "(§3.2) but provides no guidance on how to analyze multi-transformer topologies "
                    "for harmonic amplification. Three stacked transformers with leakage reactances "
                    "in series create a 5th-harmonic impedance 5× higher than the fundamental "
                    "impedance, amplifying inverter harmonic currents and causing audible transformer "
                    "vibration and potential thermal stress — even when voltage THD at the billing "
                    "meter appears compliant.")

    # ── Rate design fix ───────────────────────────────────────────────────────
    with st.expander("⚖️ Proposed Rate Design Fix — Coincidence-Adjusted Demand Credit in SRCS",
                     expanded=False):
        st.markdown(
            "**The goal:** Remove the financial incentive to build multi-transformer OS-NM "
            "topologies by making the same economic outcome available through a direct CSG "
            "connection — which is simpler, safer, and doesn't create harmonic amplification risk."
        )
        st.markdown("---")

        st.markdown("#### Why harmonic mitigation is not a reliable long-term answer")
        st.markdown(
            "Passive harmonic filters — the standard mitigation for this topology — have "
            "meaningful reliability limitations in field conditions:\n"
            "- **Detuning over time:** capacitor values drift with temperature and age, "
            "shifting the filter's resonant frequency away from h=4.7 and reducing effectiveness\n"
            "- **Fault vulnerability:** filter capacitors are exposed to fault current and "
            "transient overvoltages on the 0.48 kV bus; a single fault event can damage or "
            "destroy the filter\n"
            "- **New resonance risk:** an improperly sized or degraded filter can create a "
            "new resonance at a different harmonic, potentially making the problem worse\n"
            "- **Maintenance dependency:** a filter that isn't regularly tested and maintained "
            "may appear functional while providing little actual harmonic attenuation\n"
            "- **Active filters** are more robust but cost $150,000–$400,000 installed "
            "for a 2.25 MW site, require sophisticated controls, and still require maintenance\n\n"
            "The right answer is not to build this topology in the first place. "
            "A solar facility connected directly to a utility-owned distribution transformer "
            "has one transformer between the source and the inverters. There is no harmonic "
            "amplification mechanism to mitigate."
        )
        st.error(
            "Every future multi-transformer OS-NM site will face the same mitigation "
            "problem — and there is no guarantee the customer will actually install or "
            "maintain effective mitigation. The DG Tech Manual amendment (above) creates "
            "a study requirement and a mitigation condition, but it doesn't remove the "
            "underlying incentive. As long as OS-NM demand arbitrage is more valuable "
            "than SRCS, customers will keep trying to build this topology."
        )

        st.markdown("---")
        st.markdown("#### Proposed rate design change")
        st.markdown(
            "**Recommendation to Regulatory Affairs:** Petition the CPUC to add a "
            "**coincidence-adjusted demand credit** to Schedule SRCS for commercial and "
            "industrial subscribers, structured as follows:"
        )
        st.markdown(
            "**1. Credit mechanism:** For each billing period, a SRCS subscriber receives "
            "a demand credit equal to:\n\n"
            "> *Credit (kW) = min(subscriber's allocated CSG output during billing demand hour, "
            "subscriber's billing demand)*\n\n"
            "The 'billing demand hour' is the 15-minute interval that set the subscriber's "
            "billing demand for that month. Xcel already has interval meter data for all "
            "large commercial accounts on Schedule SG/PG — this calculation is straightforward."
        )
        st.markdown(
            "**2. Credit rate:** The demand credit is valued at the same demand charge "
            "rates currently in Schedule SG/PG:\n"
            f"- Distribution demand: **${9.36:.2f}/kW-month**\n"
            f"- G&T demand (summer): **${15.14:.2f}/kW-month** (Jun–Sep)\n"
            f"- G&T demand (winter): **${9.11:.2f}/kW-month** (Oct–May)"
        )
        st.markdown(
            "**3. Why coincidence factor matters:** "
            "Solar production at noon doesn't offset demand that peaks at 6pm. "
            "Crediting only actual coincident demand reduction:\n"
            "- Avoids crediting load that solar didn't actually serve\n"
            "- Bounds the cost shift to other ratepayers\n"
            "- Is technically defensible — it reflects real infrastructure cost causation\n"
            "- Creates the right incentive for CSG siting near high-demand loads "
            "(which is where solar is most grid-beneficial anyway)"
        )
        st.markdown(
            "**4. Expected outcome:** A large institutional customer like DIA would receive "
            "essentially the same financial value from subscribing to a SRCS-connected CSG "
            "as they currently receive from building behind a low-demand OS-NM meter. "
            "The complex multi-transformer topology provides no additional financial benefit "
            "and therefore won't be built."
        )

        st.markdown("---")
        st.markdown("#### What this does NOT require")
        st.markdown(
            "- No reduction in the net metering benefit — existing NM customers are unaffected\n"
            "- No change to the SRCS energy credit rate\n"
            "- No new metering infrastructure (interval data already exists)\n"
            "- No statutory change — demand credit valuation is a tariff design question "
            "within the CPUC's existing authority over Schedule SRCS"
        )
        st.success(
            "This is the lowest-disruption regulatory fix: it makes the safe topology "
            "as economically attractive as the risky one, without touching existing "
            "net metering rights or creating new subsidy mechanisms."
        )

    # ── Draft recommendation letter ────────────────────────────────────────────
    with st.expander("✉️ Draft Recommendation Letter — DER Engineering & Regulatory Affairs",
                     expanded=False):
        st.markdown("*Draft for review — not for distribution without engineering and legal review.*")
        st.markdown("---")
        letter = f"""
**INTERNAL MEMORANDUM**

**TO:** DER Engineering Department; Regulatory Affairs Department
**FROM:** Distribution Protection Engineering
**DATE:** {__import__('datetime').date.today().strftime('%B %d, %Y')}
**RE:** DG Tech Manual Gap — Harmonic Amplification Risk in Multi-Transformer Solar
        Interconnections; Rate Design Incentive for Complex Topologies
**CLASSIFICATION:** Internal — For Review and Action

---

**BACKGROUND**

Distribution Protection Engineering has completed a power quality investigation of the
2,250 kW AC solar interconnection at 9907 Queensburg St, Denver (DIA7 billing point,
0.48 kV secondary, Schedule SG + Schedule NM). A November 2023 power quality study
(four-day monitoring period) and a circuit impedance analysis have identified two issues
requiring action beyond the immediate site.

---

**TECHNICAL FINDING: HARMONIC AMPLIFICATION — 9907 QUEENSBURG**

The circuit topology at this site connects three power transformers in series between the
Xcel 24.9 kV source and the 0.63 kV inverter terminals:

  T1: 24.9/0.48 kV, 2,500 kVA, Yg-Yg (Xcel service transformer)
  T2: 0.48/13.8 kV, 2,350 kVA, Yg-Yg (customer step-up — billing point on LV side)
  T3: 13.8/0.63 kV, 2,520 kVA, Δ-Yg (inverter step-up)

Transformer leakage reactances scale linearly with harmonic order (X_h = h × X_60Hz).
With three transformers in series, the 5th harmonic (300 Hz) impedance presented to the
inverter harmonic current source is approximately 5× the 60 Hz source impedance. This
amplification is directly measurable: the November 2023 PQ study recorded sustained
audible vibration in T2, consistent with beat-frequency electromagnetic forces at 240 Hz
and 360 Hz (from h=1 × h=5 cross-product terms).

PQ study results at the DIA7 0.48 kV billing meter:
  - Voltage THD (Vcn): avg 2.0%, max 2.4% — within IEEE 519 5% limit at this point
  - Dominant harmonic: h=5 (300 Hz), Vcn avg 4.3 Vac (1.55% of fundamental)
  - The 0.48 kV billing meter is the lowest-impedance point in the circuit; the 13.8 kV
    bus between T2 and the feeder was not monitored and is expected to show higher distortion

The site is technically compliant at the billing meter. However, the T2 transformer is
experiencing harmonic stress that was not identified during the interconnection review
because the DG Tech Manual does not require harmonic analysis for this topology.

**Recommended immediate action (site-specific):**
  1. Install temporary PQ monitor at 13.8 kV bus — establish distortion baseline
  2. Install 5th harmonic passive filter (detuned to h≈4.7) at DIA7 0.48 kV bus
  3. Evaluate T2 winding change to Wye-Delta (zero-sequence fault isolation)

---

**STANDARDS GAP: DG TECH MANUAL**

A review of the Xcel DG Tech Manual (Rev. 10/28/2021) confirms that transformer winding
configuration is evaluated solely for grounding/zero-sequence adequacy (§10, §17).
The manual calls Grounded-Wye/Grounded-Wye "common" for step-up transformers and
imposes no requirement to analyze harmonic amplification from stacked transformer
impedances. Section 3.2 requires IEEE 519 / IEEE 1547 compliance but provides no
analytical guidance for multi-transformer topologies.

An interconnection engineer following the current manual to the letter would approve the
9907 Queensburg topology and be fully in compliance — and still produce a circuit with
measurable harmonic amplification and transformer stress.

**Recommended action for DER Engineering:**
Amend Section 3.2 of the DG Tech Manual to require a harmonic impedance study for any
DG facility >500 kW interconnecting through more than two transformers in series. The
study must demonstrate IEEE 519 compliance at each transformer bus, not only at the
billing meter. (Proposed draft language is provided separately in this memorandum.)

---

**RATE DESIGN FINDING: NET METERING INCENTIVE DRIVING TOPOLOGY CHOICE**

The customer at 9907 Queensburg is Denver International Airport (DIA). DIA pursued
this three-transformer topology to connect generation to a net metering (Schedule NM)
billing meter, rather than interconnecting directly as a Community Solar Garden (Schedule
SRCS). The financial driver is Schedule OS-NM (Off-Site Net Metering), which allows
NM credits to be applied to DIA's other billing meters across the airport campus.

Key context: The original Xcel service at 9907 Queensburg (DIA7 meter) was approximately
300 kVA — a small equipment outbuilding on airport property. Xcel upgraded the service
transformer to 2,500 kVA (an 8× increase) as part of the interconnection approval to
accommodate 2,250 kW of solar export. The November 2023 PQ study confirms the
DIA7 meter's own load is near zero (avg net flow: -18 kW exporting; avg import current:
~27A at 0.48 kV). Solar production overwhelmingly exceeds on-site consumption at DIA7.

At this specific billing meter, Schedule NM does not produce a rate advantage over
SRCS. Year-end AHIC cashout (SG, 2025) = $0.06511/kWh, which is LOWER than
the SRCS 2026 credit of $0.07891/kWh. NM at this meter earns approximately
${abs(int(nm_site_gap)):,}/year LESS than SRCS would.

The financial logic of this topology is therefore NOT at the 9907 Queensburg billing
meter itself. The economic driver is Schedule OS-NM (Off-Site Net Metering, Sheet
123), which permits NM credits to be applied against other meters owned by the same
customer anywhere in Xcel's Colorado service territory.

If NM credits from this 2,250 kW plant are applied to a high-demand commercial building
with {osmn_demand_kw:,} kW of coincident demand, the value of the NM credits is:

  Energy offset (80% of production at full retail): ~${osmn_energy_val:,.0f}/year
  Demand reduction at benefiting meter:            ~${osmn_demand_val:,.0f}/year
  Residual AHIC cashout (20%):                     ~${osmn_ahic_val:,.0f}/year
  Total OS-NM value:                               ~${osmn_total:,.0f}/year

  vs. SRCS:                                        ~${srcs_rev:,.0f}/year
  OS-NM premium over SRCS:                         ~+${osmn_gap:,.0f}/year

The demand component — which is invisible to a $/kWh comparison and has no
equivalent in the SRCS program — is the financial engine. Demand charges under
Schedule SG are $9.36/kW (distribution) + $15.14/kW (G&T, summer) + $9.11/kW
(G&T, winter). A customer with a large portfolio of commercial buildings can capture
this demand value only through OS-NM, and only by building a solar facility behind
an existing utility billing meter — which is precisely what was done here.

The T1 upgrade from 300 kVA to 2,500 kVA is the visible signature of this economic
decision: DIA built infrastructure specifically to enable OS-NM credit monetization
against the airport's high-demand terminal and concourse meters, and Xcel approved
and facilitated that upgrade without requiring a harmonic study of the resulting
three-transformer circuit. DIA is likely not the only large institutional customer
(major airports, universities, hospitals, campuses) pursuing this same OS-NM
arbitrage strategy, and similar topologies at other sites may carry identical risk.

**A note on harmonic mitigation reliability:**
Distribution Protection Engineering has low confidence that passive harmonic filter
installations at sites like 9907 Queensburg will provide reliable long-term mitigation.
Passive filters detune as capacitor values drift with age and temperature. They are
vulnerable to fault currents and transient overvoltages on the 0.48 kV bus. A degraded
or improperly sized filter can create new resonance at a different harmonic order.
Active filters are more robust but cost $150,000–$400,000 installed and require
sophisticated ongoing maintenance. Neither option is as reliable as simply not building
the three-transformer topology in the first place.

The DG Tech Manual amendment described above creates a study requirement and
mitigation condition for future applications, but it does not remove the financial
incentive to attempt this topology. As long as OS-NM demand arbitrage is more
valuable than SRCS, customers will continue to apply — and the Company will
continue processing applications that result in complex mitigation requirements that
are difficult to verify and maintain.

**Recommended action for Regulatory Affairs:**
Petition the CPUC to add a coincidence-adjusted demand credit to Schedule SRCS
for commercial and industrial subscribers. The mechanism is straightforward:

  For each billing period, a SRCS subscriber receives a demand credit equal to:
    min(subscriber's allocated CSG output during the billing demand peak interval,
        subscriber's metered billing demand)

  Valued at standard Schedule SG/PG demand charge rates:
    Distribution demand:  $9.36/kW-month
    G&T demand (summer):  $15.14/kW-month (June–September)
    G&T demand (winter):  $9.11/kW-month (October–May)

The coincidence requirement — crediting only the solar output that was actually present
during the billing demand peak — bounds cost shift to other ratepayers and is
technically defensible as it reflects real infrastructure cost causation. Interval meter
data for all large commercial accounts already exists and makes this calculation
operationally feasible without new metering infrastructure.

Expected outcome: A large institutional customer (DIA or equivalent) would receive
essentially the same financial value from a SRCS subscription as from building behind
a low-demand OS-NM meter. The complex multi-transformer topology provides no
additional financial benefit and therefore will not be built. The harmonic amplification
problem disappears at its source, not at its symptom.

This is appropriate for Xcel's next SRCS rate case or DER tariff proceeding before
the CPUC. The DG Tech Manual amendment described above should be implemented
regardless, to address existing applications in the interim.

---

**SUMMARY OF RECOMMENDED ACTIONS**

| Action | Owner | Priority |
|---|---|---|
| PQ monitor at 13.8 kV bus — 9907 Queensburg | Customer / Xcel Engr | Immediate |
| Passive harmonic filter — 9907 Queensburg | Customer (DG Tech Manual §3.2) | Short-term |
| DG Tech Manual §3.2 — harmonic study requirement for multi-transformer DG | DER Engineering | Near-term |
| Service transformer upgrade as harmonic study trigger | DER Engineering | Near-term |
| SRCS coincidence-adjusted demand credit — CPUC petition | Regulatory Affairs | Next rate case |

*Note: The passive harmonic filter for 9907 Queensburg is required under the existing*
*DG Tech Manual §3.2 IEEE 519 compliance obligation. However, Distribution Protection*
*Engineering recommends that Regulatory Affairs pursue the SRCS rate design change*
*in parallel, as filter-based mitigation is not a reliable long-term answer for this*
*class of site. Prevention is preferable to mitigation.*

*Attachments: Impedance analysis, PQ study summary, proposed DG Tech Manual language,*
*proposed SRCS tariff language*

---
*Prepared by Distribution Protection Engineering. Contact: {__import__('getpass').getuser()}@xcelenergy.com*
        """
        st.markdown(letter)
        st.download_button(
            label="Download letter as .txt",
            data=letter.replace("**", "").replace("*", ""),
            file_name="9907_Queensburg_Recommendation_Memo.txt",
            mime="text/plain"
        )
