# 9907 Queensburg Solar — Power Quality Dashboard

Streamlit dashboard for the 9907 Queensburg St solar DG interconnection at Denver International Airport (DIA). Analyzes harmonic amplification from a three-transformer stack, models IEEE 519 / IEEE 1547 compliance against the November 2023 PQ study, and surfaces engineering and policy recommendations.

---

## What the Dashboard Does

| Tab | Contents |
|-----|----------|
| **Overview** | Site one-line, project summary, service upgrade from 300 kVA → 2,500 kVA |
| **Harmonic Spectrum** | Voltage/current THD plots from the PQ study; h=5 dominance visualization |
| **Frequency Scan** | Impedance vs. frequency; resonance peak identification |
| **Sequence Networks** | Zero/positive/negative sequence models; effect of transformer winding configurations |
| **Circuit Model** | OpenDSS circuit (or analytical fallback on ARM64) with power flow results |
| **Recommendations** | Engineering priorities, rate design analysis, DG Tech Manual gap, draft internal letter |

---

## Prerequisites

- **Python 3.9 or newer**
- The packages below (all installable via `pip`):
  - `streamlit`
  - `plotly`
  - `pandas`
  - `numpy`
  - `matplotlib`

**OpenDSS (optional):** The app uses `opendssdirect.py` if available. On Apple Silicon (M1/M2/M3) it falls back automatically to analytical calculations — no action needed.

---

## Installation

### Mac

```bash
# 1. Clone the repository
git clone https://github.com/jacob-wheat-acre/9907-queensburg-solar.git
cd 9907-queensburg-solar

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install streamlit plotly pandas numpy matplotlib

# 4. (Optional) Install OpenDSS bindings — Intel Macs only
#    Skip on M-series; the app falls back to analytical mode automatically
pip install opendssdirect.py

# 5. Run the dashboard
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`. If that port is busy, Streamlit picks the next available one and prints it in the terminal.

#### Apple Silicon note
`opendssdirect.py` raises a SIGILL on ARM64 at runtime. The dashboard catches this and switches to its built-in analytical model — you will see a yellow info banner in the Circuit Model tab. All other tabs are unaffected.

---

### Windows (PC)

Open **Command Prompt** or **PowerShell**:

```bat
REM 1. Clone the repository
git clone https://github.com/jacob-wheat-acre/9907-queensburg-solar.git
cd 9907-queensburg-solar

REM 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

REM 3. Install dependencies
pip install streamlit plotly pandas numpy matplotlib

REM 4. (Optional) OpenDSS bindings
pip install opendssdirect.py

REM 5. Run the dashboard
streamlit run app.py
```

If `python` is not recognized, try `py` instead. The dashboard opens at `http://localhost:8501`.

#### Windows firewall prompt
The first time you run Streamlit, Windows may ask whether to allow network access. Click **Allow** (or **Allow access**) — this is only for localhost.

---

## Updating

```bash
# Mac / Linux
git pull
source .venv/bin/activate
pip install --upgrade streamlit plotly pandas numpy matplotlib

# Windows
git pull
.venv\Scripts\activate
pip install --upgrade streamlit plotly pandas numpy matplotlib
```

---

## Output Files

These text files are OpenDSS export results from the DIA Queensburg circuit model and are read by the dashboard:

| File | Contents |
|------|----------|
| `DIA_Queensburg_VLN_Node.txt` | Phase-to-neutral voltages at each bus |
| `DIA_Queensburg_Curr_Elem.txt` | Branch currents by element |
| `DIA_Queensburg_Power_elem_kVA.txt` | Real and reactive power flow by element |
| `DIA_Queensburg_Losses.txt` | I²R losses per element |
| `DIA_Queensburg_RegTaps.txt` | Regulator tap positions |

The circuit definition is in `9907_queensburg_solar.dss`.

---

## Troubleshooting

**`streamlit: command not found` (Mac/Linux)**
The virtual environment may not be activated. Run `source .venv/bin/activate` and try again.

**`streamlit` is not recognized (Windows)**
Run `.venv\Scripts\activate` first, or use `python -m streamlit run app.py`.

**Port already in use**
```bash
# Mac/Linux — kill whatever holds port 8501
lsof -ti :8501 | xargs kill -9

# Windows (PowerShell)
netstat -ano | findstr :8501
# note the PID in the last column, then:
taskkill /PID <PID> /F
```

**App loads but Circuit Model tab shows a yellow warning**
This is expected on Apple Silicon — the analytical fallback is active. All calculations and plots are still valid.
