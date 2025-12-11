# Repo build in Progress...

The code repository associated with the work: 'Morphological Computation in the Nonlinear Schrodinger Wavefield: Solitonic Dynamics Enable Noise-Tolerant, Self-Organizing Stability Inaccessible to Linear System' by Anderson M. Rodriguez. This is a walkthrough of the findings for the paper.

REPO STRUCTURE:

paper1_repo/
│
├── data/                       # Canonical .npz data
│   ├── etaV2paper1C_robustness_sweep_STATISTICAL.npz
│   ├── etaV2g_sweep_results.npz
│   └── time_sweep_results.npz
│
├── figures/                    # Output directory for figures
│   └── (generated PNG/PDFs go here)
│
├── scripts/                     # Python scripts
│   ├── 0_soliton_fig1_generator.ipynb
│   ├── 1_Master_script_statistics.py
│   ├── 2_Masterscript_analysis.py
│   ├── 3_1B_fig2_generator.py
│   ├── 5_etaV2paper1B_final.py
│   ├── 6_etaV2paper1B_viz_final.py
│   ├── 7_G_T_derivatives.py
│   └── 8_statistical_analysis.py
│
├── Full_call_demo.py            # Master reproducible pipeline
└── README.md



# Reproducible Pipeline for NLSE vs LSE Reservoir Experiments

This repository contains the scripts, canonical data, and visualization pipeline for reproducing the results in [Morphological Computation in the Nonlinear Schr\"odinger Wavefield: Solitonic Dynamics Enable Noise-Tolerant, Self-Organizing Stability Inaccessible to Linear Systems
]. It is organized for reproducibility and safe, fast execution without regenerating long simulations.

## Repository Structure

- `data/`  
  Contains canonical `.npz` files:
  - `etaV2paper1C_robustness_sweep_STATISTICAL.npz` — Main robustness sweep data
  - `etaV2g_sweep_results.npz` — Nonlinearity (g) sweep results
  - `time_sweep_results.npz` — Evolution time (T) sweep results

- `figures/`  
  Outputs for all generated figures (PNG/PDF).

- `scripts/`  
  Original analysis and data-generation scripts:
  - `0_soliton_fig1_generator.ipynb` — Qualitative soliton snapshots
  - `1_Master_script_statistics.py` — Robustness sweep
  - `2_Masterscript_analysis.py` — Decay-rate analysis
  - `3_1B_fig2_generator.py` — Superiority margin figure
  - `5_etaV2paper1B_final.py` — Parameter sweeps (T and g)
  - `6_etaV2paper1B_viz_final.py` — Nonlinearity margin visualization
  - `7_G_T_derivatives.py` — Derivative/curvature analysis
  - `8_statistical_analysis.py` — t-tests, Cohen's d, Bonferroni correction

- `Full_call_demo.py`  
  Safe master pipeline for reproducing all paper figures **using canonical data**, without re-running time-consuming simulations.


  --------------------------------------------

## 'Full Call' Quick Start

1. Ensure Python ≥ 3.8 and dependencies (`numpy`, `matplotlib`, `scipy`) are installed.

2. From the repository root, run:

```bash
python Full_call_demo.py
* This loads canonical .npz data from data/.
* Figures are saved to figures/.
* It does not regenerate raw simulation data, so it’s safe and fast (~minutes instead of hours).

Full Regeneration (Optional)
If you have computing resources and want to regenerate all simulation data from scratch:
1. Edit Full_call_demo.py:

RUN_FULL_GENERATION = True
2. Ensure scratch/ folder exists or configure paths in the script.
3. Run:

python Full_call_demo.py
Warning: This can take many hours. The canonical data in data/ is untouched by default, so you can always run the safe version.

Notes
* All figure numbers and statistics match the paper when using canonical .npz files.
* Paths are relative; the script expects data/ and figures/ subfolders in the repo root.
* This pipeline ensures reproducibility, safe usage, and minimal setup for reviewers or collaborators.
