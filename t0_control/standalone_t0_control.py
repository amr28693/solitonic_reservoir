"""
standalone_t0_control.py
========================
Control: 260-dimensional features extracted at t=0 (no evolution).

This script exactly replicates the ψ₀ generation from 1_Master_script_statistics.py
(same grid, same bumps, same phase wavevectors, same jitter, same seeds, same RNG sequence),
extracts the SAME 260-dim feature vector used for evolved states,
but does so BEFORE any NLSE/LSE evolution.

It then loads the existing .npz sweep results and plots all four curves:
  - Static control (4-dim)
  - t=0 control (260-dim, no evolution)
  - LSE system (260-dim, evolved g=0)
  - NLSE system (260-dim, evolved g=1)

If the 260-dim t=0 control performs at chance while the evolved systems do not,
the computational advantage is attributable to the dynamics, not dimensionality.

Usage:
    python standalone_t0_control.py --data_path preload_data/

    Point --data_path to whichever directory contains
    'preload_etaV2paper1C_robustness_sweep_STATISTICAL.npz'
    (or the user-generated version without the 'preload_' prefix).

Output:
    - fig_t0_control_comparison.png   (4-curve comparison figure)
    - t0_control_results.npz          (raw accuracy arrays)
    - t0_control_results.csv          (table for manuscript)

Anderson M. Rodriguez, 2026
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import argparse
import os
import sys

# =================================================================
# 1. SIMULATION PARAMETERS — EXACT MATCH TO MASTER SCRIPT
# =================================================================

NX, NY = 64, 64
L = 20.0
DX, DY = L / NX, L / NY
x = np.linspace(-L / 2, L / 2, NX, endpoint=False)
y = np.linspace(-L / 2, L / 2, NY, endpoint=False)
X, Y = np.meshgrid(x, y)

N_CLASSES = 4
N_SAMPLES_PER_CLASS = 100
N_TOTAL_SAMPLES = N_CLASSES * N_SAMPLES_PER_CLASS

DOWN_FACTOR = 4  # 64 / 4 = 16
N_RESERVOIR_FEATURES = (NX // DOWN_FACTOR) * (NY // DOWN_FACTOR) + 4  # 260

# =================================================================
# 2. INITIAL CONDITION GENERATION — EXACT MATCH
# =================================================================

def get_initial_solitons(class_id):
    """Creates the base initial wavefield psi_0 for a given class.
    Identical to 1_Master_script_statistics.py."""
    amplitude_base = np.zeros((NX, NY), dtype=float)
    centers = [(L / 4, L / 4), (-L / 4, -L / 4)]
    for cx, cy in centers:
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        amplitude_base += np.cosh(R) ** (-1)

    k_map = {
        0: (0.5, 0.5),    # Diagonal
        1: (0.5, -0.5),   # Anti-diagonal
        2: (0.0, 1.0),    # Vertical
        3: (1.0, 0.0),    # Horizontal
    }
    kx_init, ky_init = k_map[class_id]
    phase_base = np.exp(1j * (kx_init * X + ky_init * Y))

    psi0 = amplitude_base * phase_base
    return psi0 / np.sqrt(np.sum(np.abs(psi0) ** 2)) * L


def apply_jitter(psi, eta, rng):
    """Adds uniform phase jitter and small Gaussian amplitude perturbation.
    Identical to 1_Master_script_statistics.py."""
    if eta == 0.0:
        return psi

    delta_phi = rng.uniform(-eta, eta, (NX, NY))
    delta_A = 0.1 * eta * rng.normal(0, 1, (NX, NY))

    amplitude = np.abs(psi) + delta_A
    phase = np.angle(psi) + delta_phi
    amplitude[amplitude < 0] = 0.0

    return amplitude * np.exp(1j * phase)


# =================================================================
# 3. FEATURE EXTRACTION — EXACT MATCH
# =================================================================

def extract_features(psi):
    """Extracts 4 global + 256 spatial = 260 reservoir features.
    Identical to 1_Master_script_statistics.py."""
    amplitude = np.abs(psi)
    phase = np.angle(psi)

    mass = np.sum(amplitude ** 2) * DX * DY

    grad_psi_x = np.diff(psi, axis=0, append=psi[0:1, :]) / DX
    grad_psi_y = np.diff(psi, axis=1, append=psi[:, 0:1]) / DY
    K = np.sum(np.abs(grad_psi_x) ** 2 + np.abs(grad_psi_y) ** 2) * DX * DY

    mean_cos = np.mean(np.cos(phase))
    mean_sin = np.mean(np.sin(phase))

    global_features = np.array([mean_cos, mean_sin, mass, K])

    downsampled_amplitude = amplitude[::DOWN_FACTOR, ::DOWN_FACTOR].flatten()
    reservoir_features = np.concatenate([downsampled_amplitude, global_features])

    return global_features, reservoir_features


# =================================================================
# 4. CONTROL: 260-dim FEATURES AT t=0
# =================================================================

def run_t0_control(eta_val, seed):
    """
    Generates ψ₀ with identical RNG sequence to run_experiment_batch(),
    extracts 260-dim reservoir features at t=0 (NO evolution),
    runs the same logistic regression classifier.

    The RNG sequence is: for each sample i in [0, 400),
      - get_initial_solitons(class_id) [deterministic, no RNG]
      - apply_jitter(psi_base, eta, rng) calls rng.uniform then rng.normal

    This exactly matches what run_experiment_batch does before the evolution loop.
    """
    rng = np.random.RandomState(seed)

    X_t0_reservoir = np.zeros((N_TOTAL_SAMPLES, N_RESERVOIR_FEATURES))
    Y_labels = np.zeros(N_TOTAL_SAMPLES, dtype=int)

    for i in range(N_TOTAL_SAMPLES):
        class_id = i // N_SAMPLES_PER_CLASS

        psi_base = get_initial_solitons(class_id)
        psi0_jittered = apply_jitter(psi_base, eta_val, rng)

        # Extract 260-dim features from the INITIAL state — no evolution
        _, reservoir_feats = extract_features(psi0_jittered)
        X_t0_reservoir[i, :] = reservoir_feats
        Y_labels[i] = class_id

    # Same train/test split as master script
    X_train, X_test, y_train, y_test = train_test_split(
        X_t0_reservoir, Y_labels,
        test_size=0.3, random_state=seed, stratify=Y_labels
    )

    clf = LogisticRegression(solver='lbfgs', random_state=seed, max_iter=5000)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    return acc


# =================================================================
# 5. SWEEP: REPLICATE FULL η × SEED GRID
# =================================================================

def run_full_sweep():
    """Run the t=0 control across all 15 η values × 10 seeds."""
    N_SEEDS = 10
    SEEDS = np.arange(N_SEEDS)
    ETA_SWEEP = np.linspace(0.1, 2.0, 15)
    N_ETA = len(ETA_SWEEP)

    acc_t0 = np.zeros((N_SEEDS, N_ETA))

    print("=" * 60)
    print("Running 260-dim t=0 control (no evolution)")
    print(f"  {N_SEEDS} seeds × {N_ETA} η values = {N_SEEDS * N_ETA} runs")
    print(f"  Each run: {N_TOTAL_SAMPLES} samples, feature extraction only")
    print("=" * 60)

    for s_idx, seed in enumerate(SEEDS):
        for e_idx, eta in enumerate(ETA_SWEEP):
            acc = run_t0_control(eta, seed)
            acc_t0[s_idx, e_idx] = acc
        print(f"  Seed {seed} complete — "
              f"acc range: [{acc_t0[s_idx].min():.3f}, {acc_t0[s_idx].max():.3f}]")

    mean_t0 = np.mean(acc_t0, axis=0)
    std_t0 = np.std(acc_t0, axis=0)

    return ETA_SWEEP, mean_t0, std_t0, acc_t0


# =================================================================
# 6. LOAD EXISTING RESULTS & PLOT
# =================================================================

def load_existing_results(data_path):
    """Load the precomputed .npz from the repo."""
    # Try both naming conventions
    candidates = [
        os.path.join(data_path, 'preload_etaV2paper1C_robustness_sweep_STATISTICAL.npz'),
        os.path.join(data_path, 'etaV2paper1C_robustness_sweep_STATISTICAL.npz'),
    ]
    for fpath in candidates:
        if os.path.exists(fpath):
            print(f"  Loading existing results from: {fpath}")
            data = np.load(fpath)
            return data
    raise FileNotFoundError(
        f"Could not find robustness sweep .npz in {data_path}.\n"
        f"  Looked for: {candidates}\n"
        f"  Point --data_path to the correct directory."
    )


def make_comparison_figure(eta, mean_t0, std_t0, existing, output_dir):
    """Four-curve comparison: static, t=0(260), LSE, NLSE."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

    # Existing curves
    ax.errorbar(eta, existing['mean_acc_con_benchmark'],
                yerr=existing['std_acc_con_benchmark'],
                fmt='k-.', alpha=0.5, capsize=3, label='Static control (4-dim)')

    ax.errorbar(eta, mean_t0, yerr=std_t0,
                fmt='s-', color='#9467bd', alpha=0.85, capsize=3, markersize=5,
                label='t=0 features (260-dim, NO evolution)')

    ax.errorbar(eta, existing['mean_acc_res_linear'],
                yerr=existing['std_acc_res_linear'],
                fmt='b--', alpha=0.7, capsize=3,
                label='LSE (260-dim, g=0, evolved)')

    ax.errorbar(eta, existing['mean_acc_res_nonlinear'],
                yerr=existing['std_acc_res_nonlinear'],
                fmt='o-', color='#d45500', alpha=0.85, capsize=3, markersize=5,
                label='NLSE (260-dim, g=1, evolved)')

    ax.axhline(0.25, ls=':', color='gray', alpha=0.5, label='Chance (25%)')
    ax.set_xlabel('Phase Jitter Strength η', fontsize=13)
    ax.set_ylabel('Classification Test Accuracy', fontsize=13)
    ax.set_title('NLSE vs. LSE Robustness to Input Jitter (η), with Static and t=0 Controls',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'fig_t0_control_comparison.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"  Figure saved: {fig_path}")
    plt.close(fig)
    return fig_path


def save_csv(eta, mean_t0, std_t0, existing, output_dir):
    """Save a clean table for the manuscript."""
    csv_path = os.path.join(output_dir, 't0_control_results.csv')
    with open(csv_path, 'w') as f:
        f.write("eta,static_mean,static_std,"
                "t0_260dim_mean,t0_260dim_std,"
                "LSE_mean,LSE_std,"
                "NLSE_mean,NLSE_std\n")
        for i in range(len(eta)):
            f.write(f"{eta[i]:.4f},"
                    f"{existing['mean_acc_con_benchmark'][i]:.4f},"
                    f"{existing['std_acc_con_benchmark'][i]:.4f},"
                    f"{mean_t0[i]:.4f},{std_t0[i]:.4f},"
                    f"{existing['mean_acc_res_linear'][i]:.4f},"
                    f"{existing['std_acc_res_linear'][i]:.4f},"
                    f"{existing['mean_acc_res_nonlinear'][i]:.4f},"
                    f"{existing['std_acc_res_nonlinear'][i]:.4f}\n")
    print(f"  CSV saved: {csv_path}")
    return csv_path


# =================================================================
# 7. MAIN
# =================================================================

def main():
    parser = argparse.ArgumentParser(
        description='260-dim t=0 control for NLSE reservoir paper')
    parser.add_argument('--data_path', type=str, default='preload_data/',
                        help='Directory containing the existing .npz results')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory for output files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run the t=0 control ---
    eta, mean_t0, std_t0, acc_t0_all = run_full_sweep()

    # --- Save raw results ---
    npz_path = os.path.join(args.output_dir, 't0_control_results.npz')
    np.savez(npz_path,
             eta_values=eta,
             mean_acc_t0_260dim=mean_t0,
             std_acc_t0_260dim=std_t0,
             acc_t0_all_seeds=acc_t0_all)
    print(f"  Raw results saved: {npz_path}")

    # --- Load existing and compare ---
    try:
        existing = load_existing_results(args.data_path)
        fig_path = make_comparison_figure(eta, mean_t0, std_t0, existing,
                                          args.output_dir)
        csv_path = save_csv(eta, mean_t0, std_t0, existing, args.output_dir)
    except FileNotFoundError as e:
        print(f"\n  WARNING: {e}")
        print("  t=0 results saved but comparison figure/CSV not generated.")
        print("  Rerun with --data_path pointing to the .npz directory.")
        fig_path, csv_path = None, None

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY: 260-dim t=0 Control (No Evolution)")
    print("=" * 60)
    print(f"{'η':>8s}  {'t0(260)':>10s}  {'Static(4)':>10s}  "
          f"{'LSE(260)':>10s}  {'NLSE(260)':>10s}")
    print("-" * 60)

    try:
        for i in range(len(eta)):
            print(f"{eta[i]:8.4f}  {mean_t0[i]:10.4f}  "
                  f"{existing['mean_acc_con_benchmark'][i]:10.4f}  "
                  f"{existing['mean_acc_res_linear'][i]:10.4f}  "
                  f"{existing['mean_acc_res_nonlinear'][i]:10.4f}")
    except NameError:
        for i in range(len(eta)):
            print(f"{eta[i]:8.4f}  {mean_t0[i]:10.4f}")

    print("=" * 60)
    print("Done.")


if __name__ == '__main__':
    main()
