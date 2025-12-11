## QUICKSTART_demo.py
# Reduced parameter demonstration of NLSE vs LSE robustness
# Runtime: ~10-15 minutes (vs 6 hours for full analysis)
# 
# This script demonstrates the core finding with minimal computation:
# - 2 random seeds (vs 10 in full analysis)
# - 5 eta points (vs 15 in full analysis)
# - 50 samples per class (vs 100 in full analysis)

import numpy as np
from scipy.fft import fftn, ifftn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

print("\n" + "="*60)
print("QUICKSTART DEMO: NLSE vs LSE Robustness")
print("="*60)
print("Reduced parameters for fast demonstration:")
print("  - 2 random seeds (vs 10 full)")
print("  - 5 eta points (vs 15 full)")
print("  - 50 samples/class (vs 100 full)")
print("  - Expected runtime: 10-15 minutes")
print("="*60 + "\n")

# =================================================================
# SIMULATION PARAMETERS (REDUCED)
# =================================================================

NX, NY = 64, 64
L = 20.0
DX, DY = L / NX, L / NY
x = np.linspace(-L / 2, L / 2, NX, endpoint=False)
y = np.linspace(-L / 2, L / 2, NY, endpoint=False)
X, Y = np.meshgrid(x, y)

DT = 0.001
T_FINAL = 1.0

N_CLASSES = 4
N_SAMPLES_PER_CLASS = 50  # REDUCED from 100
N_TOTAL_SAMPLES = N_CLASSES * N_SAMPLES_PER_CLASS

DOWN_FACTOR = 4
N_RESERVOIR_FEATURES = (NX // DOWN_FACTOR) * (NY // DOWN_FACTOR) + 4

kx = 2 * np.pi / L * np.fft.fftfreq(NX) * NX
ky = 2 * np.pi / L * np.fft.fftfreq(NY) * NY
Kx, Ky = np.meshgrid(kx, ky)
K2 = Kx**2 + Ky**2

# =================================================================
# CORE FUNCTIONS (IDENTICAL TO FULL VERSION)
# =================================================================

def get_initial_solitons(class_id):
    amplitude_base = np.zeros((NX, NY), dtype=float)
    centers = [(L/4, L/4), (-L/4, -L/4)]
    for cx, cy in centers:
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        amplitude_base += np.cosh(R)**(-1)

    k_map = {
        0: (0.5, 0.5),
        1: (0.5, -0.5),
        2: (0.0, 1.0),
        3: (1.0, 0.0),
    }
    kx_init, ky_init = k_map[class_id]
    phase_base = np.exp(1j * (kx_init * X + ky_init * Y))
    
    psi0 = amplitude_base * phase_base
    return psi0 / np.sqrt(np.sum(np.abs(psi0)**2)) * L

def apply_jitter(psi, eta, rng):
    if eta == 0.0:
        return psi
    delta_phi = rng.uniform(-eta, eta, (NX, NY))
    delta_A = 0.1 * eta * rng.normal(0, 1, (NX, NY))
    amplitude = np.abs(psi) + delta_A
    amplitude[amplitude < 0] = 0.0
    phase = np.angle(psi) + delta_phi
    return amplitude * np.exp(1j * phase)

def nlse_step(psi_n, dt, g):
    psi_prime = ifftn(fftn(psi_n) * np.exp(-1j * K2 * dt / 4))
    psi_double_prime = psi_prime * np.exp(1j * g * np.abs(psi_prime)**2 * dt)
    psi_next = ifftn(fftn(psi_double_prime) * np.exp(-1j * K2 * dt / 4))
    return psi_next

def extract_features(psi):
    amplitude = np.abs(psi)
    phase = np.angle(psi)
    mass = np.sum(amplitude**2) * DX * DY
    grad_psi_x = np.diff(psi, axis=0, append=psi[0:1, :]) / DX
    grad_psi_y = np.diff(psi, axis=1, append=psi[:, 0:1]) / DY
    K = np.sum(np.abs(grad_psi_x)**2 + np.abs(grad_psi_y)**2) * DX * DY
    mean_cos = np.mean(np.cos(phase))
    mean_sin = np.mean(np.sin(phase))
    global_features = np.array([mean_cos, mean_sin, mass, K])
    downsampled_amplitude = amplitude[::DOWN_FACTOR, ::DOWN_FACTOR].flatten()
    reservoir_features = np.concatenate([downsampled_amplitude, global_features])
    return global_features, reservoir_features

def run_experiment(g_val, eta_val, seed):
    N_STEPS = int(np.round(T_FINAL / DT))
    rng = np.random.RandomState(seed)
    
    X_control = np.zeros((N_TOTAL_SAMPLES, 4))
    X_reservoir = np.zeros((N_TOTAL_SAMPLES, N_RESERVOIR_FEATURES))
    Y_labels = np.zeros(N_TOTAL_SAMPLES, dtype=int)
    
    for i in range(N_TOTAL_SAMPLES):
        class_id = i // N_SAMPLES_PER_CLASS
        psi_base = get_initial_solitons(class_id)
        psi0_jittered = apply_jitter(psi_base, eta_val, rng)
        X_control[i, :] = extract_features(psi0_jittered)[0]
        
        psi = psi0_jittered
        for _ in range(N_STEPS):
            psi = nlse_step(psi, DT, g_val)
        
        X_reservoir[i, :] = extract_features(psi)[1]
        Y_labels[i] = class_id

    X_res_train, X_res_test, y_train, y_test = train_test_split(
        X_reservoir, Y_labels, test_size=0.3, random_state=seed, stratify=Y_labels
    )
    X_con_train, X_con_test, _, _ = train_test_split(
        X_control, Y_labels, test_size=0.3, random_state=seed, stratify=Y_labels
    )
    
    clf_res = LogisticRegression(solver='lbfgs', random_state=seed, max_iter=5000)
    clf_res.fit(X_res_train, y_train)
    acc_res = accuracy_score(y_test, clf_res.predict(X_res_test))
    
    clf_con = LogisticRegression(solver='lbfgs', random_state=seed, max_iter=5000)
    clf_con.fit(X_con_train, y_train)
    acc_con = accuracy_score(y_test, clf_con.predict(X_con_test))
    
    return acc_con, acc_res

# =================================================================
# QUICKSTART SWEEP
# =================================================================

N_SEEDS = 2  # REDUCED from 10
SEEDS = np.arange(N_SEEDS)
ETA_SWEEP = np.array([0.1, 0.5, 1.0, 1.5, 2.0])  # REDUCED from 15 points
G_NONLINEAR = 1.0
G_LINEAR = 0.0

N_ETA = len(ETA_SWEEP)
ACC_RES_NONLINEAR = np.zeros((N_SEEDS, N_ETA))
ACC_RES_LINEAR = np.zeros((N_SEEDS, N_ETA))
ACC_CON_BENCHMARK = np.zeros((N_SEEDS, N_ETA))

import time
start_time = time.time()

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n[Seed {seed}/{N_SEEDS-1}] Starting...")
    
    for eta_idx, eta_val in enumerate(ETA_SWEEP):
        # NLSE
        acc_con_nl, acc_res_nl = run_experiment(G_NONLINEAR, eta_val, seed)
        ACC_RES_NONLINEAR[seed_idx, eta_idx] = acc_res_nl
        
        # LSE
        acc_con_lin, acc_res_lin = run_experiment(G_LINEAR, eta_val, seed)
        ACC_RES_LINEAR[seed_idx, eta_idx] = acc_res_lin
        ACC_CON_BENCHMARK[seed_idx, eta_idx] = acc_con_lin
        
        print(f"  eta={eta_val:.2f}: NLSE={acc_res_nl:.3f}, LSE={acc_res_lin:.3f}")

elapsed = time.time() - start_time
print(f"\n[COMPLETE] Total time: {elapsed/60:.1f} minutes")

# =================================================================
# ANALYSIS
# =================================================================

mean_nl = np.mean(ACC_RES_NONLINEAR, axis=0)
mean_lin = np.mean(ACC_RES_LINEAR, axis=0)
mean_con = np.mean(ACC_CON_BENCHMARK, axis=0)

# Calculate decay rates in high-noise regime (eta >= 1.0)
high_noise_idx = ETA_SWEEP >= 1.0
from scipy.stats import linregress

slope_nl, _, _, _, _ = linregress(ETA_SWEEP[high_noise_idx], mean_nl[high_noise_idx])
slope_lin, _, _, _, _ = linregress(ETA_SWEEP[high_noise_idx], mean_lin[high_noise_idx])

print("\n" + "="*60)
print("QUICKSTART RESULTS")
print("="*60)
print(f"NLSE decay rate: {slope_nl:.4f} acc/eta")
print(f"LSE decay rate:  {slope_lin:.4f} acc/eta")
print(f"LSE degrades {abs((slope_lin/slope_nl - 1)*100):.1f}% faster than NLSE")
print("\nNote: Full analysis (10 seeds, 15 eta points) gives 122% advantage")
print("="*60)

# =================================================================
# VISUALIZATION
# =================================================================

if not os.path.exists('./figures'):
    os.makedirs('./figures')

plt.figure(figsize=(10, 6))
plt.plot(ETA_SWEEP, mean_nl * 100, 'o-', color='red', linewidth=2, 
         markersize=8, label='NLSE Reservoir (g=1.0)')
plt.plot(ETA_SWEEP, mean_lin * 100, 's--', color='blue', linewidth=2, 
         markersize=8, label='LSE Reservoir (g=0.0)')
plt.plot(ETA_SWEEP, mean_con * 100, '^:', color='gray', linewidth=1, 
         markersize=6, label='Static Control')
plt.axhline(25, color='black', linestyle='-.', linewidth=1, label='Chance Level (25%)')

plt.xlabel(r'Jitter Strength $\eta$ (a.u.)', fontsize=14)
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.title('QUICKSTART DEMO: NLSE vs LSE Robustness', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 105)

plt.text(1.5, 85, f'NLSE advantage:\n{abs((slope_lin/slope_nl - 1)*100):.1f}% slower decay',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=11, ha='center')

plt.savefig('./demo/quickstart_demo.png', dpi=300, bbox_inches='tight')
plt.savefig('./demo/quickstart_demo.pdf', bbox_inches='tight')
print("\nFigure saved to ./demo/quickstart_demo.png")
plt.show()

print("\n" + "="*60)
print("QUICKSTART COMPLETE")
print("="*60)
print("Next steps:")
print("  1. Run full analysis: python 01_run_eta_sweep_parallel.py")
print("  2. Generate paper figures: python 03_generate_figure2.py")
print("="*60)