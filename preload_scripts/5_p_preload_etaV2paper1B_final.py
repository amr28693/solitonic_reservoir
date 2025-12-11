## This script is for the appendices
# Focus is on param sweeps of the weight of the nonlinear term as well as for time, T.

"""
This script defines a 2D grid and Fourier space; extracts 4 global features: mass; gradient energy; mean cosine; mean sine of phase.

The NLSE evolves via Split-Step Fourier method for arbitrary g and timestep DT

Jitter is added.

Logistic regression is used for both reservoir and control with 70/30 train-test split with stratification by class.


NOTE: when running the 'preload_' branch of the repository, this script is also not strictly necessary to run as this is a time intensive, data generation script.

Move along to step 6_p_preload_eta_V2paper1B_viz_final.py

"""


import numpy as np
from scipy.fft import fftn, ifftn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =================================================================
# 1. SIMULATION PARAMETERS (Based on paper's methodology)
# =================================================================

# Grid parameters
NX, NY = 64, 64
L = 20.0  # Physical size of the domain (L x L)
DX, DY = L / NX, L / NY
x = np.linspace(-L / 2, L / 2, NX, endpoint=False)
y = np.linspace(-L / 2, L / 2, NY, endpoint=False)
X, Y = np.meshgrid(x, y)

# Time stepping and evolution baseline
DT = 0.001  # Time step size (small for stability)
G_BASELINE = 1.0  # Nonlinearity (g) used in the main paper
T_BASELINE = 2.0  # Final time (T) used in the main paper

# Classification parameters
N_CLASSES = 4
N_SAMPLES_PER_CLASS = 100
N_TOTAL_SAMPLES = N_CLASSES * N_SAMPLES_PER_CLASS
RANDOM_SEED = 36

# Feature extraction (16x16 downsampling)
DOWN_FACTOR = 4  # 64 / 4 = 16
N_RESERVOIR_FEATURES = (NX // DOWN_FACTOR) * (NY // DOWN_FACTOR) + 4 # 256 + 4

# Fourier space setup (used for the linear step)
kx = 2 * np.pi / L * np.fft.fftfreq(NX) * NX
ky = 2 * np.pi / L * np.fft.fftfreq(NY) * NY
Kx, Ky = np.meshgrid(kx, ky)
K2 = Kx**2 + Ky**2


# Soliton initialization parameters (simplified for structure)
# These define the base amplitude and phase structure for the 4 classes
def get_initial_solitons(class_id):
    """
    Creates the base initial wavefield psi_0 for a given class.
    (Simplified: in real code, this would define the 3x3 lattice centers,
    velocities, and superposition as described in paper.)
    """
    # Base amplitude (sech profile sum, constant across classes)
    amplitude_base = np.zeros((NX, NY), dtype=float)
    centers = [(L/4, L/4), (-L/4, -L/4)]
    for cx, cy in centers:
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        amplitude_base += np.cosh(R)**(-1)

    # Class-dependent phase/velocity (k_x, k_y)
    k_map = {
        0: (0.5, 0.5),   # Diagonal
        1: (0.5, -0.5),  # Anti-diagonal
        2: (0.0, 1.0),   # Vertical
        3: (1.0, 0.0),   # Horizontal
    }
    kx_init, ky_init = k_map[class_id]
    phase_base = np.exp(1j * (kx_init * X + ky_init * Y))
    
    psi0 = amplitude_base * phase_base
    return psi0 / np.sqrt(np.sum(np.abs(psi0)**2)) * L # Normalize mass

# =================================================================
# 2. CORE DYNAMICS AND FEATURE EXTRACTION
# =================================================================

def apply_jitter(psi, eta):
    """Adds i.i.d. Gaussian phase and amplitude jitter with std dev eta."""
    if eta == 0.0:
        return psi
    
    # Phase jitter: Gaussian, zero mean, std dev eta
    delta_phi = np.random.uniform(-eta, eta, (NX, NY))
    
    # Amplitude jitter: 10% of eta * N(0,1)
    delta_A = 0.1 * eta * np.random.normal(0, 1, (NX, NY))
    
    amplitude = np.abs(psi) + delta_A
    phase = np.angle(psi) + delta_phi
    
    return amplitude * np.exp(1j * phase)

def nlse_step(psi_n, dt, g):
    """One step of the Split-Step Fourier Method (SSFM) for NLSE."""
    
    # 1. Linear half-step (Dispersion)
    psi_prime = ifftn(fftn(psi_n) * np.exp(-1j * K2 * dt / 4))
    
    # 2. Nonlinear full-step (Kerr term)
    psi_double_prime = psi_prime * np.exp(1j * g * np.abs(psi_prime)**2 * dt)
    
    # 3. Linear half-step (Dispersion)
    psi_next = ifftn(fftn(psi_double_prime) * np.exp(-1j * K2 * dt / 4))
    
    return psi_next

def extract_features(psi):
    """
    Extracts 4 control features and 260 reservoir features.
    """
    # --- Global Control Features (4 features) ---
    amplitude = np.abs(psi)
    phase = np.angle(psi)
    
    mass = np.sum(amplitude**2) * DX * DY
    
    # Calculate gradient energy (K) using finite differences (simplified)
    grad_psi_x = np.diff(psi, axis=0, append=psi[0:1, :]) / DX
    grad_psi_y = np.diff(psi, axis=1, append=psi[:, 0:1]) / DY
    K = np.sum(np.abs(grad_psi_x)**2 + np.abs(grad_psi_y)**2) * DX * DY
    
    mean_cos = np.mean(np.cos(phase))
    mean_sin = np.mean(np.sin(phase))
    
    global_features = np.array([mean_cos, mean_sin, mass, K])
    
    # --- Reservoir Features (256 + 4 features) ---
    
    # Downsample amplitude (16x16 = 256 features)
    downsampled_amplitude = amplitude[::DOWN_FACTOR, ::DOWN_FACTOR].flatten()
    
    # Concatenate: 256 amplitude + 4 global features
    reservoir_features = np.concatenate([downsampled_amplitude, global_features])
    
    return global_features, reservoir_features

def run_experiment_batch(T_final, g_val, eta_val):
    """
    Generates data and runs the classification for a single (T, g, eta) triplet.
    """
    N_STEPS = int(np.round(T_final / DT))
    
    X_control = np.zeros((N_TOTAL_SAMPLES, 4))
    X_reservoir = np.zeros((N_TOTAL_SAMPLES, N_RESERVOIR_FEATURES))
    Y_labels = np.zeros(N_TOTAL_SAMPLES, dtype=int)
    
    for i in range(N_TOTAL_SAMPLES):
        class_id = i // N_SAMPLES_PER_CLASS
        
        # 1. Initialize and Jitter
        psi_base = get_initial_solitons(class_id)
        psi0_jittered = apply_jitter(psi_base, eta_val)
        
        # 2. Extract Control Features (from initial state)
        X_control[i, :] = extract_features(psi0_jittered)[0]
        
        # 3. Evolve (fixed g)
        psi = psi0_jittered
        for _ in range(N_STEPS):
            psi = nlse_step(psi, DT, g_val)
            
        # 4. Extract Reservoir Features (from evolved state)
        X_reservoir[i, :] = extract_features(psi)[1]
        Y_labels[i] = class_id

    # --- Classification ---
    
    # Train/Test Split (70/30)
    X_res_train, X_res_test, y_train, y_test = train_test_split(
        X_reservoir, Y_labels, test_size=0.3, random_state=RANDOM_SEED, stratify=Y_labels
    )
    X_con_train, X_con_test, _, _ = train_test_split(
        X_control, Y_labels, test_size=0.3, random_state=RANDOM_SEED, stratify=Y_labels
    )
    
    # Reservoir Classifier
    clf_res = LogisticRegression(solver='lbfgs', random_state=RANDOM_SEED, max_iter=5000)
    clf_res.fit(X_res_train, y_train)
    acc_res = accuracy_score(y_test, clf_res.predict(X_res_test))
    
    # Control Classifier
    clf_con = LogisticRegression(solver='lbfgs', random_state=RANDOM_SEED, max_iter=5000)
    clf_con.fit(X_con_train, y_train)
    acc_con = accuracy_score(y_test, clf_con.predict(X_con_test))
    
    return acc_con, acc_res

# =================================================================
# 3. IMPLEMENTATION OF SWEEPS
# =================================================================

def run_time_sweep():
    """Sweeps final time T while holding g=1.0 constant."""
    
    print("\n--- Running Evolution Time (T) Sweep ---")
    
    # Time values to test (spanning the baseline T=1.0)
    T_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    
    # Jitter levels to test (Baseline eta=0.1 and Stress Test eta=2.0)
    eta_values = [0.1, 2.0]
    
    results = {
        'T_values': T_values,
        'control_acc_e01': [],
        'reservoir_acc_e01': [],
        'control_acc_e07': [],
        'reservoir_acc_e07': [],
    }
    
    for T in T_values:
        print(f"  Testing T = {T:.2f} (g={G_BASELINE})")
        
        # Test 1: Low Jitter (eta=0.1)
        acc_con_01, acc_res_01 = run_experiment_batch(T, G_BASELINE, eta_values[0])
        results['control_acc_e01'].append(acc_con_01)
        results['reservoir_acc_e01'].append(acc_res_01)
        
        # Test 2: High Jitter (eta=2.0)
        acc_con_07, acc_res_07 = run_experiment_batch(T, G_BASELINE, eta_values[1])
        results['control_acc_e07'].append(acc_con_07)
        results['reservoir_acc_e07'].append(acc_res_07)
        
    # Convert lists to numpy arrays for saving
    for key in results:
        if key != 'T_values':
            results[key] = np.array(results[key])
            
    # Save results
    np.savez('data/time_sweep_results.npz', **results)
    print("\nTime sweep results saved to data/time_sweep_results.npz")
    print(f"Reservoir Acc (eta=0.1): {results['reservoir_acc_e01']}")
    
def run_g_sweep():
    """Sweeps nonlinearity strength g while holding T=1.0 constant."""
    
    print("\n--- Running Nonlinearity Strength (g) Sweep ---")
    
    # Nonlinearity values to test (from baseline down to near-linear)
    g_values = np.array([1.0, 0.7, 0.5, 0.3, 0.1, 0.0])
    
    # Jitter levels to test (Baseline eta=0.1 and Stress Test eta=2.0)
    eta_values = [0.1, 2.0]
    
    results = {
        'g_values': g_values,
        'control_acc_e01': [],
        'reservoir_acc_e01': [],
        'control_acc_e07': [],
        'reservoir_acc_e07': [],
    }
    
    for g in g_values:
        print(f"  Testing g = {g:.2f} (T={T_BASELINE})")
        
        # Test 1: Low Jitter (eta=0.1)
        acc_con_01, acc_res_01 = run_experiment_batch(T_BASELINE, g, eta_values[0])
        results['control_acc_e01'].append(acc_con_01)
        results['reservoir_acc_e01'].append(acc_res_01)
        
        # Test 2: High Jitter (eta=2.0)
        acc_con_07, acc_res_07 = run_experiment_batch(T_BASELINE, g, eta_values[1])
        results['control_acc_e07'].append(acc_con_07)
        results['reservoir_acc_e07'].append(acc_res_07)

    # Convert lists to numpy arrays for saving
    for key in results:
        if key != 'g_values':
            results[key] = np.array(results[key])

    # Save results
    np.savez('data/etaV2g_sweep_results.npz', **results)
    print("\nNonlinearity sweep results saved to data/etaV2g_sweep_results.npz")
    print(f"Reservoir Acc (eta=2.0): {results['reservoir_acc_e07']}")


# =================================================================
# 4. MAIN EXECUTION
# =================================================================

if __name__ == '__main__':
    # Run the Time Sweep (T)
    run_time_sweep()
    
    # Run the Nonlinearity Sweep (g)
    run_g_sweep()
