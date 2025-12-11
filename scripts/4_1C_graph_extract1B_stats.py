# graph and extract .npz info from 1C_stats run

import numpy as np
import matplotlib.pyplot as plt

# Load the statistical results file
try:
    data = np.load('data/etaV2paper1C_robustness_sweep_STATISTICAL.npz')
except FileNotFoundError:
    print("Error: The file 'etaV21C_robustness_sweep_STATISTICAL.npz' was not found.")
    # Exit gracefully if the file isn't there
    raise

# Extract the arrays
eta_values = data['eta_values']
mean_nl = data['mean_acc_res_nonlinear']
std_nl = data['std_acc_res_nonlinear']
mean_lin = data['mean_acc_res_linear']
std_lin = data['std_acc_res_linear']
mean_con = data['mean_acc_con_benchmark']
std_con = data['std_acc_con_benchmark']

# --- Plot Generation (Final Figure 2) ---
plt.figure(figsize=(9, 6))

# 1. NLSE Reservoir (g=1.0) with error bars
plt.errorbar(eta_values, mean_nl, yerr=std_nl, fmt='o-', capsize=5,
             color='orange', linewidth=2, label=r'NLSE Reservoir ($g=1.0$)')

# 2. LSE Reservoir (g=0.0) with error bars
plt.errorbar(eta_values, mean_lin, yerr=std_lin, fmt='s--', capsize=5,
             color='purple', linewidth=2, label=r'LSE Reservoir ($g=0.0$)')

# 3. Static Control (Initial state features) with error bars
plt.errorbar(eta_values, mean_con, yerr=std_con, fmt='^:', capsize=5,
             color='gray', linewidth=1.5, label=r'Static Control ($\psi_0$ features)')

# --- Formatting ---
plt.axhline(0.25, color='black', linestyle='-.', alpha=0.5, label='Chance Level (25%)')
plt.xlabel(r'Initial Jitter Strength $\eta$ (a.u.)', fontsize=14)
plt.ylabel('Classification Test Accuracy', fontsize=14)
plt.title('Statistical Robustness Sweep: NLSE vs. LSE Reservoir', fontsize=16)
plt.ylim(0, 1.05)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the figure
plt.savefig('figures/etaV2paper1C_robustness_sweep_FINAL_PLOT.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/etaV2paper1C_robustness_sweep_FINAL_PLOT.pdf', bbox_inches='tight')

print("Plot generation complete. ")
print(rf"NLSE Accuracy at highest jitter ($\eta={eta_values[-1]:.2f}$): {mean_nl[-1]:.3f} $\pm$ {std_nl[-1]:.3f}")
print(rf"LSE Accuracy at highest jitter ($\eta={eta_values[-1]:.2f}$): {mean_lin[-1]:.3f} $\pm$ {std_lin[-1]:.3f}")