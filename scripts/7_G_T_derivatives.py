import numpy as np
import matplotlib.pyplot as plt

"""
Derivative Analysis of NLSE Reservoir Performance

This script calculates and visualizes the first and second derivatives of NLSE reservoir 
performance with respect to time and nonlinearity strength. Specifically:

1. Computes the first derivative of accuracy with respect to time (d(acc)/dT) during a time sweep.
2. Computes the second derivative of accuracy with respect to nonlinearity strength (d²(acc)/dg²) during a g-sweep.
3. Visualizes both derivative analyses in two subplots:
   - Left: Accuracy vs. time with the first derivative (rate of change)
   - Right: Accuracy vs. nonlinearity strength with the second derivative (curvature)
   
Files saved as: 'derivative_analysis.png' and 'derivative_analysis.pdf' in the 'figures/' folder.

Dependencies: numpy, matplotlib
"""


# =================================================================
# LOAD DATA
# =================================================================

g_data = np.load('data/etaV2g_sweep_results.npz')
time_data = np.load('data/time_sweep_results.npz')

# G-sweep (at eta=2.0)
g_values = g_data['g_values']
acc_g = g_data['reservoir_acc_e07']  # eta=2.0

# Time-sweep (at eta=2.0, g=1.0)
T_values = time_data['T_values']
acc_T = time_data['reservoir_acc_e07']  # eta=2.0

# =================================================================
# 1. FIRST DERIVATIVE: TIME SWEEP
# =================================================================

# Calculate d(acc)/dT using central differences
dT = np.diff(T_values)
d_acc_dT = np.diff(acc_T) / dT

# Midpoints for plotting
T_midpoints = (T_values[:-1] + T_values[1:]) / 2

print("=" * 60)
print("TIME SWEEP: First Derivative Analysis")
print("=" * 60)
print(f"d(accuracy)/dT at eta=2.0, g=1.0:")
for T_mid, derivative in zip(T_midpoints, d_acc_dT):
    print(f"  T ≈ {T_mid:.2f}: {derivative:.4f} (acc/unit time)")

mean_derivative = np.mean(d_acc_dT)
print(f"\nMean d(acc)/dT: {mean_derivative:.4f}")
if mean_derivative > 0:
    print("✓ Positive slope: NLSE performance IMPROVES with evolution time")
else:
    print("✗ Negative slope: Performance degrades over time")

# =================================================================
# 2. SECOND DERIVATIVE: G SWEEP
# =================================================================

# First derivative: d(acc)/dg
dg = np.diff(g_values)
d_acc_dg = np.diff(acc_g) / dg
g_midpoints_1st = (g_values[:-1] + g_values[1:]) / 2

# Second derivative: d²(acc)/dg²
d2_acc_dg2 = np.diff(d_acc_dg) / np.diff(g_midpoints_1st)
g_midpoints_2nd = (g_midpoints_1st[:-1] + g_midpoints_1st[1:]) / 2

print("\n" + "=" * 60)
print("G SWEEP: Second Derivative Analysis")
print("=" * 60)
print(f"d²(accuracy)/dg² at eta=2.0, T=2.0:")
for g_mid, second_deriv in zip(g_midpoints_2nd, d2_acc_dg2):
    print(f"  g ≈ {g_mid:.2f}: {second_deriv:.4f}")

mean_second_deriv = np.mean(d2_acc_dg2)
print(f"\nMean d²(acc)/dg²: {mean_second_deriv:.4f}")
if mean_second_deriv > 0:
    print("✓ Concave-up: Accelerating returns to nonlinearity (threshold effect)")
elif mean_second_deriv < 0:
    print("✗ Concave-down: Diminishing returns to nonlinearity")
else:
    print("○ Linear response to g")

# =================================================================
# 3. VISUALIZATION
# =================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- LEFT: Time Sweep + First Derivative ---
ax1_twin = ax1.twinx()
ax1.plot(T_values, acc_T * 100, 'o-', color='blue', linewidth=2, label='NLSE Accuracy')
ax1_twin.plot(T_midpoints, d_acc_dT * 100, 's--', color='red', linewidth=2, label='d(acc)/dT')
ax1_twin.axhline(0, color='gray', linestyle=':', alpha=0.5)

ax1.set_xlabel('Evolution Time T', fontsize=12)
ax1.set_ylabel('Test Accuracy [%]', fontsize=12, color='blue')
ax1_twin.set_ylabel('Rate of Change [%/unit time]', fontsize=12, color='red')
ax1.set_title(r'Temporal Self-Organization ($\eta=2.0$, $g=1.0$)', fontsize=13)
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin.tick_params(axis='y', labelcolor='red')
ax1.grid(True, alpha=0.3)

# --- RIGHT: G Sweep + Second Derivative ---
ax2_twin = ax2.twinx()
ax2.plot(g_values, acc_g * 100, 'o-', color='green', linewidth=2, label='NLSE Accuracy')
ax2_twin.plot(g_midpoints_2nd, d2_acc_dg2 * 100, 's--', color='purple', linewidth=2, label='d²(acc)/dg²')
ax2_twin.axhline(0, color='gray', linestyle=':', alpha=0.5)

ax2.set_xlabel('Nonlinearity Strength g', fontsize=12)
ax2.set_ylabel('Test Accuracy [%]', fontsize=12, color='green')
ax2_twin.set_ylabel('Curvature [%/g²]', fontsize=12, color='purple')
ax2.set_title(r'Threshold-Driven Stabilization ($\eta=2.0$, $T=2.0$)', fontsize=13)
ax2.tick_params(axis='y', labelcolor='green')
ax2_twin.tick_params(axis='y', labelcolor='purple')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/derivative_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/derivative_analysis.pdf', bbox_inches='tight')
print("\n" + "=" * 60)
print("Plots saved: derivative_analysis.png and .pdf")
print("=" * 60)