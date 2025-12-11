## This script will generate a plot of the Accuracy Margin over the Linear Case as found in Appendix C
# This is computed as percentages and includes both light and high noise conditions
# Figures will save to 'Figures' folder as .odf or .png

import matplotlib.pyplot as plt
import numpy as np

# Load data from the simulation output
data = np.load('preload_data/preload_etaV2g_sweep_results.npz')

g_values = data['g_values']
reservoir_acc_e01 = data['reservoir_acc_e01']
reservoir_acc_high = data['reservoir_acc_e07']

# Calculate the Accuracy Margin over the Linear Case (g=0.0)
# LSE baseline for eta=0.1
acc_lse_e01 = reservoir_acc_e01[g_values == 0.0][0]
Margin_e01 = reservoir_acc_e01 - acc_lse_e01

# LSE baseline for eta=2.0
acc_lse_high = reservoir_acc_high[g_values == 0.0][0]
Margin_high = reservoir_acc_high - acc_lse_high

# --- PLOTTING ---
plt.figure(figsize=(7, 5))

# Plot 1: Margin over LSE (g=0) vs. g
plt.plot(g_values, Margin_e01 * 100, 'o--', color='purple', label=r'Margin @ Low Noise ($\eta=0.1$)')
plt.plot(g_values, Margin_high * 100, 's-', color='orange', linewidth=2, label=r'Margin @ High Noise ($\eta=2.0$)')

# Scatter for raw high-noise accuracy (to show where 100% is)
plt.scatter(g_values, reservoir_acc_high * 100, marker='D', color='red', s=40, zorder=5)

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle=':')
plt.xlim(-0.05, 1.05)
plt.xlabel(r'Nonlinearity Strength $g$ (a.u.)', fontsize=12)
plt.ylabel(r'Accuracy Margin over LSE ($g=0$) [\%]', fontsize=12)
plt.title(r'Sensitivity of Reservoir Robustness to Nonlinearity $g$', fontsize=12)
plt.legend(loc='lower left')
plt.grid(True, which='major', linestyle=':', alpha=0.7)

# Adjust y-limits based on actual data range
min_margin = min(Margin_e01.min(), Margin_high.min()) * 100
max_margin = max(Margin_e01.max(), Margin_high.max()) * 100
plt.ylim(min(min_margin - 1, -5), max(max_margin + 1, 5))


# --- Save both PNG and PDF ---
# PNG for web/screen viewing
plt.savefig('preload_figures/preload_etaV2paper1B_g_sweep_margin.png', dpi=300, bbox_inches='tight')
# PDF for high-quality printing and manuscript submission
plt.savefig('preload_figures/preload_etaV2paper1B_g_sweep_margin.pdf', bbox_inches='tight')

print("Data loaded successfully:")
print(f"g values: {g_values}")
print(f"Margins at eta=2.0: {Margin_high * 100}")
print("\nPlot saved as etaV2paper1B_g_sweep_margin.png and etaV2paper1B_g_sweep_margin.pdf")