# This script generates the NLSE Superiority Margin Over Linear Reservoir Figure 
# It will generate and save a .png and .pdf file of the image

import numpy as np
import matplotlib.pyplot as plt

#  Load data from npz file 
data = np.load('data/etaV2paper1C_robustness_sweep_STATISTICAL.npz')

eta_values = data['eta_values']
acc_res_nonlinear = data['mean_acc_res_nonlinear']
acc_res_linear = data['mean_acc_res_linear']

#  Calculations 
accuracy_difference = acc_res_nonlinear - acc_res_linear
accuracy_difference_percent = accuracy_difference * 100

# Key points for annotation and readout
max_margin_index = np.argmax(accuracy_difference_percent)
max_eta = eta_values[max_margin_index]
max_diff = accuracy_difference_percent[max_margin_index]

# Mean superiority margin in the critical high-noise regime (eta > 1.05)
high_noise_margin_range = accuracy_difference_percent[eta_values > 1.05]
mean_high_noise_margin = high_noise_margin_range.mean()

# NLSE accuracy at the 70% utility threshold (eta_c)
eta_c_nlse_index = np.where(acc_res_nonlinear >= 0.70)[0][-1]
eta_c_linear_index = np.where(acc_res_linear >= 0.70)[0][-1]
eta_c_nlse = eta_values[eta_c_nlse_index]
eta_c_linear = eta_values[eta_c_linear_index]


# Plot Generation 
plt.figure(figsize=(8, 6)) # Increased height slightly for external legend

# Plot the difference
plt.plot(eta_values, accuracy_difference_percent, 'o-', color='indigo', label=r'NLSE Margin ($\Delta$ Acc)')

# Add Zero line (Break-even point)
plt.axhline(0.0, color='black', linestyle='--', linewidth=1, label=r'Break-Even ($\Delta$ Acc = 0)')

# Add shading to highlight the critical region (eta > 1.05)
plt.fill_between(eta_values, accuracy_difference_percent, 0, 
                 where=(accuracy_difference_percent > 0) & (eta_values > 1.05),
                 color='orange', alpha=0.3, label=r'NLSE Superiority Margin ($\eta > 1.05$)')

# Vertical line at the breakpoint
plt.axvline(1.05, color='gray', linestyle=':', linewidth=1, alpha=0.6)
plt.text(1.06, -9, r'$\eta_{\text{break}}$', fontsize=10, color='gray')


#  ANNOTATIONS 
# 1. Peak Margin Annotation
plt.annotate(f'Peak Margin: +{max_diff:.1f}%',
             xy=(max_eta, max_diff),
             xytext=(max_eta - 0.5, max_diff + 2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.0, headwidth=6),
             fontsize=10,
             ha='center')

# 2. Annotation for mean superiority in the high-noise regime (eta > 1.0)
# Fixed Syntax Warning by placing the LaTeX part in a raw string inside the f-string
plt.text(1.6, 17.25,
         'Avg. Superiority\n($\\eta > 1.05$): +' + f'{mean_high_noise_margin:.1f}%',
         fontsize=11,
         color='black',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))


# Corrected Y-limit
plt.ylim(-10, 25) 
plt.xlabel(r'Jitter Strength $\eta$ (a.u.)', fontsize=12)
plt.ylabel(r'Accuracy Margin ($\Delta$ Acc, \%)', fontsize=12)
plt.title('NLSE Superiority Margin over Linear Reservoir', fontsize=14)

# Legend upper left
plt.legend(loc='upper left', ncol=2, fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)


plt.savefig('figures/etaV2_paper_nlse_superiority_margin_annotated.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/etaV2_paper_nlse_superiority_margin_annotated.pdf', bbox_inches='tight')
plt.show()

#  NUMERICAL VALIDATION READOUT 
print("\n" + "="*50)
print("NLSE RESERVOIR NUMERICAL VALIDATION SUMMARY")
print("="*50)
print(f"1. Peak Margin (Maximum Difference)")
print(f"   Max $\Delta$ Acc: +{max_diff:.2f}% at $\eta={max_eta:.4f}$")
print("-" * 50)
print(f"2. Average Superiority in High-Noise Regime ($\eta > 1.05$):")
print(f"   Avg $\Delta$ Acc: +{mean_high_noise_margin:.2f}%")
print("-" * 50)
print(f"3. Utility Threshold (Acc $\ge 70\%$):")
print(f"   NLSE Critical Jitter ($\eta_c$): {eta_c_nlse:.4f}")
print(f"   Linear Critical Jitter ($\eta_c$): {eta_c_linear:.4f}")
print(f"   The NLSE extends utility by {eta_c_nlse - eta_c_linear:.4f} in $\eta$ units.")
print("="*50)