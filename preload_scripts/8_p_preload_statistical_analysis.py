# preload_statistical_analysis.py


"""
Statistical Analysis for NLSE vs LSE Reservoir Comparison

This script computes t-tests with Bonferroni correction and Cohen's d effect sizes to evaluate the 
performance differences between NLSE (Nonlinear Schrödinger Equation) and LSE (Linear Schrödinger Equation) reservoirs. 

Key Steps:
1. Loads accuracy data for NLSE and LSE reservoirs from a predefined dataset.
2. Identifies the critical regime (η > 1.05) for comparison.
3. Computes t-statistics, p-values, and Cohen's d effect sizes for each critical regime.
4. Applies Bonferroni correction to adjust for multiple comparisons.
5. Outputs results including significance after Bonferroni correction.

Files:
- Data loaded from 'preload_data/preload_etaV2paper1C_robustness_sweep_STATISTICAL.npz'.
- Results printed to the console with significance levels.

"""
import numpy as np

# Load canonical data
data = np.load('preload_data/preload_etaV2paper1C_robustness_sweep_STATISTICAL.npz')

eta_values = data['eta_values']
mean_nlse = data['mean_acc_res_nonlinear']
std_nlse = data['std_acc_res_nonlinear']
mean_lse = data['mean_acc_res_linear']
std_lse = data['std_acc_res_linear']

N_TRIALS = 10
critical_idx = np.where(eta_values > 1.05)[0]
n_comparisons = len(critical_idx)
bonferroni_alpha = 0.01 / n_comparisons

print("=" * 70)
print("STATISTICAL ANALYSIS: NLSE vs LSE Reservoir Performance")
print("=" * 70)
print(f"Critical regime: η > 1.05 ({n_comparisons} comparisons)")
print(f"Bonferroni-corrected α = 0.01 / {n_comparisons} = {bonferroni_alpha:.4f}")
print("-" * 70)
print(f"{'η':>8} {'Δ Acc':>10} {'t-stat':>10} {'p-value':>12} {'Cohen d':>10} {'Sig':>6}")
print("-" * 70)

from scipy import stats

for idx in critical_idx:
    eta = eta_values[idx]
    diff = mean_nlse[idx] - mean_lse[idx]
    
    # Pooled standard deviation and standard error
    s_pooled = np.sqrt((std_nlse[idx]**2 + std_lse[idx]**2) / 2)
    se_diff = np.sqrt(std_nlse[idx]**2 + std_lse[idx]**2) / np.sqrt(N_TRIALS)
    
    # t-statistic and p-value (two-tailed, df = 2n - 2 = 18)
    t_stat = diff / se_diff
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=18))
    
    # Cohen's d
    d = diff / s_pooled
    
    # Significance after Bonferroni
    sig = "***" if p_value < bonferroni_alpha else "ns"
    
    print(f"{eta:>8.4f} {diff:>+10.4f} {t_stat:>10.2f} {p_value:>12.6f} {d:>+10.2f} {sig:>6}")

print("=" * 70)
print("*** = significant after Bonferroni correction (p < {:.4f})".format(bonferroni_alpha))
print("ns  = not significant after correction")
print("=" * 70)