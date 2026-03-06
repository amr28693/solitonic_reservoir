# use this in paper
# for 'Paper 1' Figure 1 (CORRECTED)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Figure 1 for Paper 1: Evolution visualization (4 classes)
Matches Script 1 experimental parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# Grid setup (MATCHED TO SCRIPT 1)
# =============================================================================

L = 20.0  
Nx, Ny = 64, 64
dx, dy = L / Nx, L / Ny
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="xy")
k2 = KX**2 + KY**2

dt = 0.001  
T_final = 1.0  # Match Script 1
Nt = int(T_final / dt)  # 1000 steps
L_op = np.exp(-1j * k2 * dt / 2)

def sech(r):
    return 1.0 / np.cosh(r)

def make_initial_condition(class_id, rng, jitter=0.0):
    """
    CORRECTED: Matches Script 1 exactly.
    Fixed bump positions, only phase wavevector varies by class; higher jitter run
    """
    # Fixed centers for ALL classes (matches Script 1)
    centers = [(L/4, L/4), (-L/4, -L/4)]
    
    # Build amplitude envelope (identical for all classes)
    amplitude_base = np.zeros((Ny, Nx), dtype=float)
    for cx, cy in centers:
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        amplitude_base += sech(R)
    
    # Phase wavevector varies by class (matches Script 1)
    k_map = {
        0: (0.5, 0.5),   # Diagonal
        1: (0.5, -0.5),  # Anti-diagonal
        2: (0.0, 1.0),   # Vertical
        3: (1.0, 0.0),   # Horizontal
    }
    kx_init, ky_init = k_map[class_id]
    phase_base = np.exp(1j * (kx_init * X + ky_init * Y))
    
    psi0 = amplitude_base * phase_base
    psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * dx * dy) * L  # Normalize
    
    # Apply jitter (matches Script 1)
    if jitter > 0.0:
        phase_noise = rng.uniform(-jitter, jitter, (Ny, Nx))
        amp_noise = 0.1 * jitter * rng.standard_normal((Ny, Nx))
        amplitude = np.abs(psi0) + amp_noise
        amplitude[amplitude < 0] = 0.0
        psi0 = amplitude * np.exp(1j * (np.angle(psi0) + phase_noise))
    
    return psi0

def evolve_psi(psi0, Nt=Nt, dt=dt, g=1.0):
    """Split-step Fourier method (matches Script 1)"""
    psi = psi0.copy()
    for _ in range(Nt):
        psi_hat = np.fft.fft2(psi)
        psi_hat *= L_op
        psi = np.fft.ifft2(psi_hat)
        psi *= np.exp(1j * -g * np.abs(psi)**2 * dt)  # Nonlinear step
        psi_hat = np.fft.fft2(psi)
        psi_hat *= L_op
        psi = np.fft.ifft2(psi_hat)
    return psi


# =============================================================================
# FIGURE 1: Evolution visualization
# =============================================================================

if __name__ == "__main__":
    print("Generating Figure 1: Hi Jitter Evolution visualization (CORRECTED)...")
    
    jitter = 0.1
    seed = 36
    rng = np.random.default_rng(seed)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    class_names = ['Diagonal', 'Anti-diagonal', 'Vertical', 'Horizontal']
    
    for c in range(4):
        print(f"  Processing class {c}...")
        psi0 = make_initial_condition(c, rng, jitter=jitter)
        psiT = evolve_psi(psi0, Nt=Nt, dt=dt, g=1.0)
        
        # Initial amplitude
        ax0 = fig.add_subplot(gs[c, 0])
        im0 = ax0.imshow(np.abs(psi0), cmap='plasma', origin='lower',
                         extent=[-L/2, L/2, -L/2, L/2])
        ax0.set_ylabel(f'Class {c}\n{class_names[c]}', fontsize=10, fontweight='bold')
        if c == 0:
            ax0.set_title(r'Initial Amplitude $|\psi_0|$', fontsize=12, fontweight='bold')
        if c == 3:
            ax0.set_xlabel('x', fontsize=10)
        ax0.set_yticks([])
        ax0.set_xticks([])
        
        # Initial phase
        ax1 = fig.add_subplot(gs[c, 1])
        im1 = ax1.imshow(np.angle(psi0), cmap='twilight', origin='lower',
                         extent=[-L/2, L/2, -L/2, L/2], vmin=-np.pi, vmax=np.pi)
        if c == 0:
            ax1.set_title(r'Initial Phase arg$(\psi_0)$', fontsize=12, fontweight='bold')
        if c == 3:
            ax1.set_xlabel('x', fontsize=10)
        ax1.set_yticks([])
        ax1.set_xticks([])
        
        # Evolved amplitude
        ax2 = fig.add_subplot(gs[c, 2])
        im2 = ax2.imshow(np.abs(psiT), cmap='plasma', origin='lower',
                         extent=[-L/2, L/2, -L/2, L/2])
        if c == 0:
            ax2.set_title(r'Evolved Amplitude $|\psi_T|$', fontsize=12, fontweight='bold')
        if c == 3:
            ax2.set_xlabel('x', fontsize=10)
        ax2.set_yticks([])
        ax2.set_xticks([])
    
    # Colorbars
    cbar_ax0 = fig.add_axes([0.08, 0.05, 0.22, 0.015])
    cbar0 = fig.colorbar(im0, cax=cbar_ax0, orientation='horizontal')
    cbar0.set_label('Amplitude', fontsize=10)
    
    cbar_ax1 = fig.add_axes([0.38, 0.05, 0.22, 0.015])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Phase (radians)', fontsize=10)
    cbar1.set_ticks([-np.pi, 0, np.pi])
    cbar1.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    
    cbar_ax2 = fig.add_axes([0.68, 0.05, 0.22, 0.015])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Amplitude', fontsize=10)
    
    plt.savefig('2copped_paper1_figure1_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.savefig('2copped_paper1_figure1_CORRECTED.pdf', bbox_inches='tight')
    print("\n✓ Saved: copped_paper1_figure1_CORRECTED.png/pdf")
