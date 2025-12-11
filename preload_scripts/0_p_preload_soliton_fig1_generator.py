# use this in paper
# Script '0' for Figure 1

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Figure 1 for Paper 1: Evolution visualization (4 classes)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# Grid setup (copy from Paper 1)
# =============================================================================

L = 12.0
Nx, Ny = 64, 64
dx, dy = L / Nx, L / Ny
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="xy")
k2 = KX**2 + KY**2

dt = 0.01
Nt = 60
L_op = np.exp(-1j * k2 * dt / 2)

def sech(r):
    return 1.0 / np.cosh(r)

def place_grid(N, spacing, cx, cy):
    side = int(np.ceil(np.sqrt(N)))
    gx, gy = np.meshgrid(
        (np.arange(side) - (side - 1) / 2) * spacing,
        (np.arange(side) - (side - 1) / 2) * spacing,
    )
    centers = np.column_stack((gx.ravel(), gy.ravel()))
    return centers[:N] + np.array([cx, cy])

def make_initial_condition(class_id, rng, jitter=0.0):
    N_static = 1
    N_moving = 1
    spacing_static = 1.4
    spacing_moving = 1.4

    if class_id == 0:
        offset_static = (-2.5, 0.0)
        offset_moving = (2.5, 0.0)
        vx1, vy1 = 3.0, 0.0
        vx2, vy2 = -3.0, 0.0
    elif class_id == 1:
        offset_static = (0.0, -2.5)
        offset_moving = (0.0, 2.5)
        vx1, vy1 = 0.0, 3.0
        vx2, vy2 = 0.0, -3.0
    elif class_id == 2:
        offset_static = (-2.0, -2.0)
        offset_moving = (2.0, 2.0)
        vx1, vy1 = 2.0, 2.0
        vx2, vy2 = -2.0, -2.0
    elif class_id == 3:
        offset_static = (-2.0, 2.0)
        offset_moving = (2.0, -2.0)
        vx1, vy1 = 2.0, -2.0
        vx2, vy2 = -2.0, 2.0
    else:
        raise ValueError("class_id must be 0,1,2,3")

    static_pos = place_grid(N_static, spacing_static, *offset_static)
    moving_pos = place_grid(N_moving, spacing_moving, *offset_moving)

    psi1 = np.zeros((Ny, Nx), dtype=np.complex128)
    for cx, cy in static_pos:
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        psi1 += sech(r)
    psi1 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx * dy)
    psi1 *= np.exp(1j * (vx1 * X + vy1 * Y))

    psi2 = np.zeros((Ny, Nx), dtype=np.complex128)
    for cx, cy in moving_pos:
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        psi2 += sech(r)
    psi2 /= np.sqrt(np.sum(np.abs(psi2)**2) * dx * dy)
    psi2 *= np.exp(1j * (vx2 * X + vy2 * Y))

    psi0 = psi1 + psi2
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx * dy)

    if jitter > 0.0:
        phase_noise = rng.uniform(-jitter, jitter, (Ny, Nx))
        amp_noise = 0.1 * jitter * rng.standard_normal((Ny, Nx))
        psi0 = (np.abs(psi0) + amp_noise) * np.exp(
            1j * (np.angle(psi0) + phase_noise)
        )

    return psi0

def evolve_psi(psi0, Nt=Nt, dt=dt):
    psi = psi0.copy()
    for _ in range(Nt):
        psi_hat = np.fft.fft2(psi)
        psi_hat *= L_op
        psi = np.fft.ifft2(psi_hat)
        psi *= np.exp(-1j * np.abs(psi)**2 * dt)
        psi_hat = np.fft.fft2(psi)
        psi_hat *= L_op
        psi = np.fft.ifft2(psi_hat)
    return psi


# =============================================================================
# FIGURE 1: Evolution visualization
# =============================================================================

if __name__ == "__main__":
    print("Generating Figure 1: Evolution visualization...")
    
    jitter = 0.1
    seed = 36
    rng = np.random.default_rng(seed)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    class_names = ['Horizontal\nCollision', 'Vertical\nCollision', 
                   'Diagonal\nCollision', 'Anti-diagonal\nCollision']
    
    for c in range(4):
        print(f"  Processing class {c}...")
        psi0 = make_initial_condition(c, rng, jitter=jitter)
        psiT = evolve_psi(psi0, Nt=Nt, dt=dt)
        
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
    
    plt.savefig('preload_figures/2soli_etaV2paper1_figure1.png', dpi=300, bbox_inches='tight')
    plt.savefig('preload_figures/2_solietaV2paper1_figure1.pdf', bbox_inches='tight')
    print("\n✓ Saved: figures/2soli_etaV2paper1_figure1.png/pdf")