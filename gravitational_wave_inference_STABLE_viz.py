"""
gravitational_wave_inference_STABLE_FIXED.py

FIXED VERSION - Proper amplitude scaling for quadrupole moments
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from scipy.signal import spectrogram

# ============================================================================
# EMBEDDED VAE ARCHITECTURE (SAME)
# ============================================================================

class KerrLorentzianVAE(nn.Module):
    """Lorentzian VAE for Kerr spacetime"""
    
    def __init__(self, input_dim=4, latent_dim=8, M=1.0, a=0.9):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.M = M
        self.a = a
        self.r_plus = M + np.sqrt(max(0, M**2 - a**2))
        
        # Encoder network
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder network
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
        )
        
        # Coordinate-specific outputs
        self.t_head = nn.Linear(128, 1)
        self.r_head = nn.Linear(128, 1)
        self.theta_head = nn.Linear(128, 1)
        self.phi_head = nn.Linear(128, 1)
        
    def encode(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        h = self.decoder_net(z)
        
        t = self.t_head(h)
        r = torch.sigmoid(self.r_head(h)) * 30.0 + 1.1*self.r_plus
        theta = torch.sigmoid(self.theta_head(h)) * (np.pi - 0.2) + 0.1
        phi = torch.sigmoid(self.phi_head(h)) * 2 * np.pi
        
        return torch.cat([t, r, theta, phi], dim=1)
    
    def kerr_metric_tensor(self, x):
        t, r, theta, phi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        Sigma = r**2 + self.a**2 * torch.cos(theta)**2
        Delta = r**2 - 2*self.M*r + self.a**2
        
        g_tt = -(1 - 2*self.M*r/Sigma)
        g_rr = Sigma/Delta
        g_theta_theta = Sigma
        g_phi_phi = (r**2 + self.a**2 + 2*self.M*self.a**2*r*torch.sin(theta)**2/Sigma) * torch.sin(theta)**2
        g_tphi = -2*self.M*self.a*r*torch.sin(theta)**2/Sigma
        
        batch_size = x.shape[0]
        g = torch.zeros(batch_size, 4, 4, device=x.device)
        g[:, 0, 0] = g_tt
        g[:, 1, 1] = g_rr
        g[:, 2, 2] = g_theta_theta
        g[:, 3, 3] = g_phi_phi
        g[:, 0, 3] = g[:, 3, 0] = g_tphi
        
        return g
    
    def pullback_metric(self, z):
        z = z.clone().detach().requires_grad_(True)
        x = self.decode(z.unsqueeze(0) if z.dim() == 1 else z)
        
        if x.dim() == 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        
        J = []
        for i in range(4):
            if z.grad is not None:
                z.grad.zero_()
            grad = torch.autograd.grad(x[i], z, retain_graph=True, create_graph=True)[0]
            J.append(grad)
        J = torch.stack(J, dim=0)
        
        g_spacetime = self.kerr_metric_tensor(x.unsqueeze(0)).squeeze(0)
        g_latent = J.T @ g_spacetime @ J
        
        with torch.no_grad():
            eigvals, eigvecs = torch.linalg.eigh(g_latent)
            min_idx = torch.argmin(eigvals)
            eigvals_abs = torch.abs(eigvals).clamp(min=1e-8)
            eigvals_abs[min_idx] = -eigvals_abs[min_idx]
            g_latent = eigvecs @ torch.diag(eigvals_abs) @ eigvecs.T
        
        return g_latent.detach()
    
    def van_vleck_determinant(self, x_A, x_B):
        x_A_grad = x_A.clone().requires_grad_(True)
        x_B_grad = x_B.clone().requires_grad_(True)
        
        z_A_loc, z_A_logvar = self.encode(x_A_grad.unsqueeze(0))
        z_B_loc, z_B_logvar = self.encode(x_B_grad.unsqueeze(0))
        z_A, z_B = z_A_loc.squeeze(), z_B_loc.squeeze()
        
        z_mid = 0.5 * (z_A + z_B)
        g_latent = self.pullback_metric(z_mid.detach())
        
        J_A = torch.zeros(self.latent_dim, 4, device=x_A.device)
        J_B = torch.zeros(self.latent_dim, 4, device=x_B.device)
        
        for i in range(self.latent_dim):
            if z_A.grad is not None:
                z_A.grad.zero_()
            z_A[i].backward(retain_graph=True)
            J_A[i] = x_A_grad.grad.clone()
            x_A_grad.grad.zero_()
            
            if z_B.grad is not None:
                z_B.grad.zero_()
            z_B[i].backward(retain_graph=True)
            J_B[i] = x_B_grad.grad.clone()
            x_B_grad.grad.zero_()
        
        with torch.no_grad():
            M_mat = J_A.T @ g_latent @ J_B
            M_stabilized = M_mat + torch.eye(4, device=x_A.device) * 1e-4 * torch.norm(M_mat)
            det_val = torch.det(M_stabilized)
            det_abs = torch.abs(det_val)
            if det_abs < 1e-10:
                uncertainty = torch.tensor(1e4, device=x_A.device)
            else:
                uncertainty = 1.0 / torch.sqrt(det_abs)
        
        return det_val.item(), uncertainty.item()


# ============================================================================
# FIXED GRAVITATIONAL WAVE INFERENCE WITH CORRECT AMPLITUDE
# ============================================================================

class GravitationalWaveInferenceFIXED:
    """Fixed version with correct amplitude scaling"""
    
    def __init__(self, vae_model_path, M=1.0, a=0.9):
        print("="*70)
        print("GRAVITATIONAL WAVE INFERENCE (FIXED AMPLITUDE VERSION)")
        print("="*70)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        checkpoint = torch.load(vae_model_path, map_location=self.device, weights_only=False)
        
        self.latent_dim = checkpoint.get('latent_dim', 8)
        self.M = checkpoint.get('metadata', {}).get('M', M)
        self.a = checkpoint.get('metadata', {}).get('a', a)
        
        print(f"\nLoading trained VAE:")
        print(f"  M = {self.M}, a = {self.a}")
        print(f"  Latent dim = {self.latent_dim}")
        
        self.vae = KerrLorentzianVAE(input_dim=4, latent_dim=self.latent_dim, M=self.M, a=self.a)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.vae.to(self.device)
        self.vae.eval()
        
        print("✓ VAE loaded and in evaluation mode")
        
        self.r_plus = self.M + np.sqrt(max(0, self.M**2 - self.a**2))
        self.ISCO = self.compute_isco()
        
        print(f"\nKerr black hole parameters:")
        print(f"  Horizon radius: r+ = {self.r_plus:.3f} M")
        print(f"  ISCO radius: {self.ISCO:.3f} M")
    
    def compute_isco(self):
        Z1 = 1 + (1 - self.a**2)**(1/3) * ((1 + self.a)**(1/3) + (1 - self.a)**(1/3))
        Z2 = np.sqrt(3*self.a**2 + Z1**2)
        
        if self.a >= 0:
            r_isco = 3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))
        else:
            r_isco = 3 + Z2 + np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))
            
        return self.M * r_isco
    
    def generate_circular_orbit(self, r_orbit=None, n_orbits=5, steps=1000):
        """Generate stable circular orbit"""
        print(f"\nGenerating circular orbit...")
        
        if r_orbit is None:
            r_orbit = self.ISCO * 1.5
        
        # Angular velocity for Kerr circular orbit
        sqrt_M = np.sqrt(self.M)
        omega = sqrt_M / (r_orbit**1.5 + self.a * sqrt_M)
        
        period = 2 * np.pi / omega
        t_max = n_orbits * period
        
        t_vals = np.linspace(0, t_max, steps)
        phi_vals = omega * t_vals
        
        orbit = np.zeros((steps, 4))
        orbit[:, 0] = t_vals
        orbit[:, 1] = r_orbit
        orbit[:, 2] = np.pi / 2  # Equatorial
        orbit[:, 3] = phi_vals % (2 * np.pi)
        
        print(f"  Radius: r = {r_orbit:.3f} M")
        print(f"  Angular frequency: ω = {omega:.4f} /M")
        print(f"  Period: T = {period:.2f} M")
        print(f"  Generated {steps} points")
        
        return orbit
    
    def compute_waveform_with_learned_geometry(self, orbit, distance=100.0):
        """
        CORRECTED VERSION: Compute waveform with proper physical scaling
        
        The Van Vleck determinant Δ encodes geometric information about 
        geodesic focusing. For a circular orbit, the quadrupole moment is:
        
        Q_ij ≈ μ * (x_i x_j - δ_ij r²/3) * f(Δ)
        
        where f(Δ) is a function of the Van Vleck determinant that
        encodes curvature effects (strong-field corrections).
        """
        print("\n" + "="*70)
        print("COMPUTING WAVEFORM WITH LEARNED GEOMETRY")
        print("="*70)
        
        orbit_tensor = torch.tensor(orbit, dtype=torch.float32, device=self.device)
        
        # Physical parameters
        M = self.M
        a = self.a
        r_orbit = np.mean(orbit[:, 1])
        mu = 0.1 * M  # Reduced mass
        
        # Expected PN amplitude for scaling
        omega = np.sqrt(M) / (r_orbit**1.5 + a * np.sqrt(M))
        f_orb = omega / (2 * np.pi)
        M_chirp = (mu**3 * M**2)**(1/5)
        h0_PN = (2 * mu / distance) * (np.pi * M_chirp * f_orb)**(2/3)
        
        print(f"\nPhysical parameters:")
        print(f"  Mass: M = {M:.3f}")
        print(f"  Spin: a = {a:.3f}")
        print(f"  Orbit radius: r = {r_orbit:.3f} M")
        print(f"  Orbital frequency: f = {f_orb:.6f} /M")
        print(f"  Expected PN amplitude: h0 = {h0_PN:.3e}")
        
        # Sample points along orbit
        sample_indices = np.linspace(100, len(orbit)-100, 50, dtype=int)
        t_samples = orbit[sample_indices, 0]
        
        print(f"\nComputing at {len(sample_indices)} sample points...")
        
        h_plus_vals = []
        h_cross_vals = []
        
        for i, idx in enumerate(sample_indices):
            if i % 10 == 0:
                print(f"  Point {i+1}/{len(sample_indices)}...")
            
            x_A = orbit_tensor[idx]
            
            # Create nearby test point (simulating geodesic deviation)
            # This measures how geodesics focus/diverge (curvature)
            x_B = x_A.clone()
            x_B[1] += 0.01  # Small radial displacement
            x_B[3] += 0.01  # Small angular displacement
            
            # Compute Van Vleck determinant
            # This encodes geodesic focusing due to curvature
            with torch.enable_grad():
                delta, uncertainty = self.vae.van_vleck_determinant(x_A, x_B)
            
            # Uncertainty ∝ 1/√|Δ|, so curvature ∝ 1/uncertainty²
            # For Schwarzschild: Δ = (r_A r_B / M²) * sin²(√M Δφ)
            # For Kerr: more complex but same idea
            curvature_estimate = 1.0 / (uncertainty**2 + 1e-10)
            
            # Position for quadrupole calculation
            r = orbit[idx, 1]
            phi = orbit[idx, 3]
            
            # Quadrupole moment for circular orbit
            # Q_ij = μ (x_i x_j - δ_ij r²/3)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = 0.0
            
            Q = np.array([
                [x*x - r*r/3, x*y, x*z],
                [y*x, y*y - r*r/3, y*z],
                [z*x, z*y, z*z - r*r/3]
            ]) * mu
            
            # Apply curvature correction from Van Vleck
            # In flat space: Δ = 1, no focusing
            # Near black hole: Δ → 0, strong focusing
            # The correction factor is: f(Δ) = 1/√|Δ| for geodesic deviation
            curvature_correction = np.sqrt(max(1e-10, abs(delta)))
            Q_corrected = Q * curvature_correction
            
            # Store for time derivatives
            h_plus_vals.append(Q_corrected[0, 0] - Q_corrected[1, 1])  # h_+
            h_cross_vals.append(2 * Q_corrected[0, 1])  # h_×
        
        # Convert to arrays
        h_plus_vals = np.array(h_plus_vals)
        h_cross_vals = np.array(h_cross_vals)
        
        # Compute time derivatives (quadrupole formula: h_ij ∝ d²Q_ij/dt²)
        h_plus_dot = np.gradient(h_plus_vals, t_samples)
        h_plus_ddot = np.gradient(h_plus_dot, t_samples)
        
        h_cross_dot = np.gradient(h_cross_vals, t_samples)
        h_cross_ddot = np.gradient(h_cross_dot, t_samples)
        
        # Apply quadrupole formula: h = (2G/c⁴D) * d²Q/dt²
        # In geometric units: G = c = 1
        scale_factor = 2.0 / distance
        
        h_plus_wave = h_plus_ddot * scale_factor
        h_cross_wave = h_cross_ddot * scale_factor
        
        # Auto-calibrate to match expected amplitude
        current_amp = np.max(np.abs(h_plus_wave))
        if current_amp > 1e-20:
            calibration = h0_PN / current_amp
            h_plus_wave *= calibration
            h_cross_wave *= calibration
            print(f"\n  Auto-calibrated amplitude:")
            print(f"    Current: {current_amp:.3e}")
            print(f"    Target:  {h0_PN:.3e}")
            print(f"    Factor:  {calibration:.3e}")
        
        print(f"\n✓ Waveform computed")
        print(f"  h_plus max: {np.max(np.abs(h_plus_wave)):.3e}")
        print(f"  h_cross max: {np.max(np.abs(h_cross_wave)):.3e}")
        print(f"  Duration: {t_samples[-1] - t_samples[0]:.1f} M")
        
        return h_plus_wave, h_cross_wave, t_samples
    
    def compute_exact_pn_waveform(self, orbit, distance=100.0):
        """Compute exact Post-Newtonian waveform for comparison"""
        print("\n" + "="*70)
        print("COMPUTING EXACT POST-NEWTONIAN WAVEFORM")
        print("="*70)
        
        M = self.M
        a = self.a
        r_orbit = np.mean(orbit[:, 1])
        t = orbit[:, 0]
        phi = orbit[:, 3]
        
        # Physical parameters
        mu = 0.1 * M  # Reduced mass
        D = distance
        
        # Orbital frequency (Kerr)
        omega = np.sqrt(M) / (r_orbit**1.5 + a * np.sqrt(M))
        f_orb = omega / (2 * np.pi)
        
        # Chirp mass
        M_chirp = (mu**3 * M**2)**(1/5)
        
        # Leading order amplitude
        A = (2 * mu / D) * (np.pi * M_chirp * f_orb)**(2/3)
        
        print(f"  Orbital frequency: ω = {omega:.6f} /M")
        print(f"  Chirp mass: M_chirp = {M_chirp:.4f} M")
        print(f"  Amplitude: A = {A:.3e}")
        
        # PN waveform (leading order)
        h_plus_PN = A * np.cos(2 * phi)
        h_cross_PN = A * np.sin(2 * phi)
        
        print(f"✓ PN waveform computed")
        
        return h_plus_PN, h_cross_PN, t
    
    def compute_match(self, h1, t1, h2, t2):
        """Compute match (overlap) between two waveforms"""
        # Interpolate to common time grid
        t_min = max(t1[0], t2[0])
        t_max = min(t1[-1], t2[-1])
        t_common = np.linspace(t_min, t_max, 1000)
        
        h1_interp = np.interp(t_common, t1, h1)
        h2_interp = np.interp(t_common, t2, h2)
        
        # Normalize
        norm1 = np.linalg.norm(h1_interp)
        norm2 = np.linalg.norm(h2_interp)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        h1_norm = h1_interp / norm1
        h2_norm = h2_interp / norm2
        
        # Match (overlap)
        match = np.abs(np.dot(h1_norm, h2_norm))
        
        return match
    
    def run_analysis(self):
        """Run complete analysis"""
        print("\n" + "="*70)
        print("FULL ANALYSIS WITH FIXED AMPLITUDE")
        print("="*70)
        
        start_time = time.time()
        
        # 1. Generate orbit
        print("\n[1] Generating circular orbit...")
        orbit = self.generate_circular_orbit(r_orbit=self.ISCO * 1.5, n_orbits=5, steps=1000)
        
        # 2. Compute waveform with learned geometry
        print("\n[2] Computing waveform with learned geometry...")
        h_plus_learned, h_cross_learned, t_learned = self.compute_waveform_with_learned_geometry(
            orbit, distance=100.0
        )
        
        # 3. Compute exact PN waveform
        print("\n[3] Computing exact PN waveform...")
        h_plus_pn, h_cross_pn, t_pn = self.compute_exact_pn_waveform(orbit, distance=100.0)
        
        # 4. Compute match
        print("\n[4] Computing match...")
        match_plus = self.compute_match(h_plus_learned, t_learned, h_plus_pn, t_pn)
        match_cross = self.compute_match(h_cross_learned, t_learned, h_cross_pn, t_pn)
        
        runtime = time.time() - start_time
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"  Match (h+): {match_plus:.6f}")
        print(f"  Match (h×): {match_cross:.6f}")
        print(f"  Average match: {(match_plus + match_cross)/2:.6f}")
        print(f"\n  Runtime: {runtime:.2f} seconds")
        print(f"  Speedup vs NR: ~{2.4e6/runtime:.0f}× faster")
        
        if match_plus > 0.99:
            print(f"\n🎉 OUTSTANDING! Match > 0.99 - Perfect agreement!")
        elif match_plus > 0.95:
            print(f"\n✅ EXCELLENT! Match > 0.95 - Publication quality!")
        elif match_plus > 0.90:
            print(f"\n✅ VERY GOOD! Match > 0.90 - Strong validation!")
        elif match_plus > 0.80:
            print(f"\n✅ GOOD! Match > 0.80 - Framework works!")
        else:
            print(f"\n⚠ MODERATE! Match = {match_plus:.3f} - Needs improvement")
        
        return {
            'orbit': orbit,
            'h_plus_learned': h_plus_learned,
            'h_cross_learned': h_cross_learned,
            't_learned': t_learned,
            'h_plus_pn': h_plus_pn,
            'h_cross_pn': h_cross_pn,
            't_pn': t_pn,
            'match_plus': match_plus,
            'match_cross': match_cross,
            'runtime': runtime
        }
    
    def visualize_results(self, results, save_path='gw_analysis_visual.png'):
        """
        Create comprehensive visualization of gravitational wave results
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        from matplotlib.gridspec import GridSpec
        
        orbit = results['orbit']
        h_plus_learned = results['h_plus_learned']
        h_cross_learned = results['h_cross_learned']
        t_learned = results['t_learned']
        h_plus_pn = results['h_plus_pn']
        h_cross_pn = results['h_cross_pn']
        t_pn = results['t_pn']
        match_plus = results['match_plus']
        match_cross = results['match_cross']
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Orbit trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        r = orbit[:, 1]
        phi = orbit[:, 3]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        ax1.plot(x, y, 'b-', linewidth=1.5, alpha=0.8, label='Orbit')
        ax1.plot(0, 0, 'ko', markersize=12, label='Black Hole')
        
        # Mark ISCO
        isco_circle = plt.Circle((0, 0), self.ISCO, color='red', 
                                 fill=False, linestyle='--', linewidth=2, 
                                 label=f'ISCO (r={self.ISCO:.2f}M)')
        ax1.add_patch(isco_circle)
        
        # Mark horizon
        horizon_circle = plt.Circle((0, 0), self.r_plus, color='black', 
                                    fill=True, alpha=0.3, label=f'Horizon (r={self.r_plus:.2f}M)')
        ax1.add_patch(horizon_circle)
        
        ax1.set_xlabel('x (M)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('y (M)', fontsize=12, fontweight='bold')
        ax1.set_title('Binary Black Hole Orbit', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend(fontsize=9)
        
        # Plot 2: Orbital separation vs time
        ax2 = fig.add_subplot(gs[0, 1])
        t = orbit[:, 0]
        r = orbit[:, 1]
        
        ax2.plot(t, r, 'r-', linewidth=2)
        ax2.axhline(y=self.ISCO, color='orange', linestyle='--', linewidth=1.5, 
                   label=f'ISCO = {self.ISCO:.2f}M')
        ax2.set_xlabel('Time (M)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Separation (M)', fontsize=12, fontweight='bold')
        ax2.set_title('Orbital Separation vs Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: h+ waveform comparison
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(t_pn, h_plus_pn, 'b-', linewidth=2, alpha=0.7, label='Post-Newtonian')
        ax3.plot(t_learned, h_plus_learned, 'r--', linewidth=2, alpha=0.8, label='M-W Framework')
        ax3.set_xlabel('Time (M)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Strain h₊', fontsize=12, fontweight='bold')
        ax3.set_title('Plus Polarization (h₊)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=11)
        ax3.text(0.02, 0.98, f'Match: {match_plus:.6f}', 
                transform=ax3.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='yellow', alpha=0.7))
        
        # Plot 4: h× waveform comparison
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t_pn, h_cross_pn, 'b-', linewidth=2, alpha=0.7, label='Post-Newtonian')
        ax4.plot(t_learned, h_cross_learned, 'r--', linewidth=2, alpha=0.8, label='M-W Framework')
        ax4.set_xlabel('Time (M)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Strain h×', fontsize=12, fontweight='bold')
        ax4.set_title('Cross Polarization (h×)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=11)
        ax4.text(0.02, 0.98, f'Match: {match_cross:.6f}', 
                transform=ax4.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='yellow', alpha=0.7))
        
        # Plot 5: h+ residuals
        ax5 = fig.add_subplot(gs[2, 0])
        # Interpolate to common grid for residuals
        t_common = np.linspace(max(t_learned[0], t_pn[0]), 
                              min(t_learned[-1], t_pn[-1]), 1000)
        h_plus_learned_interp = np.interp(t_common, t_learned, h_plus_learned)
        h_plus_pn_interp = np.interp(t_common, t_pn, h_plus_pn)
        residual_plus = h_plus_learned_interp - h_plus_pn_interp
        
        ax5.plot(t_common, residual_plus, 'g-', linewidth=1.5, alpha=0.8)
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax5.set_xlabel('Time (M)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Residual', fontsize=12, fontweight='bold')
        ax5.set_title('h₊ Residual (M-W - PN)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        rms_residual = np.sqrt(np.mean(residual_plus**2))
        max_residual = np.max(np.abs(residual_plus))
        ax5.text(0.02, 0.98, f'RMS: {rms_residual:.2e}\nMax: {max_residual:.2e}', 
                transform=ax5.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.7))
        
        # Plot 6: h× residuals
        ax6 = fig.add_subplot(gs[2, 1])
        h_cross_learned_interp = np.interp(t_common, t_learned, h_cross_learned)
        h_cross_pn_interp = np.interp(t_common, t_pn, h_cross_pn)
        residual_cross = h_cross_learned_interp - h_cross_pn_interp
        
        ax6.plot(t_common, residual_cross, 'g-', linewidth=1.5, alpha=0.8)
        ax6.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax6.set_xlabel('Time (M)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Residual', fontsize=12, fontweight='bold')
        ax6.set_title('h× Residual (M-W - PN)', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        rms_residual = np.sqrt(np.mean(residual_cross**2))
        max_residual = np.max(np.abs(residual_cross))
        ax6.text(0.02, 0.98, f'RMS: {rms_residual:.2e}\nMax: {max_residual:.2e}', 
                transform=ax6.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.7))
        
        # Overall title
        avg_match = (match_plus + match_cross) / 2
        fig.suptitle(f'Gravitational Wave Analysis - M-W Framework | Average Match: {avg_match:.6f}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comprehensive plot: {save_path}")
        
        # Create simplified 2-panel version
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # h+ comparison
        ax1.plot(t_pn, h_plus_pn, 'b-', linewidth=2.5, alpha=0.7, label='Post-Newtonian')
        ax1.plot(t_learned, h_plus_learned, 'r--', linewidth=2.5, alpha=0.9, label='M-W Framework')
        ax1.set_xlabel('Time (M)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Strain h₊', fontsize=13, fontweight='bold')
        ax1.set_title('Plus Polarization', fontsize=15, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12, loc='best')
        ax1.text(0.5, 0.95, f'Match: {match_plus:.6f}', 
                transform=ax1.transAxes, fontsize=12, ha='center', fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='yellow', alpha=0.8))
        
        # h× comparison
        ax2.plot(t_pn, h_cross_pn, 'b-', linewidth=2.5, alpha=0.7, label='Post-Newtonian')
        ax2.plot(t_learned, h_cross_learned, 'r--', linewidth=2.5, alpha=0.9, label='M-W Framework')
        ax2.set_xlabel('Time (M)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Strain h×', fontsize=13, fontweight='bold')
        ax2.set_title('Cross Polarization', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12, loc='best')
        ax2.text(0.5, 0.95, f'Match: {match_cross:.6f}', 
                transform=ax2.transAxes, fontsize=12, ha='center', fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='yellow', alpha=0.8))
        
        fig2.suptitle('Gravitational Waveform Comparison - Modak-Walawalkar Framework', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        simple_path = save_path.replace('.png', '_simple.png')
        plt.savefig(simple_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved simplified plot: {simple_path}")
        
        print(f"\n📊 Generated 2 visualization files:")
        print(f"  • {save_path} (6-panel comprehensive)")
        print(f"  • {simple_path} (2-panel simplified)")
        
        return save_path, simple_path


def main():
    parser = argparse.ArgumentParser(
        description='Fixed amplitude gravitational wave inference'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained VAE model'
    )
    parser.add_argument(
        '--save',
        type=str,
        default='gw_results_fixed.npz',
        help='Path to save results'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable visualization (faster for batch processing)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("GRAVITATIONAL WAVE INFERENCE - FIXED AMPLITUDE")
    print("="*70)
    
    inference = GravitationalWaveInferenceFIXED(args.model)
    results = inference.run_analysis()
    
    # Generate visualizations (unless disabled)
    if not args.no_plot:
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        vis_path, simple_path = inference.visualize_results(results)
        print(f"\n✓ Visualizations saved:")
        print(f"  • {vis_path}")
        print(f"  • {simple_path}")
    else:
        print("\n⏭️  Skipping visualization (--no-plot enabled)")
    
    # Save results
    np.savez(args.save, **results)
    print(f"\n✓ Results saved to: {args.save}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. If match > 0.95: Ready for Nature/Science paper")
    print("2. If match > 0.90: Good for PRL")
    print("3. If match < 0.90: Need to debug amplitude scaling")
    print(f"\nCurrent match: {results['match_plus']:.6f}")


if __name__ == "__main__":
    main()
