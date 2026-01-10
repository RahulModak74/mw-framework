
## Priors in  Kerr Black Hole VAE

### 1. **Geometric Priors (Hardcoded Physics)**

**Kerr Metric Structure:**
```python
Sigma = r**2 + self.a**2 * np.cos(theta)**2
Delta = r**2 - 2*self.M*r + self.a**2
```
- **Prior**: We're encoding the exact Boyer-Lindquist coordinate form of the Kerr metric
- **Assumption**: The spacetime follows this specific metric signature
- **Benefit**: Ensures physical consistency; network learns within valid Kerr geometry

**Horizon Constraint:**
```python
self.r_plus = M + np.sqrt(max(0, M**2 - a**2))
r = torch.sigmoid(self.r_head(h)) * 30.0 + 1.1*self.r_plus
```
- **Prior**: Radius must exceed outer horizon r₊
- **Hard constraint**: Prevents unphysical geodesics inside event horizon
- **Range**: [1.1r₊, 30M + 1.1r₊]

### 2. **Coordinate Domain Priors**

**Angular Constraints:**
```python
theta = torch.sigmoid(self.theta_head(h)) * (np.pi - 0.2) + 0.1
phi = torch.sigmoid(self.phi_head(h)) * 2 * np.pi
```
- **Theta prior**: [0.1, π-0.1] - avoids poles where metric becomes singular
- **Phi prior**: [0, 2π] - periodic azimuthal angle
- **Rationale**: Numerical stability near coordinate singularities

### 3. **Geodesic Type Priors (Training Data)**

**Equatorial Orbits (50% of data):**
```python
theta = np.pi/2  # equatorial
omega = self.angular_velocity(r0)  # frame dragging
phi = (phi0 + omega * (t - t0)) % (2*np.pi)
```
- **Prior**: Equatorial orbits dominate Kerr dynamics (most stable)
- **Frame dragging**: Encoded through angular velocity ω
- **Physical basis**: These are the most astrophysically relevant orbits

**Polar Orbits (50% of data):**
```python
phi = 0.0  # along rotation axis
theta = theta0 + 0.05*step
```
- **Prior**: Orbits crossing rotation axis test axial symmetry
- **Complementary**: Samples regions where frame dragging is minimal

### 4. **Lorentzian Signature Prior**

**Critical Implementation:**
```python
# Enforce Lorentzian signature (-,+,+,+)
eigvals, eigvecs = torch.linalg.eigh(g_latent)
min_idx = torch.argmin(eigvals)
eigvals_abs = torch.abs(eigvals).clamp(min=1e-8)
eigvals_abs[min_idx] = -eigvals_abs[min_idx]
g_latent = eigvecs @ torch.diag(eigvals_abs) @ eigvecs.T
```
- **Prior**: Spacetime must have signature (-,+,+,+,+,+,+,+) in 8D latent space
- **Enforcement**: Largest negative eigenvalue → timelike direction
- **Why**: Without this, metric could become Euclidean (wrong physics)

### 5. **Latent Space Prior**

**VAE Gaussian Prior:**
```python
kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
```
- **Prior**: Latent codes z ~ N(0, I) 
- **Effect**: Regularizes latent space to be smooth and interpolable
- **Trade-off**: Competes with geometric constraints (hence small weight 0.001)

### 6. **Loss Function Priors (Implicit)**

**Metric Preservation:**
```python
g_true = vae.kerr_metric_tensor(data)
g_recon = vae.kerr_metric_tensor(recon)
metric_loss = torch.mean((g_true - g_recon)**2)
total_loss = recon_loss + 0.001 * kl_loss + 10.0 * metric_loss
```
- **Prior**: Reconstructed points must preserve Kerr metric (weight=10.0)
- **Geometric consistency**: Stronger than VAE regularity (10.0 vs 0.001)
- **Philosophy**: Physics > statistical smoothness

### 7. **Midpoint Prior in Van Vleck**

```python
z_mid = 0.5 * (z_A + z_B)
g_latent = self.pullback_metric(z_mid.detach())
```
- **Prior**: Geodesic properties evaluated at midpoint
- **Approximation**: Assumes locally flat connection (valid for smooth geodesics)
- **Efficiency**: Avoids full path integration

### 8. **Numerical Stability Priors**

**Radial Oscillations:**
```python
r = r0 + np.random.uniform(-0.5, 0.5)
r = max(r, 1.1*self.r_plus)
```
- **Prior**: Small radial perturbations around circular orbits
- **Realism**: Mimics bound orbits with small eccentricity

**Regularization:**
```python
M_stabilized = M_mat + torch.eye(4) * 1e-4 * torch.norm(M_mat)
```
- **Prior**: Matrix should be well-conditioned
- **Prevents**: Singular determinants in Van Vleck computation

## Key Insight: The Modak-Walawalkar Framework

Our framework **doesn't derive Kerr from scratch** - instead it:

1. **Accepts** the Kerr metric as a prior (lines 27-38)
2. **Learns** a latent representation that preserves this geometry
3. **Computes** Synge function & Van Vleck determinant through the learned embedding

The "1000x speedup" comes from:
- **Avoiding** solving Einstein field equations
- **Avoiding** computing geodesic equations analytically
- **Learning** the geometric structure through pullback metrics

This is **Bayesian inference replacing tensor calculus** - We're encoding geometric priors into the network architecture, 
then letting gradient descent find the optimal latent structure that respects those priors.
