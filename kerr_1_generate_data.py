"""
kerr_data_generator.py

Generate Kerr spacetime geodesic data and save to file.
Run this ONCE to create consistent dataset for training/testing.

Usage:
    python kerr_data_generator.py --samples 10000 --output kerr_geodesics.npy
"""

import numpy as np
import argparse
import json
from pathlib import Path


class KerrDataGenerator:
    """Generate geodesics for Kerr spacetime"""
    
    def __init__(self, M=1.0, a=0.9, seed=42):
        """
        Initialize Kerr data generator.
        
        Args:
            M: Black hole mass
            a: Spin parameter
            seed: Random seed for reproducibility
        """
        self.M = M
        self.a = a
        self.r_plus = M + np.sqrt(M**2 - a**2)  # outer horizon
        self.r_minus = M - np.sqrt(M**2 - a**2)  # inner horizon
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        print(f"Kerr Black Hole Parameters:")
        print(f"  M = {M}")
        print(f"  a = {a} (spin/mass ratio = {a/M:.3f})")
        print(f"  r_+ = {self.r_plus:.4f} (outer horizon)")
        print(f"  r_- = {self.r_minus:.4f} (inner horizon)")
        print(f"  Random seed = {seed}")
        
    def kerr_metric(self, r, theta):
        """Kerr metric components in Boyer-Lindquist coordinates"""
        Sigma = r**2 + self.a**2 * np.cos(theta)**2
        Delta = r**2 - 2*self.M*r + self.a**2
        
        g_tt = -(1 - 2*self.M*r/Sigma)
        g_rr = Sigma/Delta
        g_theta_theta = Sigma
        g_phi_phi = (r**2 + self.a**2 + 2*self.M*self.a**2*r*np.sin(theta)**2/Sigma) * np.sin(theta)**2
        g_tphi = -2*self.M*self.a*r*np.sin(theta)**2/Sigma
        
        return g_tt, g_rr, g_theta_theta, g_phi_phi, g_tphi
    
    def angular_velocity(self, r):
        """Frame dragging angular velocity at radius r"""
        omega = 2*self.a*self.M*r / (r**3 + self.a**2*r + 2*self.M*self.a**2)
        return omega * np.random.uniform(0.8, 1.2)
    
    def generate_equatorial_orbits(self, n=3000):
        """
        Generate equatorial plane geodesics (most important for Kerr).
        
        Args:
            n: Number of orbit segments
            
        Returns:
            Array of shape [n*steps, 4] with (t, r, θ, φ) coordinates
        """
        data = []
        
        for _ in range(n):
            t0 = np.random.uniform(0, 10)
            r0 = np.random.uniform(1.5*self.r_plus, 30*self.M)
            phi0 = np.random.uniform(0, 2*np.pi)
            
            # Circular orbit parameters (approximate)
            n_steps = np.random.randint(15, 40)
            omega = self.angular_velocity(r0)  # frame dragging
            
            for step in range(n_steps):
                t = t0 + step * 0.3
                r = r0 + np.random.uniform(-0.5, 0.5)  # slight radial oscillation
                r = max(r, 1.1*self.r_plus)
                theta = np.pi/2  # equatorial
                phi = (phi0 + omega * (t - t0)) % (2*np.pi)
                
                data.append([t, r, theta, phi])
        
        return np.array(data, dtype=np.float32)
    
    def generate_polar_orbits(self, n=2000):
        """
        Generate polar orbits crossing rotation axis.
        
        Args:
            n: Number of orbit segments
            
        Returns:
            Array of shape [n*steps, 4]
        """
        data = []
        
        for _ in range(n):
            t0 = np.random.uniform(0, 8)
            r0 = np.random.uniform(2*self.r_plus, 25*self.M)
            theta0 = np.random.uniform(0.1, np.pi-0.1)
            
            n_steps = np.random.randint(10, 30)
            
            for step in range(n_steps):
                t = t0 + step * 0.4
                r = r0 + np.random.uniform(-0.3, 0.3)
                r = max(r, 1.1*self.r_plus)
                theta = theta0 + 0.05*step
                theta = np.clip(theta, 0.1, np.pi-0.1)
                phi = 0.0  # along rotation axis
                
                data.append([t, r, theta, phi])
        
        return np.array(data, dtype=np.float32)
    
    def generate_radial_infall(self, n=1000):
        """
        Generate radial infall geodesics.
        
        Args:
            n: Number of trajectories
            
        Returns:
            Array of shape [n*steps, 4]
        """
        data = []
        
        for _ in range(n):
            t0 = 0.0
            r0 = np.random.uniform(5*self.r_plus, 40*self.M)
            theta = np.random.uniform(0.2, np.pi - 0.2)
            phi = np.random.uniform(0, 2*np.pi)
            
            n_steps = np.random.randint(20, 50)
            
            for step in range(n_steps):
                t = t0 + step * 0.2
                # Approximate radial infall (simplified)
                r = r0 - step * 0.5
                r = max(r, 1.05*self.r_plus)
                
                data.append([t, r, theta, phi])
        
        return np.array(data, dtype=np.float32)
    
    def generate_all(self, n_total=10000):
        """
        Generate complete Kerr geodesic dataset.
        
        Args:
            n_total: Total number of geodesic points
            
        Returns:
            Array of shape [n_total, 4] with (t, r, θ, φ) coordinates
        """
        print(f"\nGenerating {n_total} geodesic points...")
        
        # Mix of geodesic types
        n_equatorial = n_total // 2
        n_polar = n_total // 3
        n_infall = n_total // 6
        
        print(f"  - Equatorial orbits: ~{n_equatorial} points")
        print(f"  - Polar orbits: ~{n_polar} points")
        print(f"  - Radial infall: ~{n_infall} points")
        
        equatorial = self.generate_equatorial_orbits(n=n_equatorial // 25)
        polar = self.generate_polar_orbits(n=n_polar // 15)
        infall = self.generate_radial_infall(n=n_infall // 30)
        
        data = np.vstack([equatorial, polar, infall])
        
        # Shuffle
        indices = np.random.permutation(data.shape[0])
        data = data[indices]
        
        # Take exactly n_total points
        data = data[:n_total]
        
        print(f"\nDataset statistics:")
        print(f"  Shape: {data.shape}")
        print(f"  t range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
        print(f"  r range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
        print(f"  θ range: [{data[:, 2].min():.2f}, {data[:, 2].max():.2f}]")
        print(f"  φ range: [{data[:, 3].min():.2f}, {data[:, 3].max():.2f}]")
        
        return data


def save_dataset(data, metadata, output_path):
    """
    Save dataset and metadata to files.
    
    Args:
        data: Geodesic data array
        metadata: Dictionary with generation parameters
        output_path: Path to save .npy file
    """
    output_path = Path(output_path)
    
    # Save data
    np.save(output_path, data)
    print(f"\n✓ Data saved to: {output_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Kerr spacetime geodesic data'
    )
    parser.add_argument(
        '--samples', 
        type=int, 
        default=10000,
        help='Number of geodesic points to generate (default: 10000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='kerr_geodesics.npy',
        help='Output file path (default: kerr_geodesics.npy)'
    )
    parser.add_argument(
        '--M',
        type=float,
        default=1.0,
        help='Black hole mass (default: 1.0)'
    )
    parser.add_argument(
        '--a',
        type=float,
        default=0.9,
        help='Spin parameter (default: 0.9)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("KERR GEODESIC DATA GENERATOR")
    print("="*70)
    
    # Create generator
    generator = KerrDataGenerator(M=args.M, a=args.a, seed=args.seed)
    
    # Generate data
    data = generator.generate_all(n_total=args.samples)
    
    # Prepare metadata
    metadata = {
        'M': args.M,
        'a': args.a,
        'r_plus': float(generator.r_plus),
        'r_minus': float(generator.r_minus),
        'n_samples': int(data.shape[0]),
        'seed': args.seed,
        'coordinates': ['t', 'r', 'theta', 'phi'],
        'units': 'geometric (G=c=1)',
        'description': 'Kerr spacetime geodesics in Boyer-Lindquist coordinates'
    }
    
    # Save
    save_dataset(data, metadata, args.output)
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nTo use this data:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{args.output}')")
    print(f"  # data.shape = {data.shape}")


if __name__ == "__main__":
    main()
