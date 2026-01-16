"""
batch_generate_1000_original.py

Generate 1000 gravitational wave templates using the VALIDATED 
gravitational_wave_inference_STABLE_9.py code (98.5% match proven).

This script:
1. Uses your WORKING code (not the SDK)
2. Varies parameters systematically (spin, mass ratio, orbital radius)
3. Saves high-quality templates (>95% match)
4. Provides progress tracking and statistics

Usage:
    python batch_generate_1000_original.py --model kerr_vae.pth --n-templates 1000
"""

import sys
import os
import numpy as np
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Import your working code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravitational_wave_inference_STABLE_9 import GravitationalWaveInferenceFIXED


class ParameterSampler:
    """Generate diverse parameter combinations for template library"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def latin_hypercube_sample(self, n_samples, 
                                spin_range=(0.0, 0.95),
                                r_mult_range=(1.5, 3.0)):
        """
        Latin Hypercube Sampling for optimal parameter space coverage.
        
        Args:
            n_samples: Number of parameter combinations
            spin_range: (min, max) for spin parameter a
            r_mult_range: (min, max) for ISCO multipliers
        
        Returns:
            List of (a, r_multiplier) tuples
        """
        # Create evenly spaced intervals
        spin_intervals = np.linspace(0, 1, n_samples + 1)
        r_intervals = np.linspace(0, 1, n_samples + 1)
        
        # Random shuffle and sample
        samples = []
        
        spin_order = np.random.permutation(n_samples)
        r_order = np.random.permutation(n_samples)
        
        for i in range(n_samples):
            # Sample within each interval
            a = spin_range[0] + (spin_range[1] - spin_range[0]) * \
                np.random.uniform(spin_intervals[spin_order[i]], 
                                 spin_intervals[spin_order[i] + 1])
            
            r_mult = r_mult_range[0] + (r_mult_range[1] - r_mult_range[0]) * \
                     np.random.uniform(r_intervals[r_order[i]], 
                                      r_intervals[r_order[i] + 1])
            
            samples.append((a, r_mult))
        
        return samples


class TemplateLibrary:
    """Manage collection of generated templates"""
    
    def __init__(self, library_path):
        self.library_path = Path(library_path)
        self.library_path.mkdir(exist_ok=True)
        self.templates = []
        self.metadata = {
            'generation_date': datetime.now().isoformat(),
            'framework': 'Modak-Walawalkar (M-W)',
            'code_version': 'gravitational_wave_inference_STABLE_9.py',
            'validation': 'Post-Newtonian (quadrupole)',
        }
    
    def add_template(self, template_data):
        """Add a template to the library"""
        self.templates.append(template_data)
    
    def save(self):
        """Save library to disk"""
        # Save templates
        templates_file = self.library_path / 'templates.json'
        with open(templates_file, 'w') as f:
            json.dump(self.templates, f, indent=2)
        
        # Update and save metadata
        self.metadata['total_templates'] = len(self.templates)
        self.metadata['statistics'] = self._compute_statistics()
        
        metadata_file = self.library_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n✓ Library saved to: {self.library_path}")
        print(f"  Total templates: {len(self.templates)}")
    
    def _compute_statistics(self):
        """Compute library statistics"""
        if not self.templates:
            return {}
        
        match_plus = [t['match_plus'] for t in self.templates]
        match_cross = [t['match_cross'] for t in self.templates]
        gen_times = [t['generation_time'] for t in self.templates]
        
        return {
            'match_plus': {
                'mean': float(np.mean(match_plus)),
                'std': float(np.std(match_plus)),
                'min': float(np.min(match_plus)),
                'max': float(np.max(match_plus)),
            },
            'match_cross': {
                'mean': float(np.mean(match_cross)),
                'std': float(np.std(match_cross)),
                'min': float(np.min(match_cross)),
                'max': float(np.max(match_cross)),
            },
            'generation_time': {
                'mean': float(np.mean(gen_times)),
                'total': float(np.sum(gen_times)),
            },
            'quality_distribution': self._quality_distribution()
        }
    
    def _quality_distribution(self):
        """Count templates by quality tier"""
        counts = {
            'excellent_95plus': 0,
            'very_good_90to95': 0,
            'good_80to90': 0,
            'moderate_below80': 0
        }
        
        for t in self.templates:
            match = t['match_plus']
            if match >= 0.95:
                counts['excellent_95plus'] += 1
            elif match >= 0.90:
                counts['very_good_90to95'] += 1
            elif match >= 0.80:
                counts['good_80to90'] += 1
            else:
                counts['moderate_below80'] += 1
        
        return counts


def generate_single_template(inference, a, r_mult, template_id):
    """
    Generate a single template by modifying parameters in the working code.
    
    Args:
        inference: GravitationalWaveInferenceFIXED instance
        a: Spin parameter (0-0.95)
        r_mult: ISCO multiplier (1.5-3.0)
        template_id: Unique identifier
    
    Returns:
        Dictionary with template data
    """
    start_time = time.time()
    
    # Temporarily modify the inference object's parameters
    # We'll generate orbit with new parameters
    original_a = inference.a
    inference.a = a
    
    # Recompute ISCO for new spin
    inference.ISCO = inference.compute_isco()
    r_orbit = inference.ISCO * r_mult
    
    try:
        # Generate orbit
        orbit = inference.generate_circular_orbit(
            r_orbit=r_orbit, 
            n_orbits=5, 
            steps=1000
        )
        
        # Compute waveform with learned geometry
        h_plus_learned, h_cross_learned, t_learned = \
            inference.compute_waveform_with_learned_geometry(orbit, distance=100.0)
        
        # Compute PN reference
        h_plus_pn, h_cross_pn, t_pn = \
            inference.compute_exact_pn_waveform(orbit, distance=100.0)
        
        # Compute match
        match_plus = inference.compute_match(h_plus_learned, t_learned, 
                                             h_plus_pn, t_pn)
        match_cross = inference.compute_match(h_cross_learned, t_learned, 
                                              h_cross_pn, t_pn)
        
        generation_time = time.time() - start_time
        
        # Compute frequency
        omega = np.sqrt(inference.M) / (r_orbit**1.5 + a * np.sqrt(inference.M))
        frequency = omega / (2 * np.pi)
        
        template_data = {
            'template_id': template_id,
            'M': float(inference.M),
            'a': float(a),
            'q': 1.0,  # Equal mass for now
            'r_orbit': float(r_orbit),
            'r_mult': float(r_mult),
            'ISCO': float(inference.ISCO),
            'match_plus': float(match_plus),
            'match_cross': float(match_cross),
            'avg_match': float((match_plus + match_cross) / 2),
            'frequency': float(frequency),
            'generation_time': float(generation_time),
            'h_plus': h_plus_learned.tolist(),
            'h_cross': h_cross_learned.tolist(),
            'time': t_learned.tolist(),
        }
        
        # Restore original parameters
        inference.a = original_a
        inference.ISCO = inference.compute_isco()
        
        return template_data
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        # Restore original parameters
        inference.a = original_a
        inference.ISCO = inference.compute_isco()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate 1000 GW templates using validated code'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to kerr_vae.pth model'
    )
    parser.add_argument(
        '--n-templates',
        type=int,
        default=1000,
        help='Number of templates to generate (default: 1000)'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.95,
        help='Minimum match score to save (default: 0.95)'
    )
    parser.add_argument(
        '--library-path',
        type=str,
        default='gw_lego_1000_validated',
        help='Path to save library (default: gw_lego_1000_validated)'
    )
    parser.add_argument(
        '--spin-range',
        type=float,
        nargs=2,
        default=[0.0, 0.95],
        help='Spin range (default: 0.0 0.95)'
    )
    parser.add_argument(
        '--radius-range',
        type=float,
        nargs=2,
        default=[1.5, 3.0],
        help='ISCO multiplier range (default: 1.5 3.0)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=100,
        help='Save every N templates (default: 100)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("GW-LEGO 1000 TEMPLATE GENERATION")
    print("Using VALIDATED gravitational_wave_inference_STABLE_9.py")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Target templates: {args.n_templates}")
    print(f"Minimum quality: {args.min_quality*100:.0f}% match")
    print(f"Library path: {args.library_path}")
    print(f"\nParameter ranges:")
    print(f"  Spin (a): {args.spin_range[0]:.2f} to {args.spin_range[1]:.2f}")
    print(f"  Orbit: {args.radius_range[0]:.1f}×ISCO to {args.radius_range[1]:.1f}×ISCO")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize inference engine (your working code!)
    print("Initializing inference engine...")
    inference = GravitationalWaveInferenceFIXED(args.model)
    print("✓ Inference engine ready")
    
    # Initialize library
    library = TemplateLibrary(args.library_path)
    print(f"✓ Library initialized at: {args.library_path}")
    
    # Generate parameter samples
    print(f"\nGenerating {args.n_templates} parameter samples...")
    sampler = ParameterSampler()
    parameter_samples = sampler.latin_hypercube_sample(
        args.n_templates,
        spin_range=tuple(args.spin_range),
        r_mult_range=tuple(args.radius_range)
    )
    print(f"✓ Parameter samples ready")
    
    # Save parameter samples
    params_file = Path(args.library_path) / 'parameter_samples.json'
    with open(params_file, 'w') as f:
        json.dump({
            'total_samples': len(parameter_samples),
            'spin_range': args.spin_range,
            'radius_range': args.radius_range,
            'samples': [
                {'a': float(a), 'r_mult': float(r)} 
                for a, r in parameter_samples
            ]
        }, f, indent=2)
    print(f"✓ Parameter samples saved")
    
    # Generate templates
    print("\n" + "="*70)
    print("GENERATING TEMPLATES")
    print("="*70)
    
    start_time = time.time()
    generated = 0
    high_quality = 0
    failed = 0
    
    for i, (a, r_mult) in enumerate(parameter_samples, 1):
        template_id = f"T{i:04d}_a{a:.2f}_r{r_mult:.1f}"
        
        # Progress indicator
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (args.n_templates - i) / rate if rate > 0 else 0
            
            print(f"[{i}/{args.n_templates}] Progress: {100*i/args.n_templates:.1f}% | "
                  f"Rate: {rate:.1f} templates/min | "
                  f"Remaining: {remaining/60:.1f}min | "
                  f"Quality: {high_quality}/{generated}", end='\r')
        
        # Generate template
        template = generate_single_template(inference, a, r_mult, template_id)
        
        if template is None:
            failed += 1
            continue
        
        generated += 1
        
        # Check quality threshold
        if template['match_plus'] >= args.min_quality:
            library.add_template(template)
            high_quality += 1
            
            if i % 50 == 0:  # Detailed output every 50
                print(f"\n[{i}] {template_id}: "
                      f"match={template['match_plus']:.3f}, "
                      f"a={a:.2f}, r={r_mult:.1f}×ISCO, "
                      f"time={template['generation_time']:.2f}s")
        
        # Checkpoint
        if i % args.checkpoint_interval == 0:
            library.save()
            print(f"\n[Checkpoint] Saved {high_quality} high-quality templates")
    
    # Final save
    library.save()
    
    total_time = time.time() - start_time
    
    # Final statistics
    print("\n\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total attempted: {args.n_templates}")
    print(f"Successfully generated: {generated}")
    print(f"Failed: {failed}")
    print(f"High quality (≥{args.min_quality*100:.0f}%): {high_quality}")
    print(f"Acceptance rate: {100*high_quality/generated:.1f}%")
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average time: {total_time/generated:.2f}s per template")
    print(f"Rate: {generated*60/total_time:.1f} templates per minute")
    
    if high_quality > 0:
        stats = library.metadata['statistics']
        print(f"\nQuality statistics (saved templates):")
        print(f"  Match (h+): {stats['match_plus']['mean']:.4f} ± {stats['match_plus']['std']:.4f}")
        print(f"  Range: [{stats['match_plus']['min']:.4f}, {stats['match_plus']['max']:.4f}]")
        
        print(f"\nQuality distribution:")
        quality = stats['quality_distribution']
        print(f"  Excellent (≥95%): {quality['excellent_95plus']} "
              f"({100*quality['excellent_95plus']/high_quality:.1f}%)")
        print(f"  Very Good (90-95%): {quality['very_good_90to95']} "
              f"({100*quality['very_good_90to95']/high_quality:.1f}%)")
        print(f"  Good (80-90%): {quality['good_80to90']} "
              f"({100*quality['good_80to90']/high_quality:.1f}%)")
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    avg_time = total_time / generated if generated > 0 else 0
    speedup = 2.4e6 / avg_time if avg_time > 0 else 0
    
    print(f"Speedup vs NR: {speedup:.1e}×")
    print(f"10,000 templates: {avg_time * 10000 / 3600:.1f} hours estimated")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"1. Library saved at: {args.library_path}/")
    print(f"2. Review templates: cat {args.library_path}/metadata.json")
    print(f"3. Analyze quality distribution")
    print(f"4. Generate visualizations")
    print(f"5. Prepare for publication!")


if __name__ == "__main__":
    main()
