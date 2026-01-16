#!/usr/bin/env python
"""Test edge detection integration in eclipsebin."""
import numpy as np
import eclipsebin as ebin

print("="*70)
print("Testing Edge Detection Integration in eclipsebin")
print("="*70)

# Generate synthetic light curve with ellipsoidal variation
np.random.seed(42)
n_points = 1000
phases = np.linspace(0, 1, n_points, endpoint=False)
fluxes = np.ones(n_points)

# Add ellipsoidal variation (2× orbital frequency)
ellipsoidal_amplitude = 0.05
fluxes += ellipsoidal_amplitude * np.cos(2 * np.pi * 2 * phases)

# Add primary eclipse
primary_phase = 0.25
primary_width = 0.05
primary_depth = 0.2
fluxes[np.abs(phases - primary_phase) < primary_width/2] -= primary_depth

# Add secondary eclipse
secondary_phase = 0.75
secondary_width = 0.03
secondary_depth = 0.1
fluxes[np.abs(phases - secondary_phase) < secondary_width/2] -= secondary_depth

# Normalize to median = 1
flux_median = np.median(fluxes)
fluxes = fluxes / flux_median
flux_errors = np.ones(n_points) * 0.01 / flux_median

print(f"\nGenerated light curve:")
print(f"  Points: {n_points}")
print(f"  Ellipsoidal amplitude: {ellipsoidal_amplitude}")
print(f"  Primary eclipse: phase {primary_phase}, depth {primary_depth}")
print(f"  Secondary eclipse: phase {secondary_phase}, depth {secondary_depth}")

# Test 1: Traditional method (flux_return)
print("\n" + "-"*70)
print("Test 1: Traditional flux_return method")
print("-"*70)
try:
    binner_traditional = ebin.EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='flux_return'
    )
    primary_trad = binner_traditional.primary_eclipse
    secondary_trad = binner_traditional.secondary_eclipse
    print(f"Primary eclipse boundaries: {primary_trad}")
    print(f"Secondary eclipse boundaries: {secondary_trad}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Edge detection method
print("\n" + "-"*70)
print("Test 2: Edge detection method")
print("-"*70)
try:
    binner_edge = ebin.EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        edge_slope_threshold_percentile=90.0,
        edge_return_threshold_fraction=0.1
    )
    primary_edge = binner_edge.primary_eclipse
    secondary_edge = binner_edge.secondary_eclipse
    print(f"Primary eclipse boundaries: {primary_edge}")
    print(f"Secondary eclipse boundaries: {secondary_edge}")
    
    # Compare with expected values
    print(f"\nExpected primary: ~0.20-0.30")
    print(f"Expected secondary: ~0.72-0.78")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Binning with edge detection
print("\n" + "-"*70)
print("Test 3: Binning with edge detection")
print("-"*70)
try:
    binner = ebin.EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )
    bin_centers, bin_means, bin_errors = binner.bin_light_curve(plot=False)
    print(f"Successfully binned: {len(bin_centers)} bins")
    print(f"Bin centers range: [{bin_centers.min():.3f}, {bin_centers.max():.3f}]")
    print(f"Bin means range: [{bin_means.min():.3f}, {bin_means.max():.3f}]")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Integration test complete!")
print("="*70)

