"""Verification script to check if phase unwrapping is working correctly"""
import numpy as np
from eclipsebin import EclipsingBinaryBinner

# Create wrapped light curve (from test fixture)
np.random.seed(1)
phases = np.linspace(0, 0.999, 10000)
fluxes = np.ones_like(phases)
fluxes[4500:5000] = np.linspace(0.95, 0.8, 500)
fluxes[5000:5500] = np.linspace(0.81, 0.95, 500)
fluxes[0:300] = np.linspace(0.9, 0.95, 300)  # Secondary eclipse
fluxes[9700:10000] = np.linspace(0.94, 0.91, 300)  # Wrap secondary eclipse
flux_errors = np.random.normal(0.01, 0.001, 10000)
random_indices = np.random.choice(range(len(phases)), size=5000, replace=False)
phases = phases[random_indices]
fluxes = fluxes[random_indices]
flux_errors = flux_errors[random_indices]

binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=100, fraction_in_eclipse=0.2)

print("="*60)
print("UNWRAPPING VERIFICATION")
print("="*60)
print(f"\nPhase shift applied: {binner._phase_shift}")
print(f"\nPrimary eclipse: {binner.primary_eclipse}")
print(f"Primary start < end: {binner.primary_eclipse[0] < binner.primary_eclipse[1]}")
print(f"\nSecondary eclipse: {binner.secondary_eclipse}")
print(f"Secondary start < end: {binner.secondary_eclipse[0] < binner.secondary_eclipse[1]}")

# Check if data was actually shifted
print(f"\n--- Data Shift Information ---")
print(f"Original phases range: [0, ~1.0)")
print(f"Shifted phases range: [{binner.data['phases'].min():.3f}, {binner.data['phases'].max():.3f}]")
print(f"Number of phases > 1.0: {np.sum(binner.data['phases'] > 1.0)}")

# Check if unwrapping actually occurred
if binner._phase_shift > 0:
    print(f"\nUNWRAPPING DETECTED: Phase shift = {binner._phase_shift:.3f}")
    print(f"This means {binner._phase_shift*100:.1f}% of the phase space was shifted")
else:
    print(f"\nNO UNWRAPPING: Phase shift = {binner._phase_shift}")

print("\n" + "="*60)
