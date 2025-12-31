"""Comprehensive test to verify unwrapping implementation is complete and correct"""
import numpy as np
from eclipsebin import EclipsingBinaryBinner

def test_unwrapping():
    """Test that unwrapping properly converts wrapped eclipses to unwrapped"""

    # Create wrapped light curve (from test fixture)
    np.random.seed(1)
    phases = np.linspace(0, 0.999, 10000)
    fluxes = np.ones_like(phases)
    # Primary eclipse (not wrapped)
    fluxes[4500:5000] = np.linspace(0.95, 0.8, 500)
    fluxes[5000:5500] = np.linspace(0.81, 0.95, 500)
    # Secondary eclipse (WRAPPED around 0/1 boundary)
    fluxes[0:300] = np.linspace(0.9, 0.95, 300)
    fluxes[9700:10000] = np.linspace(0.94, 0.91, 300)
    flux_errors = np.random.normal(0.01, 0.001, 10000)
    random_indices = np.random.choice(range(len(phases)), size=5000, replace=False)
    phases = phases[random_indices]
    fluxes = fluxes[random_indices]
    flux_errors = flux_errors[random_indices]

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=100, fraction_in_eclipse=0.2)

    print("\n" + "="*70)
    print("COMPREHENSIVE UNWRAPPING TEST")
    print("="*70)

    # Test 1: Phase shift was applied
    print("\nTest 1: Phase shift was applied")
    assert binner._phase_shift > 0, f"Expected phase shift > 0, got {binner._phase_shift}"
    print(f"  PASS: Phase shift = {binner._phase_shift:.6f}")

    # Test 2: Primary eclipse is unwrapped (should have been unwrapped to begin with)
    print("\nTest 2: Primary eclipse is unwrapped")
    assert binner.primary_eclipse[0] < binner.primary_eclipse[1], \
        f"Primary eclipse still wrapped: {binner.primary_eclipse}"
    print(f"  PASS: Primary [{binner.primary_eclipse[0]:.6f}, {binner.primary_eclipse[1]:.6f}]")

    # Test 3: Secondary eclipse is NOW unwrapped (was wrapped before)
    print("\nTest 3: Secondary eclipse is NOW unwrapped")
    assert binner.secondary_eclipse[0] < binner.secondary_eclipse[1], \
        f"Secondary eclipse still wrapped: {binner.secondary_eclipse}"
    print(f"  PASS: Secondary [{binner.secondary_eclipse[0]:.6f}, {binner.secondary_eclipse[1]:.6f}]")

    # Test 4: Eclipse minima are inside their boundaries
    print("\nTest 4: Eclipse minima are inside their boundaries")
    primary_min = binner.find_minimum_flux_phase()
    secondary_min = binner.find_secondary_minimum_phase()

    assert binner.primary_eclipse[0] < primary_min < binner.primary_eclipse[1], \
        f"Primary minimum {primary_min} not inside [{binner.primary_eclipse[0]}, {binner.primary_eclipse[1]}]"
    print(f"  PASS: Primary min {primary_min:.6f} inside eclipse")

    assert binner.secondary_eclipse[0] < secondary_min < binner.secondary_eclipse[1], \
        f"Secondary minimum {secondary_min} not inside [{binner.secondary_eclipse[0]}, {binner.secondary_eclipse[1]}]"
    print(f"  PASS: Secondary min {secondary_min:.6f} inside eclipse")

    # Test 5: Phases are within [0, 1]
    print("\nTest 5: All phases are within [0, 1]")
    assert np.all(binner.data['phases'] >= 0) and np.all(binner.data['phases'] <= 1), \
        f"Phases outside [0, 1] range"
    print(f"  PASS: Phase range [{binner.data['phases'].min():.6f}, {binner.data['phases'].max():.6f}]")

    # Test 6: Binning works without errors
    print("\nTest 6: Binning works without errors")
    bin_centers, bin_means, bin_errors, bin_numbers, _ = binner.calculate_bins()
    assert len(bin_centers) > 0, "No bins calculated"
    assert not np.any(np.isnan(bin_centers)), "NaN values in bin centers"
    assert not np.any(np.isnan(bin_means)), "NaN values in bin means"
    print(f"  PASS: {len(bin_centers)} bins calculated successfully")

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nSummary:")
    print(f"  - Unwrapping is working correctly")
    print(f"  - Phase shift of {binner._phase_shift:.4f} was applied")
    print(f"  - Both eclipses are now unwrapped (start < end)")
    print(f"  - Eclipse minima are properly located inside boundaries")
    print(f"  - All functionality works with unwrapped data")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_unwrapping()
