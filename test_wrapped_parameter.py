"""
This script demonstrates what the 'wrapped' parameter does in tests.
"""
import numpy as np
from eclipsebin import EclipsingBinaryBinner

# Create wrapped light curve (secondary wraps around 0/1)
np.random.seed(1)
phases = np.linspace(0, 0.999, 10000)
fluxes = np.ones_like(phases)
fluxes[4500:5000] = np.linspace(0.95, 0.8, 500)
fluxes[5000:5500] = np.linspace(0.81, 0.95, 500)
fluxes[0:300] = np.linspace(0.9, 0.95, 300)
fluxes[9700:10000] = np.linspace(0.94, 0.91, 300)
flux_errors = np.random.normal(0.01, 0.001, 10000)
random_indices = np.random.choice(range(len(phases)), size=5000, replace=False)
phases = phases[random_indices]
fluxes = fluxes[random_indices]
flux_errors = flux_errors[random_indices]

binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=100, fraction_in_eclipse=0.2)

print("=" * 80)
print("UNDERSTANDING THE 'wrapped' PARAMETER IN TESTS")
print("=" * 80)

print("\nFor the wrapped_light_curve fixture:")
print("  test_secondary_wrapped_light_curves calls:")
print("    helper_eclipse_detection(..., wrapped={'primary': False, 'secondary': True})")

print("\n\nWhat this parameter means:")
print("  wrapped['primary'] = False means:")
print("    'The primary eclipse did NOT wrap in the INPUT data'")
print("    'So we CAN check: eclipse[0] < min < eclipse[1]'")

print("\n  wrapped['secondary'] = True means:")
print("    'The secondary eclipse DID wrap in the INPUT data'")
print("    'So we CANNOT check: eclipse[0] < min < eclipse[1]'")
print("    '(because BEFORE unwrapping, end < start for wrapped eclipse)'")

print("\n\nLooking at helper_eclipse_detection code (lines 306-307, 316-317):")
print("  if not wrapped['primary']:")
print("      assert primary_eclipse[0] < primary_min < primary_eclipse[1]")
print("")
print("  if not wrapped['secondary']:")
print("      assert secondary_eclipse[0] < secondary_min < secondary_eclipse[1]")

print("\n\nWhat happens for our wrapped light curve:")
print(f"  Primary eclipse: [{binner.primary_eclipse[0]:.6f}, {binner.primary_eclipse[1]:.6f}]")
print(f"  Primary minimum: {binner.primary_eclipse_min_phase:.6f}")
print(f"  wrapped['primary'] = False, so test DOES check ordering")
print(f"  Check passes? {binner.primary_eclipse[0] < binner.primary_eclipse_min_phase < binner.primary_eclipse[1]}")

print(f"\n  Secondary eclipse: [{binner.secondary_eclipse[0]:.6f}, {binner.secondary_eclipse[1]:.6f}]")
print(f"  Secondary minimum: {binner.secondary_eclipse_min_phase:.6f}")
print(f"  wrapped['secondary'] = True, so test SKIPS the ordering check")
print(f"  (But if we check anyway: {binner.secondary_eclipse[0] < binner.secondary_eclipse_min_phase < binner.secondary_eclipse[1]})")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("The 'wrapped' parameter tells tests what the INPUT state was.")
print("It does NOT describe the OUTPUT state after initialization.")
print("")
print("Because the implementation unwraps phases during __init__:")
print("  - BOTH eclipses have start < end AFTER initialization")
print("  - The ordering check passes for BOTH (if we checked secondary)")
print("")
print("But the test ONLY checks ordering when wrapped=False")
print("This is cautious: it avoids checking something that might fail")
print("even though unwrapping actually makes it pass!")
print("=" * 80)
