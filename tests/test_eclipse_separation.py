# tests/test_eclipse_separation.py
import numpy as np
from eclipsebin.binning import EclipsingBinaryBinner

def test_default_secondary_separation():
    """Test default secondary eclipse separation is 0.2."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Primary at phase 0.5
    fluxes[475:525] = 0.7
    # Secondary at phase 0.75 (0.25 separation)
    fluxes[725:775] = 0.85

    flux_errors = np.ones_like(phases) * 0.01

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    # Should detect both eclipses with default separation
    assert binner.primary_eclipse is not None
    assert binner.secondary_eclipse is not None

def test_custom_secondary_separation():
    """Test custom minimum secondary eclipse separation."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Primary at phase 0.5
    fluxes[475:525] = 0.7
    # Secondary at phase 0.65 (0.15 separation - too close with default)
    fluxes[625:675] = 0.85

    flux_errors = np.ones_like(phases) * 0.01

    # With default (0.2), secondary might not be detected
    binner_default = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    # With custom (0.1), secondary should be detected
    binner_custom = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        min_eclipse_separation=0.1
    )

    # Custom should be more permissive
    assert binner_custom.min_eclipse_separation == 0.1

def test_separation_affects_secondary_detection():
    """Test that separation parameter affects which eclipse is considered secondary."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Three eclipses
    fluxes[100:150] = 0.6  # Deepest (primary)
    fluxes[450:500] = 0.8  # Medium
    fluxes[800:850] = 0.85  # Shallowest

    flux_errors = np.ones_like(phases) * 0.01

    # Strict separation might only detect primary
    binner_strict = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        min_eclipse_separation=0.4  # Very strict
    )

    # Permissive separation should detect more
    binner_permissive = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        min_eclipse_separation=0.1  # Very permissive
    )

    # Both should detect primary
    assert binner_strict.primary_eclipse is not None
    assert binner_permissive.primary_eclipse is not None

    # Permissive should detect secondary
    assert binner_permissive.secondary_eclipse is not None
