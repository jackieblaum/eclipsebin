import numpy as np
from eclipsebin.binning import EclipsingBinaryBinner


def test_edge_detection_stores_diagnostics():
    """Test that edge detection diagnostics are stored and accessible."""
    # Create synthetic light curve with clear eclipse
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)
    # Add primary eclipse at phase 0.0
    eclipse_mask = (phases < 0.1) | (phases > 0.9)
    fluxes[eclipse_mask] = 0.8
    flux_errors = np.ones_like(phases) * 0.01

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=200, boundary_method="edge_detection"
    )

    # Check that diagnostics exist
    assert hasattr(binner, "_edge_diagnostics")
    assert binner._edge_diagnostics is not None

    # Check diagnostic content
    diag = binner._edge_diagnostics
    assert "slopes" in diag
    assert "smoothed_fluxes" in diag
    assert "threshold" in diag
    assert "return_threshold" in diag
    assert "smoothing_window" in diag
    assert "ingress_candidates" in diag
    assert "egress_candidates" in diag
    assert "detected_count" in diag

    # Verify data types
    assert isinstance(diag["slopes"], np.ndarray)
    assert isinstance(diag["threshold"], (float, np.floating))
    assert isinstance(diag["detected_count"], (int, np.integer))
