"""Tests for robust baseline calculation in edge detection."""

import numpy as np
from eclipsebin.binning import _detect_eclipse_edges_slope


def test_baseline_handles_outliers():
    """Test that baseline calculation is robust to outliers."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Add eclipse
    fluxes[450:550] = 0.8

    # Add outliers near eclipse edges
    fluxes[440] = 0.5  # Outlier before ingress
    fluxes[560] = 1.5  # Outlier after egress

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should still detect eclipse correctly
    assert len(boundaries) > 0
    # Eclipse should be roughly centered around phase 0.5
    ingress, egress = boundaries[0]
    center = (ingress + egress) / 2
    assert 0.45 < center < 0.55


def test_baseline_handles_sparse_edges():
    """Test baseline calculation with very few points near eclipse edges."""
    # Sparse sampling with gaps
    phases = np.concatenate(
        [
            np.linspace(0, 0.4, 50),
            np.linspace(0.45, 0.55, 200),  # Dense in eclipse
            np.linspace(0.6, 1.0, 50),
        ]
    )
    fluxes = np.ones_like(phases)
    eclipse_mask = (phases >= 0.48) & (phases <= 0.52)
    fluxes[eclipse_mask] = 0.75

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should handle sparse edges gracefully
    assert len(boundaries) >= 0  # May or may not detect, but shouldn't crash


def test_baseline_with_phase_boundary_eclipse():
    """Test baseline near phase=0/1 boundary."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Eclipse near phase 0
    fluxes[0:50] = 0.8
    fluxes[950:1000] = 0.8  # Wraps around

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should not crash
    assert isinstance(boundaries, list)
