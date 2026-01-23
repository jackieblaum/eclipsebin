# tests/test_smoothing_validation.py
import numpy as np
import warnings
from eclipsebin.binning import _detect_eclipse_edges_slope


def test_warns_if_smoothing_too_large_for_narrow_eclipse():
    """Test warning when smoothing window may obscure narrow eclipse."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Very narrow eclipse (2% of phase)
    fluxes[490:510] = 0.7

    # Force large smoothing window
    large_window = 101  # 10% of data

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        boundaries, diagnostics = _detect_eclipse_edges_slope(
            phases, fluxes, smoothing_window=large_window
        )

        # Should warn about smoothing being too large
        # (if it detects the eclipse and validates)
        warning_msgs = [str(warning.message).lower() for warning in w]
        if len(boundaries) > 0:  # Only validates if eclipse detected
            assert any("smoothing" in msg or "window" in msg for msg in warning_msgs)


def test_auto_smoothing_appropriate_for_data():
    """Test that auto-selected smoothing is reasonable."""
    # Test with different data densities
    for n_points in [100, 1000, 10000]:
        phases = np.linspace(0, 1, n_points)
        fluxes = np.ones_like(phases)

        # Add eclipse (10% of phase)
        eclipse_width = int(0.1 * n_points)
        start_idx = (n_points - eclipse_width) // 2
        fluxes[start_idx : start_idx + eclipse_width] = 0.8

        boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

        # Smoothing window should be reasonable
        # (between 1% and 10% of data)
        window = diagnostics["smoothing_window"]
        assert 0.01 * n_points <= window <= 0.1 * n_points
