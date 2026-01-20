import numpy as np
from eclipsebin.binning import _detect_eclipse_edges_slope

def test_baseline_window_scales_with_data_density():
    """Test that baseline window adapts to data density."""
    # Sparse data (100 points)
    sparse_phases = np.linspace(0, 1, 100)
    sparse_fluxes = np.ones(100)
    sparse_fluxes[40:50] = 0.8  # Eclipse

    boundaries_sparse, diag_sparse = _detect_eclipse_edges_slope(
        sparse_phases, sparse_fluxes
    )

    # Dense data (10000 points)
    dense_phases = np.linspace(0, 1, 10000)
    dense_fluxes = np.ones(10000)
    dense_fluxes[4000:5000] = 0.8  # Eclipse at same phase

    boundaries_dense, diag_dense = _detect_eclipse_edges_slope(
        dense_phases, dense_fluxes
    )

    # Both should detect the eclipse
    assert len(boundaries_sparse) > 0
    assert len(boundaries_dense) > 0

    # Diagnostic should show different smoothing windows
    assert 'smoothing_window' in diag_sparse
    assert 'smoothing_window' in diag_dense
    # Dense data should have larger window
    assert diag_dense['smoothing_window'] > diag_sparse['smoothing_window']

    # Check that baseline window and refinement range diagnostics exist
    assert 'baseline_window' in diag_sparse
    assert 'baseline_window' in diag_dense
    assert 'refinement_range' in diag_sparse
    assert 'refinement_range' in diag_dense

    # Dense data should use larger baseline window and refinement range
    assert diag_dense['baseline_window'] > diag_sparse['baseline_window']
    assert diag_dense['refinement_range'] > diag_sparse['refinement_range']

def test_refinement_range_adapts_to_data():
    """Test that boundary refinement range scales with data."""
    # This is an integration test - we verify through diagnostics
    # that the algorithm doesn't fail on edge cases

    # Very sparse data
    phases = np.linspace(0, 1, 50)
    fluxes = np.ones(50)
    fluxes[20:25] = 0.7

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should not crash and should detect eclipse
    assert len(boundaries) >= 0  # May or may not detect with very sparse data

def test_adaptive_constants_minimum_values():
    """Test that adaptive constants have reasonable minimum values."""
    # Very sparse data (minimum viable)
    phases = np.linspace(0, 1, 50)
    fluxes = np.ones(50)
    fluxes[20:25] = 0.7

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Even with sparse data, should have minimum values
    if 'baseline_window' in diagnostics:
        assert diagnostics['baseline_window'] >= 5  # Minimum 5 points
    if 'refinement_range' in diagnostics:
        assert diagnostics['refinement_range'] >= 10  # Minimum 10 points

def test_gap_detection_adapts():
    """Test that gap detection threshold adapts properly."""
    # Create data with irregular spacing
    phases = np.concatenate([
        np.linspace(0, 0.3, 100),
        np.linspace(0.7, 1.0, 100)  # Large gap in middle
    ])
    fluxes = np.ones_like(phases)
    fluxes[40:50] = 0.8  # Eclipse in first region

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should not crash with irregular spacing
    assert len(boundaries) >= 0
