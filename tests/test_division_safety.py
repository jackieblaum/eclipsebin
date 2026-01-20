import numpy as np
from eclipsebin.binning import _detect_eclipse_edges_slope

def test_handles_zero_baseline_gracefully():
    """Test that algorithm handles near-zero baseline without crashing."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases) * 0.05  # Very faint baseline

    # Add relative eclipse
    fluxes[450:550] = 0.03

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should not crash with ZeroDivisionError
    assert isinstance(boundaries, list)

def test_handles_zero_phase_spacing():
    """Test handling of duplicate phase values."""
    phases = np.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fluxes = np.ones_like(phases)
    fluxes[6:8] = 0.7  # Eclipse

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should handle duplicate phases without division by zero
    assert isinstance(boundaries, list)

def test_handles_all_same_flux():
    """Test handling of constant flux (no variation)."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)  # Constant

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should return empty boundaries (no eclipses)
    assert boundaries == []
