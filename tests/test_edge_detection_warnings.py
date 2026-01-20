# tests/test_edge_detection_warnings.py
import numpy as np
import warnings
from eclipsebin.binning import EclipsingBinaryBinner

def test_no_eclipses_warning_is_informative():
    """Test that warning provides actionable guidance when no eclipses found."""
    # Create flat light curve (no eclipses)
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases) + np.random.normal(0, 0.01, len(phases))
    flux_errors = np.ones_like(phases) * 0.01

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should have issued a warning
        assert len(w) > 0
        warning_msg = str(w[0].message)

        # Check that warning mentions:
        # 1. What failed (no eclipses found)
        # 2. Diagnostic info (thresholds, candidates)
        # 3. Actionable suggestion (parameter to adjust)
        assert 'no eclipses' in warning_msg.lower() or 'not find' in warning_msg.lower()
        assert 'edge_' in warning_msg  # Mentions parameter names
        assert 'percentile' in warning_msg.lower() or 'depth' in warning_msg.lower()

def test_missing_eclipse_warning_mentions_eclipse_type():
    """Test that warning specifies which eclipse (primary/secondary) wasn't found."""
    # Create light curve with only one strong eclipse
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)
    # Strong primary only
    fluxes[450:550] = 0.7
    flux_errors = np.ones_like(phases) * 0.01

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # May warn about missing secondary
        if len(w) > 0:
            # Check all warnings for eclipse-related messages
            # (may have smoothing warnings first)
            warning_msgs = [str(warning.message).lower() for warning in w]
            eclipse_warnings = [msg for msg in warning_msgs
                              if 'primary' in msg or 'secondary' in msg]
            # Should have at least one warning mentioning which eclipse is missing
            assert len(eclipse_warnings) > 0
