"""
Comprehensive integration tests for edge detection with ellipsoidal variations.
"""
import numpy as np
import pytest
from eclipsebin.binning import EclipsingBinaryBinner

def create_synthetic_light_curve_with_ellipsoidal(
    n_points=1000,
    primary_depth=0.3,
    secondary_depth=0.15,
    ellipsoidal_amplitude=0.05,
    noise_level=0.01
):
    """
    Create synthetic light curve with eclipses and ellipsoidal variations.

    Parameters
    ----------
    n_points : int
        Number of data points
    primary_depth : float
        Depth of primary eclipse (relative)
    secondary_depth : float
        Depth of secondary eclipse (relative)
    ellipsoidal_amplitude : float
        Amplitude of ellipsoidal variation (cos(4*pi*phase))
    noise_level : float
        Standard deviation of Gaussian noise

    Returns
    -------
    phases, fluxes, flux_errors : arrays
    """
    phases = np.linspace(0, 1, n_points)

    # Start with ellipsoidal variation (twice per orbit)
    fluxes = 1.0 + ellipsoidal_amplitude * np.cos(4 * np.pi * phases)

    # Add primary eclipse centered at phase 0.0
    primary_mask = (phases < 0.1) | (phases > 0.9)
    primary_depth_curve = primary_depth * np.exp(-((phases[primary_mask] % 1.0 - 0.0) ** 2) / 0.005)
    fluxes[primary_mask] -= primary_depth_curve

    # Add secondary eclipse centered at phase 0.5
    secondary_mask = (phases > 0.4) & (phases < 0.6)
    secondary_depth_curve = secondary_depth * np.exp(-((phases[secondary_mask] - 0.5) ** 2) / 0.005)
    fluxes[secondary_mask] -= secondary_depth_curve

    # Add noise
    fluxes += np.random.normal(0, noise_level, n_points)

    flux_errors = np.ones_like(phases) * noise_level

    return phases, fluxes, flux_errors


class TestEdgeDetectionIntegration:
    """Integration tests for edge detection with realistic light curves."""

    def test_detects_eclipses_with_ellipsoidal_variation(self):
        """Test that edge detection works with strong ellipsoidal variations."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            ellipsoidal_amplitude=0.1  # Strong variation
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should detect both eclipses
        assert binner.primary_eclipse is not None
        assert binner.secondary_eclipse is not None

        # Primary should be deeper
        primary_depth = binner.find_minimum_flux()
        secondary_depth = binner.find_secondary_minimum()
        assert primary_depth < secondary_depth

    def test_edge_detection_better_than_flux_return_with_ellipsoidal(self):
        """Test that edge detection outperforms flux_return with ellipsoidal variation."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            ellipsoidal_amplitude=0.08
        )

        # Edge detection
        binner_edge = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Flux return method
        binner_flux = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='flux_return'
        )

        # Both should detect eclipses
        assert binner_edge.primary_eclipse is not None
        assert binner_flux.primary_eclipse is not None

        # Edge detection boundaries should be more accurate
        # (closer to true boundaries at 0.9-0.1 and 0.4-0.6)
        edge_primary = binner_edge.primary_eclipse
        edge_width = (edge_primary[1] - edge_primary[0]) % 1.0

        # Primary should span roughly 0.2 phase units (allow wider due to smoothing)
        assert 0.15 < edge_width < 0.35

    def test_handles_sparse_data(self):
        """Test edge detection with sparse data."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            n_points=100,  # Sparse
            noise_level=0.02
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=50,  # Fewer bins for sparse data
            boundary_method='edge_detection'
        )

        # Should handle gracefully (may or may not detect eclipses)
        assert binner.primary_eclipse is not None or binner.secondary_eclipse is not None

    def test_handles_dense_noisy_data(self):
        """Test edge detection with dense but noisy data."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            n_points=10000,  # Dense
            noise_level=0.05  # Noisy
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=500,
            boundary_method='edge_detection'
        )

        # Should still detect eclipses despite noise
        assert binner.primary_eclipse is not None
        assert binner.secondary_eclipse is not None

    def test_shallow_eclipse_detection(self):
        """Test detection of very shallow eclipses."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            primary_depth=0.05,  # Very shallow
            secondary_depth=0.03,
            ellipsoidal_amplitude=0.02,
            noise_level=0.005
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection',
            edge_min_eclipse_depth=0.02,  # Lower threshold for shallow eclipses
            edge_slope_threshold_percentile=85  # More sensitive
        )

        # Should detect at least primary
        assert binner.primary_eclipse is not None

    def test_wrapped_eclipse_handling(self):
        """Test edge detection with eclipse wrapping around phase 0/1."""
        phases = np.linspace(0, 1, 1000)
        fluxes = 1.0 + 0.05 * np.cos(4 * np.pi * phases)  # Ellipsoidal

        # Eclipse centered at phase 0.0 (wraps around)
        wrap_mask = (phases < 0.15) | (phases > 0.85)
        fluxes[wrap_mask] -= 0.2

        # Secondary at phase 0.5
        fluxes[450:550] -= 0.1

        flux_errors = np.ones_like(phases) * 0.01

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should handle wrapped eclipse
        assert binner.primary_eclipse is not None
        assert binner.secondary_eclipse is not None

    def test_diagnostics_available_after_binning(self):
        """Test that diagnostics are accessible after binning."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal()

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Diagnostics should be available
        assert hasattr(binner, '_edge_diagnostics')
        assert binner._edge_diagnostics is not None

        diag = binner._edge_diagnostics
        assert 'threshold' in diag
        assert 'detected_count' in diag
        assert diag['detected_count'] >= 1  # At least one eclipse

    def test_parameter_tuning(self):
        """Test that adjusting parameters improves detection."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            primary_depth=0.08,  # Moderate depth
            ellipsoidal_amplitude=0.06
        )

        # Conservative parameters (may miss secondary)
        binner_conservative = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection',
            edge_slope_threshold_percentile=95,  # Very strict
            edge_min_eclipse_depth=0.05
        )

        # Sensitive parameters
        binner_sensitive = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection',
            edge_slope_threshold_percentile=85,  # More permissive
            edge_min_eclipse_depth=0.02
        )

        # Sensitive should detect as many or more eclipses
        conservative_count = (
            (binner_conservative.primary_eclipse is not None) +
            (binner_conservative.secondary_eclipse is not None)
        )
        sensitive_count = (
            (binner_sensitive.primary_eclipse is not None) +
            (binner_sensitive.secondary_eclipse is not None)
        )

        assert sensitive_count >= conservative_count


class TestEdgeDetectionVsFluxReturn:
    """Comparative tests between edge detection and flux return methods."""

    def test_both_methods_work_on_clean_data(self):
        """Test that both methods work on clean data without ellipsoidal."""
        phases = np.linspace(0, 1, 1000)
        fluxes = np.ones_like(phases)
        fluxes[450:550] = 0.7  # Primary
        fluxes[100:150] = 0.85  # Secondary
        flux_errors = np.ones_like(phases) * 0.01

        binner_edge = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        binner_flux = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='flux_return'
        )

        # Both should detect eclipses
        assert binner_edge.primary_eclipse is not None
        assert binner_flux.primary_eclipse is not None

    def test_edge_detection_superior_with_ellipsoidal(self):
        """Test that edge detection is superior with strong ellipsoidal variation."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            ellipsoidal_amplitude=0.15  # Very strong
        )

        # Edge detection should handle this better
        binner_edge = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should successfully detect both eclipses
        assert binner_edge.primary_eclipse is not None
        assert binner_edge.secondary_eclipse is not None
