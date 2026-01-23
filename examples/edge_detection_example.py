"""
Example demonstrating edge detection method for eclipse boundary detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from eclipsebin.binning import EclipsingBinaryBinner


def create_ellipsoidal_light_curve():
    """Create synthetic light curve with ellipsoidal variations."""
    phases = np.linspace(0, 1, 2000)

    # Ellipsoidal variation (twice per orbit)
    fluxes = 1.0 + 0.08 * np.cos(4 * np.pi * phases)

    # Primary eclipse at phase 0.0
    primary_mask = (phases < 0.12) | (phases > 0.88)
    fluxes[primary_mask] -= 0.25 * np.exp(-((phases[primary_mask] % 1.0) ** 2) / 0.003)

    # Secondary eclipse at phase 0.5
    secondary_mask = (phases > 0.38) & (phases < 0.62)
    fluxes[secondary_mask] -= 0.12 * np.exp(
        -((phases[secondary_mask] - 0.5) ** 2) / 0.003
    )

    # Add noise
    fluxes += np.random.normal(0, 0.01, len(phases))
    flux_errors = np.ones_like(phases) * 0.01

    return phases, fluxes, flux_errors


def example_basic_usage():
    """Basic usage of edge detection."""
    print("=" * 60)
    print("Example 1: Basic Edge Detection")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=200, boundary_method="edge_detection"
    )

    print(f"Primary eclipse: {binner.primary_eclipse}")
    print(f"Secondary eclipse: {binner.secondary_eclipse}")
    print(f"Primary depth: {1 - binner.find_minimum_flux():.3f}")
    print(f"Secondary depth: {1 - binner.find_secondary_minimum():.3f}")

    # Plot
    binner.plot_unbinned_light_curve()


def example_comparison():
    """Compare edge detection vs flux return."""
    print("\n" + "=" * 60)
    print("Example 2: Edge Detection vs Flux Return")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    # Edge detection
    binner_edge = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=200, boundary_method="edge_detection"
    )

    # Flux return
    binner_flux = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=200, boundary_method="flux_return"
    )

    print("\nEdge Detection:")
    print(f"  Primary: {binner_edge.primary_eclipse}")
    print(f"  Secondary: {binner_edge.secondary_eclipse}")

    print("\nFlux Return:")
    print(f"  Primary: {binner_flux.primary_eclipse}")
    print(f"  Secondary: {binner_flux.secondary_eclipse}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Edge detection
    ax1.plot(phases, fluxes, "k.", alpha=0.3, markersize=1)
    ax1.axvline(
        binner_edge.primary_eclipse[0], color="r", linestyle="--", label="Primary"
    )
    ax1.axvline(binner_edge.primary_eclipse[1], color="r", linestyle="--")
    ax1.axvline(
        binner_edge.secondary_eclipse[0], color="b", linestyle="--", label="Secondary"
    )
    ax1.axvline(binner_edge.secondary_eclipse[1], color="b", linestyle="--")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title("Edge Detection Method")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Flux return
    ax2.plot(phases, fluxes, "k.", alpha=0.3, markersize=1)
    ax2.axvline(
        binner_flux.primary_eclipse[0], color="r", linestyle="--", label="Primary"
    )
    ax2.axvline(binner_flux.primary_eclipse[1], color="r", linestyle="--")
    ax2.axvline(
        binner_flux.secondary_eclipse[0], color="b", linestyle="--", label="Secondary"
    )
    ax2.axvline(binner_flux.secondary_eclipse[1], color="b", linestyle="--")
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Normalized Flux")
    ax2.set_title("Flux Return Method")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def example_parameter_tuning():
    """Demonstrate parameter tuning."""
    print("\n" + "=" * 60)
    print("Example 3: Parameter Tuning")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    # Conservative parameters
    binner_conservative = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=200,
        boundary_method="edge_detection",
        edge_slope_threshold_percentile=95,
        edge_min_eclipse_depth=0.05,
    )

    # Sensitive parameters
    binner_sensitive = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=200,
        boundary_method="edge_detection",
        edge_slope_threshold_percentile=85,
        edge_min_eclipse_depth=0.01,
    )

    print("\nConservative (strict thresholds):")
    print(f"  Primary: {binner_conservative.primary_eclipse}")
    print(f"  Secondary: {binner_conservative.secondary_eclipse}")
    print(
        f"  Detected: {binner_conservative._edge_diagnostics['detected_count']} eclipses"
    )

    print("\nSensitive (permissive thresholds):")
    print(f"  Primary: {binner_sensitive.primary_eclipse}")
    print(f"  Secondary: {binner_sensitive.secondary_eclipse}")
    print(
        f"  Detected: {binner_sensitive._edge_diagnostics['detected_count']} eclipses"
    )


def example_diagnostics():
    """Show diagnostic information."""
    print("\n" + "=" * 60)
    print("Example 4: Diagnostics")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=200, boundary_method="edge_detection"
    )

    diag = binner._edge_diagnostics

    print(f"\nDiagnostic Information:")
    print(f"  Slope threshold: {diag['threshold']:.3e}")
    print(f"  Return threshold: {diag['return_threshold']:.3e}")
    print(f"  Smoothing window: {diag['smoothing_window']} points")
    print(f"  Detected eclipses: {diag['detected_count']}")
    print(f"  Ingress candidates: {len(diag['ingress_candidates'])}")
    print(f"  Egress candidates: {len(diag['egress_candidates'])}")

    # Plot slopes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Original and smoothed flux
    ax1.plot(phases, fluxes, "k.", alpha=0.3, markersize=1, label="Raw")
    ax1.plot(phases, diag["smoothed_fluxes"], "r-", linewidth=2, label="Smoothed")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title("Light Curve and Smoothing")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Slopes
    ax2.plot(phases, diag["slopes"], "b-", linewidth=1, label="Slope")
    ax2.axhline(diag["threshold"], color="r", linestyle="--", label="Ingress threshold")
    ax2.axhline(-diag["threshold"], color="r", linestyle="--", label="Egress threshold")
    ax2.axhline(
        diag["return_threshold"],
        color="orange",
        linestyle=":",
        label="Return threshold",
    )
    ax2.axhline(-diag["return_threshold"], color="orange", linestyle=":")
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Slope (dFlux/dPhase)")
    ax2.set_title("Slopes and Thresholds")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_comparison()
    example_parameter_tuning()
    example_diagnostics()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
