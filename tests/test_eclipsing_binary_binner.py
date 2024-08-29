"""
Unit tests for the EclipsingBinaryBinner class from the eclipsebin module.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import matplotlib
from eclipsebin import EclipsingBinaryBinner

matplotlib.use("Agg")


@pytest.fixture
def wrapped_light_curve():
    """
    Fixture to set up a wrapped eclipsing binary light curve.
    """
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[45:55] = 0.9  # Simulate primary eclipse
    fluxes[0:3] = 0.95  # Simulate secondary eclipse
    fluxes[97:100] = 0.95  # Wrap secondary eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)
    return phases, fluxes, flux_errors


@pytest.fixture
def unwrapped_light_curve():
    """
    Fixture to set up an unwrapped eclipsing binary light curve.
    """
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[65:75] = 0.9  # Simulate primary eclipse
    fluxes[20:30] = 0.95  # Simulate secondary eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)
    return phases, fluxes, flux_errors


def test_initialization(wrapped_light_curve):
    """
    Test the initialization of EclipsingBinaryBinner with valid data.
    """
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=50, fraction_in_eclipse=0.3
    )
    assert binner.params["nbins"] == 50
    assert binner.params["fraction_in_eclipse"] == 0.3
    assert len(binner.data["phases"]) == 100
    assert len(binner.data["fluxes"]) == 100
    assert len(binner.data["flux_errors"]) == 100
    # Check if the data is sorted
    assert np.all(np.diff(binner.data["phases"]) >= 0)


def test_initialization_invalid_data(unwrapped_light_curve):
    """
    Test that EclipsingBinaryBinner raises ValueError with invalid data.
    """
    phases, fluxes, flux_errors = unwrapped_light_curve

    # Fewer than 10 data points
    phases_invalid = np.linspace(0, 1, 9)
    fluxes_invalid = np.random.normal(1, 0.01, 9)
    flux_errors_invalid = np.random.normal(0.01, 0.001, 9)

    with pytest.raises(ValueError, match="Number of data points must be at least 10."):
        EclipsingBinaryBinner(
            phases_invalid, fluxes_invalid, flux_errors_invalid, nbins=10
        )

    # Fewer than 10 bins
    with pytest.raises(ValueError, match="Number of bins must be at least 10."):
        EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=9)

    # Data points fewer than bins
    with pytest.raises(
        ValueError,
        match="Number of data points must be greater than or equal to the number of bins.",
    ):
        EclipsingBinaryBinner(phases[:50], fluxes[:50], flux_errors[:50], nbins=60)


def test_eclipse_detection(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the eclipse detection capabilities of EclipsingBinaryBinner.
    """
    # Test wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    primary_min = binner.find_minimum_flux()
    assert np.isclose(primary_min, 0.45, atol=0.05)
    primary_eclipse = binner.get_eclipse_boundaries(primary_min)
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]

    # Test unwrapped light curve
    phases_unwrapped, fluxes_unwrapped, flux_errors_unwrapped = (
        unwrapped_light_curve
    )
    binner_unwrapped = EclipsingBinaryBinner(
        phases_unwrapped, fluxes_unwrapped, flux_errors_unwrapped, nbins=50
    )
    primary_min_unwrapped = binner_unwrapped.find_minimum_flux()
    assert np.isclose(primary_min_unwrapped, 0.65, atol=0.05)
    primary_eclipse_unwrapped = binner_unwrapped.get_eclipse_boundaries(
        primary_min_unwrapped
    )
    assert (
        primary_eclipse_unwrapped[0]
        < primary_min_unwrapped
        < primary_eclipse_unwrapped[1]
    )


def test_calculate_eclipse_bins(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the calculate_eclipse_bins method
    """
    # Test the wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    bins_in_primary = int(
        (binner.params["nbins"] * binner.params["fraction_in_eclipse"]) / 2
    )
    bins_in_secondary = int(
        (binner.params["nbins"] * binner.params["fraction_in_eclipse"])
        - bins_in_primary
    )

    primary_bin_right_edges = binner.calculate_eclipse_bins(
        binner.primary_eclipse, bins_in_primary
    )
    secondary_bin_right_edges = binner.calculate_eclipse_bins(
        binner.secondary_eclipse, bins_in_secondary
    )

    # Check if the number of bin edges are as expected
    assert len(primary_bin_right_edges) == bins_in_primary
    assert len(secondary_bin_right_edges) == bins_in_secondary

    # Check if the bin edges are unique
    assert len(np.unique(primary_bin_right_edges)) == bins_in_primary
    assert len(np.unique(secondary_bin_right_edges)) == bins_in_secondary

    # Test the unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner_unwrapped = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=50
    )
    bins_in_primary_unwrapped = int(
        (
            binner_unwrapped.params["nbins"]
            * binner_unwrapped.params["fraction_in_eclipse"]
        )
        / 2
    )
    bins_in_secondary_unwrapped = int(
        (
            binner_unwrapped.params["nbins"]
            * binner_unwrapped.params["fraction_in_eclipse"]
        )
        - bins_in_primary_unwrapped
    )

    primary_bin_right_edges_unwrapped = binner_unwrapped.calculate_eclipse_bins(
        binner_unwrapped.primary_eclipse, bins_in_primary_unwrapped
    )
    secondary_bin_right_edges_unwrapped = binner_unwrapped.calculate_eclipse_bins(
        binner_unwrapped.secondary_eclipse, bins_in_secondary_unwrapped
    )
    # Check if the number of bin edges are as expected
    assert len(primary_bin_right_edges_unwrapped) == bins_in_primary_unwrapped
    assert len(secondary_bin_right_edges_unwrapped) == bins_in_secondary_unwrapped

    # Check if the bin edges are unique
    assert len(np.unique(primary_bin_right_edges_unwrapped)) == bins_in_primary_unwrapped
    assert len(np.unique(secondary_bin_right_edges_unwrapped)) == bins_in_secondary_unwrapped


def test_calculate_out_of_eclipse_bins(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the calculate_out_of_eclipse_bins method
    """
    # Test the wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    bins_in_primary = int(
        (binner.params["nbins"] * binner.params["fraction_in_eclipse"]) / 2
    )
    bins_in_secondary = int(
        (binner.params["nbins"] * binner.params["fraction_in_eclipse"])
        - bins_in_primary
    )
    bins_in_ooe1 = int(
        (binner.params["nbins"] - bins_in_primary - bins_in_secondary) / 2
    )
    bins_in_ooe2 = (
        binner.params["nbins"] - bins_in_primary - bins_in_secondary - bins_in_ooe1
    )

    ooe1_right_edges, ooe2_right_edges = binner.calculate_out_of_eclipse_bins(
        bins_in_primary, bins_in_secondary
    )
    assert len(ooe1_right_edges) == bins_in_ooe1
    assert len(ooe2_right_edges) == bins_in_ooe2

    # Check if the bin edges are unique
    assert len(np.unique(ooe1_right_edges)) == bins_in_ooe1
    assert len(np.unique(ooe2_right_edges)) == bins_in_ooe2

    # Test the unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    bins_in_primary = int(
        (binner.params["nbins"] * binner.params["fraction_in_eclipse"]) / 2
    )
    bins_in_secondary = int(
        (binner.params["nbins"] * binner.params["fraction_in_eclipse"])
        - bins_in_primary
    )
    bins_in_ooe1 = int(
        (binner.params["nbins"] - bins_in_primary - bins_in_secondary) / 2
    )
    bins_in_ooe2 = (
        binner.params["nbins"] - bins_in_primary - bins_in_secondary - bins_in_ooe1
    )

    ooe1_right_edges, ooe2_right_edges = binner.calculate_out_of_eclipse_bins(
        bins_in_primary, bins_in_secondary
    )
    assert len(ooe1_right_edges) == bins_in_ooe1
    assert len(ooe2_right_edges) == bins_in_ooe2

    # Check if the bin edges are unique
    assert len(np.unique(ooe1_right_edges)) == bins_in_ooe1
    assert len(np.unique(ooe2_right_edges)) == bins_in_ooe2


def test_find_bin_edges(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the find_bin_edges method
    """
    # Test the wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    all_bins = binner.find_bin_edges()
    # Check if the bins are sorted
    assert np.all(np.diff(all_bins) >= 0)
    # Check if the number of bins is as expected
    expected_bins_count = binner.params["nbins"]
    assert len(all_bins) == expected_bins_count
    # Check that all bin edges are different
    assert len(np.unique(all_bins)) == len(all_bins)

    # Test the unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    all_bins = binner.find_bin_edges()
    # Check if the bins are sorted
    assert np.all(np.diff(all_bins) >= 0)
    # Check if the number of bins is as expected
    expected_bins_count = binner.params["nbins"]
    assert len(all_bins) == expected_bins_count
    # Check that all bin edges are different
    assert len(np.unique(all_bins)) == len(all_bins)


def test_bin_calculation(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the bin calculation capabilities of EclipsingBinaryBinner.
    """
    # Test wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=False)
    assert len(bin_centers) > 0
    assert len(bin_means) == len(bin_centers)
    assert len(bin_stds) == len(bin_centers)

    # Test unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=False)
    assert len(bin_centers) > 0
    assert len(bin_means) == len(bin_centers)
    assert len(bin_stds) == len(bin_centers)


def test_plot_functions(wrapped_light_curve):
    """
    Test the plotting capabilities of EclipsingBinaryBinner.
    """
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=True)
    binner.plot_binned_light_curve(bin_centers, bin_means, bin_stds, bin_centers)
    binner.plot_unbinned_light_curve()
