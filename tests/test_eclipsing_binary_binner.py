"""
Unit tests for the EclipsingBinaryBinner class from the eclipsebin module.
"""

# Test using real light curve data from TESS (Ricker et al., 2015)
# and ASAS-SN (Shappee et al., 2014).
# TESS data: https://archive.stsci.edu/tess/
# ASAS-SN data: https://asas-sn.osu.edu/

# pylint: disable=redefined-outer-name

from pathlib import Path
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
    np.random.seed(1)
    # Increase the number of original points to have enough for random sampling
    phases = np.linspace(0, 0.999, 10000)
    fluxes = np.ones_like(phases)
    # Simulate primary eclipse
    fluxes[4500:5000] = np.linspace(0.95, 0.8, 500)
    fluxes[5000:5500] = np.linspace(0.81, 0.95, 500)
    fluxes[0:300] = np.linspace(0.9, 0.95, 300)  # Simulate secondary eclipse
    fluxes[9700:10000] = np.linspace(0.94, 0.91, 300)  # Wrap secondary eclipse
    flux_errors = np.random.normal(0.01, 0.001, 10000)
    # Select a random, unevenly spaced subset of the data (500 points)
    random_indices = np.random.choice(range(len(phases)), size=5000, replace=False)
    phases = phases[random_indices]
    fluxes = fluxes[random_indices]
    flux_errors = flux_errors[random_indices]
    return phases, fluxes, flux_errors


@pytest.fixture
def unwrapped_light_curve():
    """
    Fixture to set up an unwrapped eclipsing binary light curve.
    """
    # Increase the number of original points to have enough for random sampling
    phases = np.linspace(0, 0.999, 10000)
    fluxes = np.ones_like(phases)
    # Simulate primary eclipse
    fluxes[6500:7000] = np.linspace(0.95, 0.8, 500)
    fluxes[7000:7500] = np.linspace(0.81, 0.95, 500)
    # Simulate secondary eclipse
    fluxes[2000:2500] = np.linspace(0.95, 0.91, 500)
    fluxes[2500:3000] = np.linspace(0.91, 0.95, 500)
    flux_errors = np.random.normal(0.01, 0.001, 10000)

    # Select a random, unevenly spaced subset of the data (500 points)
    random_indices = np.random.choice(range(len(phases)), size=5000, replace=False)
    phases = phases[random_indices]
    fluxes = fluxes[random_indices]
    flux_errors = flux_errors[random_indices]

    return phases, fluxes, flux_errors


@pytest.fixture
def asas_sn_unwrapped_light_curve():
    """
    Fixture to set up a real unwrapped ASAS-SN eclipsing binary light curve.
    """
    data_path = Path(__file__).parent / "data" / "lc_asas_sn_unwrapped.npy"
    phases, fluxes, flux_errors = np.load(data_path)
    return phases, fluxes, flux_errors


@pytest.fixture
def tess_unwrapped_light_curve():
    """
    Fixture to set up a real unwrapped TESS eclipsing binary light curve.
    """
    data_path = Path(__file__).parent / "data" / "lc_tess_unwrapped.npy"
    phases, fluxes, flux_errors = np.load(data_path)
    return phases, fluxes, flux_errors


@pytest.mark.parametrize("fraction_in_eclipse", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("nbins", [50, 100, 200])
def test_unwrapped_light_curves(
    unwrapped_light_curve,
    asas_sn_unwrapped_light_curve,
    tess_unwrapped_light_curve,
    fraction_in_eclipse,
    nbins,
):
    """
    Call tests on the light curves in which neither the primary nor 
    secondary eclipse crosses the 1-0 phase boundary.
    """
    unwrapped_light_curves = [
        unwrapped_light_curve,
        asas_sn_unwrapped_light_curve,
        tess_unwrapped_light_curve,
    ]
    for phases, fluxes, flux_errors in unwrapped_light_curves:
        helper_eclipse_detection(
            phases, fluxes, flux_errors, nbins, fraction_in_eclipse,
            wrapped={"primary": False, "secondary": False}
        )
        helper_initialization(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_find_bin_edges(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_shift_bins(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_find_eclipse_minima(
            phases, fluxes, flux_errors, nbins, fraction_in_eclipse
        )
        helper_calculate_eclipse_bins(
            phases, fluxes, flux_errors, nbins, fraction_in_eclipse
        )
        helper_calculate_out_of_eclipse_bins(
            phases, fluxes, flux_errors, nbins, fraction_in_eclipse
        )
        helper_bin_calculation(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_plot_functions(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)


@pytest.mark.parametrize("fraction_in_eclipse", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("nbins", [50, 100, 200])
def test_secondary_wrapped_light_curves(
    wrapped_light_curve, fraction_in_eclipse, nbins
):
    """
    Call tests on the light curves in which the secondary eclipse crosses the 1-0 phase boundary.
    """
    secondary_wrapped_light_curves = [wrapped_light_curve]
    for phases, fluxes, flux_errors in secondary_wrapped_light_curves:
        helper_eclipse_detection(
            phases,
            fluxes,
            flux_errors,
            nbins,
            fraction_in_eclipse,
            wrapped={"primary": False, "secondary": True},
        )
        helper_initialization(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_find_bin_edges(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_shift_bins(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_find_eclipse_minima(
            phases, fluxes, flux_errors, nbins, fraction_in_eclipse
        )
        helper_calculate_eclipse_bins(
            phases, fluxes, flux_errors, nbins, fraction_in_eclipse
        )
        helper_calculate_out_of_eclipse_bins(
            phases, fluxes, flux_errors, nbins, fraction_in_eclipse
        )
        helper_bin_calculation(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)
        helper_plot_functions(phases, fluxes, flux_errors, nbins, fraction_in_eclipse)


def helper_initialization(phases, fluxes, flux_errors, nbins, fraction_in_eclipse):
    """
    Helper function to test the initialization of EclipsingBinaryBinner.
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    assert binner.params["nbins"] == nbins
    assert binner.params["fraction_in_eclipse"] == fraction_in_eclipse
    assert len(binner.data["phases"]) == len(phases)
    assert len(binner.data["fluxes"]) == len(phases)
    assert len(binner.data["flux_errors"]) == len(phases)
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
        match="Number of data points must be greater than or equal to 5 times the number of bins.",
    ):
        EclipsingBinaryBinner(phases[:50], fluxes[:50], flux_errors[:50], nbins=60)


def helper_find_eclipse_minima(phases, fluxes, flux_errors, nbins, fraction_in_eclipse):
    """
    Test the find_minimum_flux method of EclipsingBinaryBinner.
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    primary_minimum_phase = binner.find_minimum_flux()
    assert 0 <= primary_minimum_phase <= 1.0

    secondary_minimum_phase = binner.find_secondary_minimum()
    assert 0 <= secondary_minimum_phase <= 1.0

    bins = binner.find_bin_edges()
    _ = binner.shift_bin_edges(bins)

    primary_minimum_shifted_phase = binner.find_minimum_flux(use_shifted_phases=True)
    assert primary_minimum_shifted_phase >= primary_minimum_phase
    assert 0 <= primary_minimum_shifted_phase <= 1.0

    secondary_minimum_shifted_phase = binner.find_secondary_minimum(
        use_shifted_phases=True
    )
    assert secondary_minimum_shifted_phase >= secondary_minimum_phase
    assert 0 <= secondary_minimum_shifted_phase <= 1.0


def helper_eclipse_detection(
    phases,
    fluxes,
    flux_errors,
    nbins,
    fraction_in_eclipse,
    wrapped
):
    """
    Test the eclipse detection capabilities of EclipsingBinaryBinner on a wrapped light curve.
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    # Test for shifted phases
    bins = binner.find_bin_edges()
    _ = binner.shift_bin_edges(bins)
    for shifted in [False, True]:
        primary_min = binner.find_minimum_flux(use_shifted_phases=shifted)
        primary_eclipse = binner.get_eclipse_boundaries(
            primary=True, use_shifted_phases=shifted
        )
        assert 0 <= primary_min <= 1
        assert 0 <= primary_eclipse[0] <= 1
        assert 0 <= primary_eclipse[1] <= 1
        if not wrapped["primary"]:
            assert primary_eclipse[0] < primary_min < primary_eclipse[1]

        secondary_min = binner.find_secondary_minimum(use_shifted_phases=shifted)
        secondary_eclipse = binner.get_eclipse_boundaries(
            primary=False, use_shifted_phases=shifted
        )
        assert 0 <= secondary_min <= 1
        assert 0 <= secondary_eclipse[0] <= 1
        assert 0 <= secondary_eclipse[1] <= 1
        if not wrapped["secondary"]:
            assert secondary_eclipse[0] < secondary_min < secondary_eclipse[1]


def helper_calculate_eclipse_bins(
    phases, fluxes, flux_errors, nbins, fraction_in_eclipse
):
    """
    Test the calculate_eclipse_bins method
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )

    bins_in_primary, bins_in_secondary = binner.calculate_eclipse_bins_distribution()

    primary_bin_right_edges = binner.calculate_eclipse_bins(
        binner.primary_eclipse, bins_in_primary
    )
    secondary_bin_right_edges = binner.calculate_eclipse_bins(
        binner.secondary_eclipse, bins_in_secondary
    )
    # Check if the bin edges are unique
    assert len(np.unique(primary_bin_right_edges)) == len(primary_bin_right_edges)
    assert len(np.unique(secondary_bin_right_edges)) == len(secondary_bin_right_edges)
    # Check if the bin edges are within the range [0, 1)
    assert np.all(primary_bin_right_edges <= 1) and np.all(primary_bin_right_edges >= 0)
    assert np.all(secondary_bin_right_edges <= 1) and np.all(
        secondary_bin_right_edges >= 0
    )
    # Check if there are more than one right bin edges
    assert len(primary_bin_right_edges) > 1
    assert len(secondary_bin_right_edges) > 1


def helper_calculate_out_of_eclipse_bins(
    phases, fluxes, flux_errors, nbins, fraction_in_eclipse
):
    """
    Test the calculate_out_of_eclipse_bins method
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
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

    # Check if the bin edges are within the range [0, 1)
    assert np.all(ooe1_right_edges <= 1) and np.all(ooe1_right_edges >= 0)
    assert np.all(ooe2_right_edges <= 1) and np.all(ooe2_right_edges >= 0)


def helper_find_bin_edges(phases, fluxes, flux_errors, nbins, fraction_in_eclipse):
    """
    Test the find_bin_edges method
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    all_bins = binner.find_bin_edges()
    # Check if the bins are sorted
    assert np.all(np.diff(all_bins) >= 0)
    # Check if the number of bins is as expected
    expected_bins_count = binner.params["nbins"]
    assert len(all_bins) == expected_bins_count
    # Check that all bin edges are different
    assert len(np.unique(all_bins)) == len(all_bins)
    # Check if the bin edges are within the range [0, 1)
    assert np.all(all_bins <= 1) and np.all(all_bins >= 0)


def helper_shift_bins(phases, fluxes, flux_errors, nbins, fraction_in_eclipse):
    """
    Test the find_bin_edges method
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    all_bins = binner.find_bin_edges()
    shifted_bins = binner.shift_bin_edges(all_bins)
    # Check that the last bin edge is 1
    assert np.isclose(shifted_bins[-1], 1)
    assert np.isclose(shifted_bins[0], 0)
    assert np.all(shifted_bins <= 1) and np.all(shifted_bins >= 0)
    assert len(shifted_bins) == len(all_bins) + 1


def helper_bin_calculation(phases, fluxes, flux_errors, nbins, fraction_in_eclipse):
    """
    Test the bin calculation capabilities of EclipsingBinaryBinner.
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    bin_centers, bin_means, bin_errors, bin_numbers, _ = binner.calculate_bins()
    assert len(bin_centers) > 0
    assert len(bin_means) == len(bin_centers)
    assert len(bin_errors) == len(bin_centers)
    assert np.all(bin_errors > 0)
    assert not np.any(np.isnan(bin_centers))
    assert not np.any(np.isnan(bin_means))
    assert not np.any(np.isnan(bin_errors))
    assert np.all(bin_centers <= 1) and np.all(bin_centers >= 0)
    assert len(np.unique(bin_centers)) == len(bin_centers)
    assert np.all(np.bincount(bin_numbers)[1:] > 0)


def helper_plot_functions(phases, fluxes, flux_errors, nbins, fraction_in_eclipse):
    """
    Test the plotting capabilities of EclipsingBinaryBinner.
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    bin_centers, bin_means, bin_errors = binner.bin_light_curve(plot=True)
    binner.plot_binned_light_curve(bin_centers, bin_means, bin_errors)
    binner.plot_unbinned_light_curve()
    matplotlib.pyplot.close()
