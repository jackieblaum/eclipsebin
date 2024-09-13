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
    phases = np.linspace(0, 0.999, 100)
    fluxes = np.ones_like(phases)
    # Simulate primary eclipse
    fluxes[45:55] = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.75, 0.8, 0.85, 0.9]
    fluxes[0:3] = [0.9, 0.93, 0.95]  # Simulate secondary eclipse
    fluxes[97:100] = [0.94, 0.93, 0.91]  # Wrap secondary eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)
    # Select a random, unevenly spaced subset of the data
    random_indices = np.random.choice(range(len(phases)), size=80, replace=False)
    phases = phases[random_indices]
    fluxes = fluxes[random_indices]
    flux_errors = flux_errors[random_indices]
    return phases, fluxes, flux_errors


@pytest.fixture
def unwrapped_light_curve():
    """
    Fixture to set up an unwrapped eclipsing binary light curve.
    """
    phases = np.linspace(0, 0.999, 100)
    fluxes = np.ones_like(phases)
    # Simulate primary eclipse
    fluxes[65:75] = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.75, 0.8, 0.85, 0.9]
    # Simulate secondary eclipse
    fluxes[20:30] = [0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.92, 0.93, 0.94, 0.95]
    flux_errors = np.random.normal(0.01, 0.001, 100)
    # Select a random, unevenly spaced subset of the data
    random_indices = np.random.choice(range(len(phases)), size=80, replace=False)
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


def test_initialization(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the initialization of EclipsingBinaryBinner with valid data.
    """
    for phases, fluxes, flux_errors in [wrapped_light_curve, unwrapped_light_curve]:
        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors, nbins=50, fraction_in_eclipse=0.3
        )
        assert binner.params["nbins"] == 50
        assert binner.params["fraction_in_eclipse"] == 0.3
        assert len(binner.data["phases"]) == 80
        assert len(binner.data["fluxes"]) == 80
        assert len(binner.data["flux_errors"]) == 80
        # Check if the data is sorted
        assert np.all(np.diff(binner.data["phases"]) >= 0)


def test_initialization_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test initialization of EclipsingBinaryBinner with real ASAS-SN and TESS light curve data.
    """
    for phases, fluxes, flux_errors in [
        asas_sn_unwrapped_light_curve,
        tess_unwrapped_light_curve,
    ]:
        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors, nbins=200, fraction_in_eclipse=0.5
        )
        assert binner.params["nbins"] == 200
        assert binner.params["fraction_in_eclipse"] == 0.5
        assert len(binner.data["phases"]) == len(phases)
        assert len(binner.data["fluxes"]) == len(phases)
        assert len(binner.data["flux_errors"]) == len(phases)
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


def test_find_eclipse_minima(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the find_minimum_flux method of EclipsingBinaryBinner.
    """
    for phases, fluxes, flux_errors in [wrapped_light_curve, unwrapped_light_curve]:
        binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
        primary_minimum_phase = binner.find_minimum_flux()
        assert 0 <= primary_minimum_phase <= 1.0

        secondary_minimum_phase = binner.find_secondary_minimum()
        assert 0 <= secondary_minimum_phase <= 1.0

        bins = binner.find_bin_edges()
        _ = binner.shift_bin_edges(bins)

        primary_minimum_shifted_phase = binner.find_minimum_flux(
            use_shifted_phases=True
        )
        assert primary_minimum_shifted_phase >= primary_minimum_phase
        assert 0 <= primary_minimum_shifted_phase <= 1.0

        secondary_minimum_shifted_phase = binner.find_secondary_minimum(
            use_shifted_phases=True
        )
        assert secondary_minimum_shifted_phase >= secondary_minimum_phase
        assert 0 <= secondary_minimum_shifted_phase <= 1.0


def test_find_eclipse_minima_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test the find_minimum_flux method of EclipsingBinaryBinner
    with real ASAS-SN and TESS light curve data.
    """
    for phases, fluxes, flux_errors in [
        asas_sn_unwrapped_light_curve,
        tess_unwrapped_light_curve,
    ]:
        binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=200)
        primary_minimum_phase = binner.find_minimum_flux()
        assert 0 <= primary_minimum_phase <= 1.0

        secondary_minimum_phase = binner.find_secondary_minimum()
        assert 0 <= secondary_minimum_phase <= 1.0

        bins = binner.find_bin_edges()
        _ = binner.shift_bin_edges(bins)

        primary_minimum_shifted_phase = binner.find_minimum_flux(
            use_shifted_phases=True
        )
        assert primary_minimum_shifted_phase >= primary_minimum_phase
        assert 0 <= primary_minimum_shifted_phase <= 1.0

        secondary_minimum_shifted_phase = binner.find_secondary_minimum(
            use_shifted_phases=True
        )
        assert secondary_minimum_shifted_phase >= secondary_minimum_phase
        assert 0 <= secondary_minimum_shifted_phase <= 1.0


def test_eclipse_detection(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the eclipse detection capabilities of EclipsingBinaryBinner.
    """
    # Test wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)

    primary_min = binner.find_minimum_flux()
    primary_eclipse = binner.get_eclipse_boundaries(primary=True)
    assert np.isclose(primary_min, 0.5, atol=0.05)
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]
    assert np.isclose(primary_eclipse[0], 0.44, atol=0.01)
    assert np.isclose(primary_eclipse[1], 0.56, atol=0.01)

    secondary_min = binner.find_secondary_minimum()
    secondary_eclipse = binner.get_eclipse_boundaries(primary=False)
    assert np.isclose(secondary_eclipse[0], 0.96, atol=0.01)
    assert np.isclose(secondary_eclipse[1], 0.04, atol=0.01)

    # Test for shifted phases
    bins = binner.find_bin_edges()
    _ = binner.shift_bin_edges(bins)
    primary_min = binner.find_minimum_flux(use_shifted_phases=True)
    primary_eclipse = binner.get_eclipse_boundaries(
        primary=True, use_shifted_phases=True
    )
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]

    secondary_min = binner.find_secondary_minimum(use_shifted_phases=True)
    secondary_eclipse = binner.get_eclipse_boundaries(
        primary=False, use_shifted_phases=True
    )
    assert 0 <= secondary_eclipse[0] <= 1 and 0 <= secondary_eclipse[1] <= 1

    # Test unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    primary_min = binner.find_minimum_flux()
    assert np.isclose(primary_min, 0.7, atol=0.01)
    primary_eclipse = binner.get_eclipse_boundaries(primary=True)
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]

    secondary_min = binner.find_secondary_minimum()
    secondary_eclipse = binner.get_eclipse_boundaries(primary=False)
    assert secondary_eclipse[0] < secondary_min < secondary_eclipse[1]

    # Test for shifted phases
    bins = binner.find_bin_edges()
    _ = binner.shift_bin_edges(bins)
    primary_min = binner.find_minimum_flux(use_shifted_phases=True)
    primary_eclipse = binner.get_eclipse_boundaries(
        primary=True, use_shifted_phases=True
    )
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]

    secondary_min = binner.find_secondary_minimum(use_shifted_phases=True)
    secondary_eclipse = binner.get_eclipse_boundaries(
        primary=False, use_shifted_phases=True
    )
    assert secondary_eclipse[0] < secondary_min < secondary_eclipse[1]


def test_eclipse_detection_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test EclipsingBinaryBinner's eclipse detection using real ASAS-SN and TESS data.
    This test ensures primary and secondary eclipses are correctly identified and 
    their boundaries are accurately determined.
    """
    # Test with ASAS-SN data
    phases, fluxes, flux_errors = asas_sn_unwrapped_light_curve
    binner_asas_sn = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=200)
    primary_min_asas_sn = binner_asas_sn.find_minimum_flux()
    primary_eclipse_asas_sn = binner_asas_sn.get_eclipse_boundaries(primary=True)
    assert primary_eclipse_asas_sn[0] < primary_min_asas_sn < primary_eclipse_asas_sn[1]

    secondary_min_asas_sn = binner_asas_sn.find_secondary_minimum()
    secondary_eclipse_asas_sn = binner_asas_sn.get_eclipse_boundaries(primary=False)
    assert (
        secondary_eclipse_asas_sn[0]
        < secondary_min_asas_sn
        < secondary_eclipse_asas_sn[1]
    )

    # Test with TESS data
    phases, fluxes, flux_errors = tess_unwrapped_light_curve
    binner_tess = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    primary_min_tess = binner_tess.find_minimum_flux()
    primary_eclipse_tess = binner_tess.get_eclipse_boundaries(primary=True)
    assert primary_eclipse_tess[0] < primary_min_tess < primary_eclipse_tess[1]

    secondary_min_tess = binner_tess.find_secondary_minimum()
    secondary_eclipse_tess = binner_tess.get_eclipse_boundaries(primary=False)
    assert secondary_eclipse_tess[0] < secondary_min_tess < secondary_eclipse_tess[1]


def test_calculate_eclipse_bins(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the calculate_eclipse_bins method
    """
    # Test the wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)

    primary_bin_right_edges = binner.calculate_eclipse_bins(
        binner.primary_eclipse
    )
    secondary_bin_right_edges = binner.calculate_eclipse_bins(
        binner.secondary_eclipse
    )

    # Check if the bin edges are unique
    assert len(np.unique(primary_bin_right_edges)) == len(primary_bin_right_edges)
    assert len(np.unique(secondary_bin_right_edges)) == len(secondary_bin_right_edges)

    # Check if the bin edges are within the range [0, 1)
    assert np.all(primary_bin_right_edges <= 1) and np.all(primary_bin_right_edges >= 0)
    assert np.all(secondary_bin_right_edges <= 1) and np.all(
        secondary_bin_right_edges >= 0
    )

    # Test the unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner_unwrapped = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)

    primary_bin_right_edges_unwrapped = binner_unwrapped.calculate_eclipse_bins(
        binner_unwrapped.primary_eclipse
    )
    secondary_bin_right_edges_unwrapped = binner_unwrapped.calculate_eclipse_bins(
        binner_unwrapped.secondary_eclipse
    )
    # Check if the bin edges are unique
    assert (
        len(np.unique(primary_bin_right_edges_unwrapped)) == len(primary_bin_right_edges_unwrapped)
    )
    assert (
        len(np.unique(secondary_bin_right_edges_unwrapped))
        == len(secondary_bin_right_edges_unwrapped)
    )

    # Check if the bin edges are within the range [0, 1)
    assert np.all(primary_bin_right_edges_unwrapped <= 1) and np.all(
        primary_bin_right_edges_unwrapped >= 0
    )
    assert np.all(secondary_bin_right_edges_unwrapped <= 1) and np.all(
        secondary_bin_right_edges_unwrapped >= 0
    )


def test_calculate_eclipse_bins_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test the calculate_eclipse_bins method with real ASAS-SN and TESS data.
    """
    # Test with ASAS-SN data
    phases, fluxes, flux_errors = asas_sn_unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=200)

    # Combine bin calculations into a single function call
    primary_bin_right_edges, secondary_bin_right_edges = (
        binner.calculate_eclipse_bins(binner.primary_eclipse),
        binner.calculate_eclipse_bins(binner.secondary_eclipse),
    )

    # Check if the bin edges are unique
    assert len(np.unique(primary_bin_right_edges)) == len(primary_bin_right_edges)
    assert len(np.unique(secondary_bin_right_edges)) == len(secondary_bin_right_edges)

    # Check if there are more than one right bin edges
    assert len(primary_bin_right_edges) > 1
    assert len(secondary_bin_right_edges) > 1

    # Check if the bin edges are within the range [0, 1)
    assert np.all(primary_bin_right_edges <= 1) and np.all(primary_bin_right_edges >= 0)
    assert np.all(secondary_bin_right_edges <= 1) and np.all(secondary_bin_right_edges >= 0)

    # Test with TESS data
    phases, fluxes, flux_errors = tess_unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)

    # Combine bin calculations into a single function call
    primary_bin_right_edges, secondary_bin_right_edges = (
        binner.calculate_eclipse_bins(binner.primary_eclipse),
        binner.calculate_eclipse_bins(binner.secondary_eclipse),
    )

    # Check if the bin edges are unique
    assert len(np.unique(primary_bin_right_edges)) == len(primary_bin_right_edges)
    assert len(np.unique(secondary_bin_right_edges)) == len(secondary_bin_right_edges)

    # Check if the bin edges are within the range [0, 1)
    assert np.all(primary_bin_right_edges <= 1) and np.all(primary_bin_right_edges >= 0)
    assert np.all(secondary_bin_right_edges <= 1) and np.all(secondary_bin_right_edges >= 0)


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

    # Check if the bin edges are within the range [0, 1)
    assert np.all(ooe1_right_edges <= 1) and np.all(ooe1_right_edges >= 0)
    assert np.all(ooe2_right_edges <= 1) and np.all(ooe2_right_edges >= 0)

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

    # Check if the bin edges are within the range [0, 1)
    assert np.all(ooe1_right_edges <= 1) and np.all(ooe1_right_edges >= 0)
    assert np.all(ooe2_right_edges <= 1) and np.all(ooe2_right_edges >= 0)


def test_calculate_out_of_eclipse_bins_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test the calculate_out_of_eclipse_bins method with real data from ASAS-SN and TESS.
    This test verifies that the out-of-eclipse bins are correctly calculated and their 
    edges are accurately determined for both ASAS-SN and TESS data.
    """
    # Test with ASAS-SN data
    phases, fluxes, flux_errors = asas_sn_unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=200)
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

    # Test with TESS data
    phases, fluxes, flux_errors = tess_unwrapped_light_curve
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

    # Check if the bin edges are within the range [0, 1)
    assert np.all(ooe1_right_edges <= 1) and np.all(ooe1_right_edges >= 0)
    assert np.all(ooe2_right_edges <= 1) and np.all(ooe2_right_edges >= 0)


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
    # Check if the bin edges are within the range [0, 1)
    assert np.all(all_bins <= 1) and np.all(all_bins >= 0)

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
    # Check if the bin edges are within the range [0, 1)
    assert np.all(all_bins <= 1) and np.all(all_bins >= 0)


def test_find_bin_edges_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test the find_bin_edges method with real data
    """
    # Test the ASAS-SN unwrapped light curve
    phases_asas, fluxes_asas, flux_errors_asas = asas_sn_unwrapped_light_curve
    binner_asas = EclipsingBinaryBinner(
        phases_asas, fluxes_asas, flux_errors_asas, nbins=200
    )
    all_bins_asas = binner_asas.find_bin_edges()
    # Check if the bins are sorted
    assert np.all(np.diff(all_bins_asas) >= 0)
    # Check if the number of bins is as expected
    expected_bins_count_asas = binner_asas.params["nbins"]
    assert len(all_bins_asas) == expected_bins_count_asas
    # Check that all bin edges are different
    assert len(np.unique(all_bins_asas)) == len(all_bins_asas)
    # Check if the bin edges are within the range [0, 1)
    assert np.all(all_bins_asas <= 1) and np.all(all_bins_asas >= 0)

    # Test the TESS unwrapped light curve
    phases_tess, fluxes_tess, flux_errors_tess = tess_unwrapped_light_curve
    binner_tess = EclipsingBinaryBinner(
        phases_tess, fluxes_tess, flux_errors_tess, nbins=50
    )
    all_bins_tess = binner_tess.find_bin_edges()
    # Check if the bins are sorted
    assert np.all(np.diff(all_bins_tess) >= 0)
    # Check if the number of bins is as expected
    expected_bins_count_tess = binner_tess.params["nbins"]
    assert len(all_bins_tess) == expected_bins_count_tess
    # Check that all bin edges are different
    assert len(np.unique(all_bins_tess)) == len(all_bins_tess)
    # Check if the bin edges are within the range [0, 1)
    assert np.all(all_bins_tess <= 1) and np.all(all_bins_tess >= 0)


def test_shift_bins(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the find_bin_edges method
    """
    # Test the wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    all_bins = binner.find_bin_edges()
    shifted_bins = binner.shift_bin_edges(all_bins)
    # Check that the last bin edge is 1
    assert np.isclose(shifted_bins[-1], 1)
    assert np.all(shifted_bins <= 1) and np.all(shifted_bins >= 0)

    # Test the unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    all_bins = binner.find_bin_edges()
    shifted_bins = binner.shift_bin_edges(all_bins)
    # Check that the last bin edge is 1
    assert np.isclose(shifted_bins[-1], 1)
    assert np.all(shifted_bins <= 1) and np.all(shifted_bins >= 0)


def test_shift_bins_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test the shift_bin_edges method with real data
    """
    # Test the ASAS-SN unwrapped light curve
    phases_asas, fluxes_asas, flux_errors_asas = asas_sn_unwrapped_light_curve
    binner_asas = EclipsingBinaryBinner(
        phases_asas, fluxes_asas, flux_errors_asas, nbins=200
    )
    all_bins_asas = binner_asas.find_bin_edges()
    shifted_bins_asas = binner_asas.shift_bin_edges(all_bins_asas)
    # Check that the last bin edge is 1
    assert np.isclose(shifted_bins_asas[-1], 1)
    assert np.all(shifted_bins_asas <= 1) and np.all(shifted_bins_asas >= 0)

    # Test the TESS unwrapped light curve
    phases_tess, fluxes_tess, flux_errors_tess = tess_unwrapped_light_curve
    binner_tess = EclipsingBinaryBinner(
        phases_tess, fluxes_tess, flux_errors_tess, nbins=50
    )
    all_bins_tess = binner_tess.find_bin_edges()
    shifted_bins_tess = binner_tess.shift_bin_edges(all_bins_tess)
    # Check that the last bin edge is 1
    assert np.isclose(shifted_bins_tess[-1], 1)
    assert np.all(shifted_bins_tess <= 1) and np.all(shifted_bins_tess >= 0)


def test_bin_calculation(wrapped_light_curve, unwrapped_light_curve):
    """
    Test the bin calculation capabilities of EclipsingBinaryBinner.
    """
    # Test wrapped light curve
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
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
    assert np.all(np.bincount(bin_numbers) > 0)

    # Test unwrapped light curve
    phases, fluxes, flux_errors = unwrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
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
    assert np.all(np.bincount(bin_numbers) > 0)


def test_bin_calculation_with_real_data(
    asas_sn_unwrapped_light_curve, tess_unwrapped_light_curve
):
    """
    Test EclipsingBinaryBinner's bin calculation with real ASAS-SN and TESS data.
    This test ensures the bin calculation method handles real-world data correctly.
    """
    # Test ASAS-SN unwrapped light curve
    phases, fluxes, flux_errors = asas_sn_unwrapped_light_curve
    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=200
    )
    bin_centers, bin_means, bin_errors, bin_numbers, _ = (
        binner.calculate_bins()
    )
    # Consolidate assertions into a helper function
    def assert_bins(bin_centers, bin_means, bin_errors, bin_numbers):
        assert len(bin_centers) > 0
        assert len(bin_means) == len(bin_centers)
        assert len(bin_errors) == len(bin_centers)
        assert np.all(bin_errors > 0)
        assert not np.any(np.isnan(bin_centers))
        assert not np.any(np.isnan(bin_means))
        assert not np.any(np.isnan(bin_errors))
        assert np.all(bin_centers <= 1) and np.all(bin_centers >= 0)
        assert len(np.unique(bin_centers)) == len(bin_centers)
        assert np.all(np.bincount(bin_numbers) > 0)

    assert_bins(bin_centers, bin_means, bin_errors, bin_numbers)

    # Test TESS unwrapped light curve
    phases, fluxes, flux_errors = tess_unwrapped_light_curve
    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=50
    )
    bin_centers, bin_means, bin_errors, bin_numbers, _ = (
        binner.calculate_bins()
    )
    assert_bins(bin_centers, bin_means, bin_errors, bin_numbers)


def test_plot_functions(wrapped_light_curve):
    """
    Test the plotting capabilities of EclipsingBinaryBinner.
    """
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)
    bin_centers, bin_means, bin_errors = binner.bin_light_curve(plot=True)
    binner.plot_binned_light_curve(bin_centers, bin_means, bin_errors)
    binner.plot_unbinned_light_curve()
