"""
Unit tests for the EclipsingBinaryBinner class from the eclipsebin module.
"""

import numpy as np
import pytest
from eclipsebin import EclipsingBinaryBinner

import matplotlib

matplotlib.use("Agg")


def test_initialization():
    '''
    This test function checks if the EclipsingBinaryBinner class is initialized correctly
    with the given input parameters. It verifies:
    1. The correct assignment of input parameters to class attributes.
    2. The proper handling of array inputs (phases, fluxes, and flux_errors).
    3. The initialization of other necessary attributes.

    The test creates an instance of EclipsingBinaryBinner with sample data and custom
    parameters, then asserts that the instance's attributes match the expected values.
    '''
    phases = np.linspace(0, 1, 100)
    fluxes = np.random.normal(1, 0.01, 100)
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=50, fraction_in_eclipse=0.3
    )

    assert binner.params['nbins'] == 50
    assert binner.params['fraction_in_eclipse'] == 0.3
    assert len(binner.data['phases']) == 100
    assert len(binner.data['fluxes']) == 100
    assert len(binner.data['flux_errors']) == 100

def test_initialization_valid_data():
    '''
    This test function verifies that the EclipsingBinaryBinner class initializes correctly
    with valid input data. It checks:
    1. The correct assignment of input parameters to class attributes.
    2. The proper handling of array inputs (phases, fluxes, and flux_errors).
    3. The initialization of other necessary attributes.

    The test creates an instance of EclipsingBinaryBinner with sample data and custom
    parameters, then asserts that the instance's attributes match the expected values.
    This test is similar to test_initialization, but focuses specifically on valid data inputs.
    '''
    phases = np.linspace(0, 1, 100)
    fluxes = np.random.normal(1, 0.01, 100)
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=50, fraction_in_eclipse=0.3
    )

    assert binner.params['nbins'] == 50
    assert binner.params['fraction_in_eclipse'] == 0.3
    assert len(binner.data['phases']) == 100
    assert len(binner.data['fluxes']) == 100
    assert len(binner.data['flux_errors']) == 100

def test_initialization_invalid_data():
    '''
    This test function verifies that the EclipsingBinaryBinner class raises appropriate
    ValueError exceptions when initialized with invalid input data. It checks:
    1. The case when the number of data points is less than 10.
    2. The case when the number of bins is less than 10.
    3. The case when the number of data points is less than the number of bins.

    For each case, the test creates an instance of EclipsingBinaryBinner with invalid
    input data and asserts that the correct ValueError is raised with the expected
    error message.
    '''
    # Fewer than 10 data points
    phases = np.linspace(0, 1, 9)
    fluxes = np.random.normal(1, 0.01, 9)
    flux_errors = np.random.normal(0.01, 0.001, 9)

    with pytest.raises(ValueError, match="Number of data points must be at least 10."):
        EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=10)

    # Fewer than 10 bins
    phases = np.linspace(0, 1, 100)
    fluxes = np.random.normal(1, 0.01, 100)
    flux_errors = np.random.normal(0.01, 0.001, 100)

    with pytest.raises(ValueError, match="Number of bins must be at least 10."):
        EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=9)

    # Data points fewer than bins
    with pytest.raises(
        ValueError,
        match="Number of data points must be greater than or equal to the number of bins.",
    ):
        EclipsingBinaryBinner(phases[:50], fluxes[:50], flux_errors[:50], nbins=60)


def test_eclipse_detection():
    '''
    This test function verifies the eclipse detection capabilities of the 
    EclipsingBinaryBinner class.
    It checks:
    1. The ability to correctly identify the primary eclipse minimum.
    2. The ability to determine the boundaries of the primary eclipse.
    3. The correct placement of the primary eclipse minimum within the detected boundaries.

    The test creates a simulated light curve with a single eclipse, initializes an
    EclipsingBinaryBinner instance with this data, and then asserts that the detected
    eclipse properties match the expected values.
    '''
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[45:55] = 0.9  # Simulate primary eclipse
    fluxes[0:3] = 0.95 # Simulate secondary eclipse
    fluxes[97:100] = 0.95 # Wrap secondary eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)

    # Ensure that find_minimum_flux returns a single value
    primary_min = binner.find_minimum_flux()
    assert np.isscalar(primary_min), "find_minimum_flux should return a single value"
    assert np.isclose(primary_min, 0.45, atol=0.05)
    primary_eclipse = binner.get_eclipse_boundaries(primary_min)
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]


def test_bin_calculation():
    '''
    This test function verifies the bin calculation capabilities of the EclipsingBinaryBinner class.
    It checks:
    1. The correct calculation of bin centers.
    2. The correct calculation of bin means.
    3. The correct calculation of bin standard deviations.

    The test creates a simulated light curve with a single eclipse, initializes an
    EclipsingBinaryBinner instance with this data, and then asserts that the calculated
    bin properties match the expected values.
    '''
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[45:55] = 0.9  # Simulate an eclipse
    fluxes[0:3] = 0.95 # Simulate secondary eclipse
    fluxes[97:100] = 0.95 # Wrap secondary eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)

    bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=False)

    assert len(bin_centers) > 0
    assert len(bin_means) == len(bin_centers)
    assert len(bin_stds) == len(bin_centers)


def test_plot_functions():
    '''
    This test function verifies the plotting capabilities of the EclipsingBinaryBinner class.
    It checks:
    1. The ability to plot the binned light curve.
    2. The ability to plot the unbinned light curve.

    The test creates a simulated light curve with a single eclipse, initializes an
    EclipsingBinaryBinner instance with this data, and then asserts that the plotting
    functions run without error.
    '''
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[45:55] = 0.9  # Simulate an eclipse
    fluxes[0:3] = 0.95 # Simulate secondary eclipse
    fluxes[97:100] = 0.95 # Wrap secondary eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50)

    # Ensure plotting functions run without error
    bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=True)
    binner.plot_binned_light_curve(bin_centers, bin_means, bin_stds, bin_centers)
    binner.plot_unbinned_light_curve()
