import numpy as np
import pytest
from eclipsebin import EclipsingBinaryBinner

import matplotlib
matplotlib.use('Agg')

def test_initialization():
    phases = np.linspace(0, 1, 100)
    fluxes = np.random.normal(1, 0.01, 100)
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50, fraction_in_eclipse=0.3)

    assert binner.nbins == 50
    assert binner.fraction_in_eclipse == 0.3
    assert len(binner.phases) == 100
    assert len(binner.fluxes) == 100
    assert len(binner.flux_errors) == 100

def test_initialization_valid_data():
    phases = np.linspace(0, 1, 100)
    fluxes = np.random.normal(1, 0.01, 100)
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=50, fraction_in_eclipse=0.3)

    assert binner.nbins == 50
    assert binner.fraction_in_eclipse == 0.3
    assert len(binner.phases) == 100
    assert len(binner.fluxes) == 100
    assert len(binner.flux_errors) == 100

def test_initialization_invalid_data():
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
    with pytest.raises(ValueError, match="Number of data points must be greater than or equal to the number of bins."):
        EclipsingBinaryBinner(phases[:50], fluxes[:50], flux_errors[:50], nbins=60)

def test_eclipse_detection():
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[45:55] = 0.9  # Simulate an eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors)

    primary_min = binner.find_minimum_flux()
    assert np.isclose(primary_min, 0.5, atol=0.05)

    primary_eclipse = binner.get_eclipse_boundaries(primary_min)
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]

def test_bin_calculation():
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[45:55] = 0.9  # Simulate an eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors)

    bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=False)

    assert len(bin_centers) > 0
    assert len(bin_means) == len(bin_centers)
    assert len(bin_stds) == len(bin_centers)

def test_plot_functions():
    phases = np.linspace(0, 1, 100)
    fluxes = np.ones_like(phases)
    fluxes[45:55] = 0.9  # Simulate an eclipse
    flux_errors = np.random.normal(0.01, 0.001, 100)

    binner = EclipsingBinaryBinner(phases, fluxes, flux_errors)

    # Ensure plotting functions run without error
    bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=True)
    binner.plot_binned_light_curve(bin_centers, bin_means, bin_stds, bin_centers)
    binner.plot_unbinned_light_curve()

