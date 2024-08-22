import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class EclipsingBinaryBinner:
    """
    A class to perform non-uniform binning of eclipsing binary star light curves.

    This class identifies primary and secondary eclipses within the light curve
    and allocates bins to better capture these eclipse events, while also binning
    the out-of-eclipse regions.

    Attributes:
        phases (np.ndarray): Array of phase values corresponding to the light curve.
        fluxes (np.ndarray): Array of flux values corresponding to the light curve.
        flux_errors (np.ndarray): Array of flux errors corresponding to the light curve.
        nbins (int): Total number of bins to be used for the light curve.
        fraction_in_eclipse (float): Fraction of bins allocated to the eclipse regions.
        primary_eclipse_min_phase (float): Phase value of the primary eclipse minimum.
        secondary_eclipse_min_phase (float): Phase value of the secondary eclipse minimum.
        primary_eclipse (tuple): Start and end phase values of the primary eclipse.
        secondary_eclipse (tuple): Start and end phase values of the secondary eclipse.
    """

    def __init__(self, phases, fluxes, flux_errors, nbins=200, fraction_in_eclipse=0.2):
        """
        Initializes the EclipsingBinaryBinner with the given light curve data and parameters.

        Args:
            phases (np.ndarray): Array of phase values.
            fluxes (np.ndarray): Array of flux values.
            flux_errors (np.ndarray): Array of flux errors.
            nbins (int, optional): Number of bins to use. Defaults to 200.
            fraction_in_eclipse (float, optional): Fraction of bins within eclipses. Defaults to 0.2.

        Raises:
            ValueError: If the number of data points is less than 10, or if the number of bins is less than 10,
                        or if the number of data points is less than the number of bins.
        """
        if len(phases) < 10:
            raise ValueError("Number of data points must be at least 10.")
        if nbins < 10:
            raise ValueError("Number of bins must be at least 10.")
        if len(phases) < nbins:
            raise ValueError(
                "Number of data points must be greater than or equal to the number of bins."
            )

        self.phases = phases
        self.fluxes = fluxes
        self.flux_errors = flux_errors
        self.nbins = nbins
        self.fraction_in_eclipse = fraction_in_eclipse

        # Identify primary and secondary eclipse minima
        self.primary_eclipse_min_phase = self.find_minimum_flux()
        self.secondary_eclipse_min_phase = self.find_secondary_minimum()

        # Determine start and end of each eclipse
        self.primary_eclipse = self.get_eclipse_boundaries(
            self.primary_eclipse_min_phase
        )
        self.secondary_eclipse = self.get_eclipse_boundaries(
            self.secondary_eclipse_min_phase
        )

    def find_minimum_flux(self):
        """
        Finds the phase of the minimum flux, corresponding to the primary eclipse.

        Returns:
            float: Phase value of the primary eclipse minimum.
        """
        idx_min = np.argmin(self.fluxes)
        return self.phases[idx_min]

    def find_secondary_minimum(self):
        """
        Finds the phase of the secondary eclipse by identifying the minimum flux
        at least 0.2 phase units away from the primary eclipse.

        Returns:
            float: Phase value of the secondary eclipse minimum.
        """
        mask = np.abs(self.phases - self.primary_eclipse_min_phase) > 0.2
        idx_secondary_min = np.argmin(self.fluxes[mask])
        return self.phases[mask][idx_secondary_min]

    def get_eclipse_boundaries(self, eclipse_min_phase):
        """
        Finds the start and end phase of an eclipse based on the minimum flux.

        Args:
            eclipse_min_phase (float): Phase of the minimum flux.

        Returns:
            tuple: Start and end phases of the eclipse.
        """
        start_idx, end_idx = self._find_eclipse_boundaries(eclipse_min_phase)
        return (self.phases[start_idx], self.phases[end_idx])

    def _find_eclipse_boundaries(self, eclipse_min_phase):
        """
        Determines the start and end indices of an eclipse.

        Args:
            eclipse_min_phase (float): Phase of the minimum flux.

        Returns:
            tuple: Indices of the start and end of the eclipse.
        """
        start_idx = self._find_eclipse_boundary(eclipse_min_phase, direction="start")
        end_idx = self._find_eclipse_boundary(eclipse_min_phase, direction="end")
        return start_idx, end_idx

    def _find_eclipse_boundary(self, eclipse_min_phase, direction):
        """
        Finds the boundary index of an eclipse either before (start) or after (end) the minimum flux.

        Args:
            eclipse_min_phase (float): Phase of the minimum flux.
            direction (str): Direction to search ('start' or 'end').

        Returns:
            int: Index of the boundary point.
        """
        if direction == "start":
            mask = self.phases < eclipse_min_phase
        else:  # direction == 'end'
            mask = self.phases > eclipse_min_phase

        idx_boundary = np.where(mask & np.isclose(self.fluxes, 1.0, atol=0.01))[0]

        if len(idx_boundary) == 0:
            # If no boundary found, use the closest point to 1.0 flux
            if direction == "start":
                return np.where(np.isclose(self.fluxes, 1.0, atol=0.01))[0][-1]
            else:
                return np.where(np.isclose(self.fluxes, 1.0, atol=0.01))[0][0]
        else:
            # Return the last or first index depending on direction
            return idx_boundary[-1] if direction == "start" else idx_boundary[0]

    def calculate_bins(self):
        """
        Calculates the bin centers, means, and standard deviations for the binned light curve.

        Returns:
            tuple: Arrays of bin centers, bin means, bin standard deviations, bin numbers, and bin edges.
        """
        bins_in_primary = int((self.nbins * self.fraction_in_eclipse) / 2)
        bins_in_secondary = int(
            (self.nbins * self.fraction_in_eclipse) - bins_in_primary
        )

        primary_bin_edges = self.calculate_eclipse_bins(
            self.primary_eclipse, bins_in_primary
        )
        secondary_bin_edges = self.calculate_eclipse_bins(
            self.secondary_eclipse, bins_in_secondary
        )

        ooe1_bins, ooe2_bins = self.calculate_out_of_eclipse_bins(
            bins_in_primary, bins_in_secondary
        )

        all_bins = np.sort(
            np.concatenate(
                (primary_bin_edges, secondary_bin_edges, ooe1_bins, ooe2_bins)
            )
        )
        bin_means, bin_edges, bin_number = stats.binned_statistic(
            self.phases, self.fluxes, statistic="mean", bins=all_bins
        )
        bin_centers = (bin_edges[1:] - bin_edges[:-1]) / 2 + bin_edges[:-1]
        bin_stds, _, bin_number = stats.binned_statistic(
            self.phases, self.fluxes, statistic="std", bins=all_bins
        )

        return bin_centers, bin_means, bin_stds, bin_number, bin_edges

    def calculate_eclipse_bins(self, eclipse_boundaries, bins_in_eclipse):
        """
        Calculates bin edges within an eclipse.

        Args:
            eclipse_boundaries (tuple): Start and end phases of the eclipse.
            bins_in_eclipse (int): Number of bins within the eclipse.

        Returns:
            np.ndarray: Array of bin edges within the eclipse.
        """
        start_idx, end_idx = np.searchsorted(self.phases, eclipse_boundaries)
        eclipse_phases = (
            np.concatenate((self.phases[start_idx:], self.phases[: end_idx + 1] + 1))
            if end_idx < start_idx
            else self.phases[start_idx : end_idx + 1]
        )
        bins = pd.qcut(eclipse_phases, q=bins_in_eclipse)
        return np.array([interval.right for interval in np.unique(bins)]) % 1

    def calculate_out_of_eclipse_bins(self, bins_in_primary, bins_in_secondary):
        """
        Calculates bin edges for out-of-eclipse regions.

        Args:
            bins_in_primary (int): Number of bins in the primary eclipse.
            bins_in_secondary (int): Number of bins in the secondary eclipse.

        Returns:
            tuple: Arrays of bin edges for the two out-of-eclipse regions.
        """
        bins_in_ooe1 = int((self.nbins - bins_in_primary - bins_in_secondary) / 2)
        bins_in_ooe2 = self.nbins - bins_in_primary - bins_in_secondary - bins_in_ooe1

        ooe1_bins = pd.qcut(
            np.concatenate(
                (
                    self.phases[
                        np.searchsorted(self.phases, self.secondary_eclipse[1]) :
                    ],
                    self.phases[: np.searchsorted(self.phases, self.primary_eclipse[0])]
                    + 1,
                )
            ),
            q=bins_in_ooe1,
        )
        ooe1_edges = (
            np.array([interval.right for interval in np.unique(ooe1_bins)])[:-1] % 1
        )

        ooe2_bins = pd.qcut(
            np.concatenate(
                (
                    self.phases[
                        np.searchsorted(self.phases, self.primary_eclipse[1]) :
                    ],
                    self.phases[
                        : np.searchsorted(self.phases, self.secondary_eclipse[0])
                    ]
                    + 1,
                )
            ),
            q=bins_in_ooe2 + 2,
        )
        ooe2_edges = (
            np.array([interval.right for interval in np.unique(ooe2_bins)])[:-1] % 1
        )

        return ooe1_edges, ooe2_edges

    def plot_binned_light_curve(self, bin_centers, bin_means, bin_stds, bin_edges):
        """
        Plots the binned light curve and the bin edges.

        Args:
            bin_centers (np.ndarray): Array of bin centers.
            bin_means (np.ndarray): Array of bin means.
            bin_stds (np.ndarray): Array of bin standard deviations.
            bin_edges (np.ndarray): Array of bin edges.
        """
        plt.figure(figsize=(20, 5))
        plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt=".", color="red")
        plt.scatter(self.phases, self.fluxes, s=20)
        plt.vlines(
            self.primary_eclipse, ymin=0.6, ymax=1.1, linestyle="--", color="red"
        )
        plt.vlines(
            self.secondary_eclipse, ymin=0.6, ymax=1.1, linestyle="--", color="red"
        )
        plt.vlines(bin_edges, ymin=0.6, ymax=1.1, linestyle="--", color="green")
        plt.ylim(0.8, 1.05)
        plt.xlim(0.97, 1.001)
        plt.show()

    def plot_unbinned_light_curve(self):
        """
        Plots the unbinned light curve with the calculated eclipse minima and bin edges.
        """
        plt.figure(figsize=(20, 5))
        plt.scatter(self.phases, self.fluxes, s=3)
        plt.scatter(self.primary_eclipse_min_phase, min(self.fluxes), c="red", s=5)
        plt.scatter(
            self.secondary_eclipse_min_phase,
            min(
                self.fluxes[
                    np.abs(self.phases - self.secondary_eclipse_min_phase) > 0.2
                ]
            ),
            c="red",
            s=5,
        )
        plt.vlines(
            self.primary_eclipse, ymin=0.6, ymax=1.1, linestyle="--", color="red"
        )
        plt.vlines(
            self.secondary_eclipse, ymin=0.6, ymax=1.1, linestyle="--", color="red"
        )
        plt.ylim(0.8, 1.05)
        plt.show()

    def bin_light_curve(self, plot=True):
        """
        Bins the light curve data and optionally plots the results.

        Args:
            plot (bool, optional): Whether to plot the binned and unbinned light curves. Defaults to True.

        Returns:
            tuple: Arrays of bin centers, bin means, and bin standard deviations.
        """
        bin_centers, bin_means, bin_stds, bin_number, bin_edges = self.calculate_bins()

        if plot:
            self.plot_binned_light_curve(bin_centers, bin_means, bin_stds, bin_edges)
            self.plot_unbinned_light_curve()

        return bin_centers, bin_means, bin_stds
