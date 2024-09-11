"""
This module contains the EclipsingBinaryBinner class, which performs non-uniform binning
of eclipsing binary star light curves.
"""

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
        data (dict): Dictionary containing the light curve data.
        params (dict): Dictionary containing the binning parameters.
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
            fraction_in_eclipse (float, optional): Fraction of bins within eclipses.
                Defaults to 0.2.

        Raises:
            ValueError: If the number of data points is less than 10, or if the number of bins
                is less than 10, or if the number of data points is less than the number of bins.
        """
        if len(phases) < 10:
            raise ValueError("Number of data points must be at least 10.")
        if nbins < 10:
            raise ValueError("Number of bins must be at least 10.")
        if len(phases) < nbins:
            raise ValueError(
                "Number of data points must be greater than or equal to the number of bins."
            )
        sort_idx = np.argsort(phases)
        self.data = {
            "phases": phases[sort_idx],
            "fluxes": fluxes[sort_idx],
            "flux_errors": flux_errors[sort_idx],
        }
        self.params = {"nbins": nbins, "fraction_in_eclipse": fraction_in_eclipse}

        # Identify primary and secondary eclipse minima
        self.primary_eclipse_min_phase = self.find_minimum_flux()
        self.secondary_eclipse_min_phase = self.find_secondary_minimum()

        # Determine start and end of each eclipse
        self.primary_eclipse = self.get_eclipse_boundaries(primary=True)
        self.secondary_eclipse = self.get_eclipse_boundaries(primary=False)

    def find_minimum_flux(self, use_shifted_phases=False):
        """
        Finds the phase of the minimum flux, corresponding to the primary eclipse.

        Returns:
            float: Phase value of the primary eclipse minimum.
        """
        if use_shifted_phases:
            phases = self.data["shifted_phases"]
        else:
            phases = self.data["phases"]
        idx_min = np.argmin(self.data["fluxes"])
        return phases[idx_min]

    def find_secondary_minimum(self, use_shifted_phases=False):
        """
        Finds the phase of the secondary eclipse by identifying the minimum flux
        at least 0.2 phase units away from the primary eclipse.

        Returns:
            float: Phase value of the secondary eclipse minimum.
        """
        if use_shifted_phases:
            phases = self.data["shifted_phases"]
            primary_min_phase = self.find_minimum_flux(use_shifted_phases=True)
        else:
            phases = self.data["phases"]
            primary_min_phase = self.primary_eclipse_min_phase
        mask = np.abs(phases - primary_min_phase) > 0.2
        idx_secondary_min = np.argmin(self.data["fluxes"][mask])
        return phases[mask][idx_secondary_min]

    def get_eclipse_boundaries(self, primary=True, use_shifted_phases=False):
        """
        Finds the start and end phase of an eclipse based on the minimum flux.

        Args:
            eclipse_min_phase (float): Phase of the minimum flux.

        Returns:
            tuple: Start and end phases of the eclipse.
        """
        if use_shifted_phases:
            phases = self.data["shifted_phases"]
            if primary:
                eclipse_min_phase = self.find_minimum_flux(use_shifted_phases=True)
            else:
                eclipse_min_phase = self.find_secondary_minimum(use_shifted_phases=True)
        else:
            phases = self.data["phases"]
            if primary:
                eclipse_min_phase = self.primary_eclipse_min_phase
            else:
                eclipse_min_phase = self.secondary_eclipse_min_phase
        start_idx, end_idx = self._find_eclipse_boundaries(
            eclipse_min_phase, use_shifted_phases=use_shifted_phases
        )
        return (phases[start_idx], phases[end_idx])

    def _find_eclipse_boundaries(self, eclipse_min_phase, use_shifted_phases=False):
        """
        Determines the start and end indices of an eclipse.

        Args:
            eclipse_min_phase (float): Phase of the minimum flux.

        Returns:
            tuple: Indices of the start and end of the eclipse.
        """
        start_idx = self._find_eclipse_boundary(
            eclipse_min_phase, direction="start", use_shifted_phases=use_shifted_phases
        )
        end_idx = self._find_eclipse_boundary(
            eclipse_min_phase, direction="end", use_shifted_phases=use_shifted_phases
        )
        return start_idx, end_idx

    def _find_eclipse_boundary(
        self, eclipse_min_phase, direction, use_shifted_phases=False
    ):
        """
        Finds the boundary index of an eclipse either before (start) or after (end)
            the minimum flux.

        Args:
            eclipse_min_phase (float): Phase of the minimum flux.
            direction (str): Direction to search ('start' or 'end').

        Returns:
            int: Index of the boundary point.
        """
        if use_shifted_phases:
            phases = self.data["shifted_phases"]
        else:
            phases = self.data["phases"]
        if direction == "start":
            mask = phases < eclipse_min_phase
        else:  # direction == 'end'
            mask = phases > eclipse_min_phase

        idx_boundary = np.where(mask & np.isclose(self.data["fluxes"], 1.0, atol=0.01))[
            0
        ]

        if len(idx_boundary) == 0:
            # If no boundary found, use the closest point to 1.0 flux
            if direction == "start":
                return np.where(np.isclose(self.data["fluxes"], 1.0, atol=0.01))[0][-1]
            return np.where(np.isclose(self.data["fluxes"], 1.0, atol=0.01))[0][0]
        # Return the last or first index depending on direction
        return idx_boundary[-1] if direction == "start" else idx_boundary[0]

    def find_bin_edges(self):
        """
        Finds the bin edges within the light curve.
        """
        bins_in_primary = int(
            (self.params["nbins"] * self.params["fraction_in_eclipse"]) / 2
        )
        bins_in_secondary = int(
            (self.params["nbins"] * self.params["fraction_in_eclipse"])
            - bins_in_primary
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
        # Now adjust the phases and bins so that the bin edges start and end at 0 and 1
        rightmost_edge = all_bins[-1]
        shifted_bins = all_bins + (1 - rightmost_edge)
        # Also shift the phases to match
        self.data["shifted_phases"] = (self.data["phases"] + (1 - rightmost_edge)) % 1
        return shifted_bins

    def calculate_bins(self):
        """
        Calculates the bin centers, means, and standard deviations for the binned light curve.

        Returns:
            tuple: Arrays of bin centers, bin means, bin standard deviations, bin numbers,
                and bin edges.
        """
        all_bins = self.find_bin_edges()
        bin_means, bin_edges, bin_number = stats.binned_statistic(
            self.data["shifted_phases"],
            self.data["fluxes"],
            statistic="mean",
            bins=all_bins,
        )
        bin_centers = (bin_edges[1:] - bin_edges[:-1]) / 2 + bin_edges[:-1]
        bin_errors = np.zeros(len(bin_means))
        # Calculate the propagated errors for each bin
        for i in range(len(all_bins) - 1):
            # Get the indices of the data points in this bin
            bin_mask = (self.data["shifted_phases"] >= all_bins[i]) & (
                self.data["shifted_phases"] < all_bins[i + 1]
            )
            # Get the errors for these data points
            flux_errors_in_bin = self.data["flux_errors"][bin_mask]
            # Calculate the propagated error for the bin
            bin_errors[i] = np.sqrt(np.sum(flux_errors_in_bin**2)) / len(
                flux_errors_in_bin
            )

        return bin_centers, bin_means, bin_errors, bin_number, bin_edges

    def calculate_eclipse_bins(self, eclipse_boundaries, bins_in_eclipse):
        """
        Calculates bin edges within an eclipse.

        Args:
            eclipse_boundaries (tuple): Start and end phases of the eclipse.
            bins_in_eclipse (int): Number of bins within the eclipse.

        Returns:
            np.ndarray: Array of bin edges within the eclipse.
        """
        start_idx, end_idx = np.searchsorted(self.data["phases"], eclipse_boundaries)
        eclipse_phases = (
            np.concatenate(
                (
                    self.data["phases"][start_idx:],
                    self.data["phases"][: end_idx + 1] + 1,
                )
            )
            if end_idx < start_idx
            else self.data["phases"][start_idx : end_idx + 1]
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
        bins_in_ooe1 = int(
            (self.params["nbins"] - bins_in_primary - bins_in_secondary) / 2
        )
        bins_in_ooe2 = (
            self.params["nbins"] - bins_in_primary - bins_in_secondary - bins_in_ooe1
        )

        # Calculate bin edges between end of secondary eclipse and start of primary eclipse
        end_idx_secondary_eclipse = np.searchsorted(
            self.data["phases"], self.secondary_eclipse[1]
        )
        start_idx_primary_eclipse = np.searchsorted(
            self.data["phases"], self.primary_eclipse[0]
        )
        ooe1_phases = (
            np.concatenate(
                (
                    self.data["phases"][end_idx_secondary_eclipse:],
                    self.data["phases"][: start_idx_primary_eclipse + 1] + 1,
                )
            )
            if end_idx_secondary_eclipse > start_idx_primary_eclipse
            else self.data["phases"][
                end_idx_secondary_eclipse : start_idx_primary_eclipse + 1
            ]
        )
        ooe1_bins = pd.qcut(ooe1_phases, q=bins_in_ooe1)
        ooe1_edges = np.array([interval.right for interval in np.unique(ooe1_bins)])

        # Calculate bin edges between end of primary eclipse and start of secondary eclipse
        end_idx_primary_eclipse = np.searchsorted(
            self.data["phases"], self.primary_eclipse[1]
        )
        start_idx_secondary_eclipse = np.searchsorted(
            self.data["phases"], self.secondary_eclipse[0]
        )
        ooe2_phases = (
            np.concatenate(
                (
                    self.data["phases"][end_idx_primary_eclipse:],
                    self.data["phases"][: start_idx_secondary_eclipse + 1] + 1,
                )
            )
            if end_idx_primary_eclipse > start_idx_secondary_eclipse
            else self.data["phases"][
                end_idx_primary_eclipse : start_idx_secondary_eclipse + 1
            ]
        )
        ooe2_bins = pd.qcut(ooe2_phases, q=bins_in_ooe2)
        ooe2_edges = np.array([interval.right for interval in np.unique(ooe2_bins)]) % 1

        return ooe1_edges, ooe2_edges

    def plot_binned_light_curve(self, bin_centers, bin_means, bin_stds):
        """
        Plots the binned light curve and the bin edges.

        Args:
            bin_centers (np.ndarray): Array of bin centers.
            bin_means (np.ndarray): Array of bin means.
            bin_stds (np.ndarray): Array of bin standard deviations.
            bin_edges (np.ndarray): Array of bin edges.
        """
        plt.figure(figsize=(20, 5))
        plt.title("Binned Light Curve")
        plt.errorbar(
            bin_centers, bin_means, yerr=bin_stds, linestyle="none", marker="."
        )
        plt.xlabel("Phases", fontsize=14)
        plt.ylabel("Normalized Flux", fontsize=14)
        plt.xlim(0, 1)
        ylims = plt.ylim()
        plt.vlines(
            self.get_eclipse_boundaries(primary=True, use_shifted_phases=True),
            ymin=ylims[0],
            ymax=ylims[1],
            linestyle="--",
            color="red",
            label="Primary Eclipse",
        )
        plt.vlines(
            self.get_eclipse_boundaries(primary=False, use_shifted_phases=True),
            ymin=ylims[0],
            ymax=ylims[1],
            linestyle="--",
            color="blue",
            label="Secondary Eclipse",
        )
        plt.ylim(ylims)
        plt.legend()
        plt.show()

    def plot_unbinned_light_curve(self):
        """
        Plots the unbinned light curve with the calculated eclipse minima and bin edges.
        """
        plt.figure(figsize=(20, 5))
        plt.title("Unbinned Light Curve")
        plt.errorbar(
            self.data["shifted_phases"],
            self.data["fluxes"],
            yerr=self.data["flux_errors"],
            linestyle="none",
            marker=".",
        )
        ylims = plt.ylim()
        plt.vlines(
            self.get_eclipse_boundaries(primary=True, use_shifted_phases=True),
            ymin=ylims[0],
            ymax=ylims[1],
            linestyle="--",
            color="red",
            label="Primary Eclipse",
        )
        plt.vlines(
            self.get_eclipse_boundaries(primary=False, use_shifted_phases=True),
            ymin=ylims[0],
            ymax=ylims[1],
            linestyle="--",
            color="blue",
            label="Secondary Eclipse",
        )
        plt.ylim(ylims)
        plt.xlim(0, 1)
        plt.ylabel("Normalized Flux", fontsize=14)
        plt.xlabel("Phases", fontsize=14)
        plt.legend()
        plt.show()

    def bin_light_curve(self, plot=True):
        """
        Bins the light curve data and optionally plots the results.

        Args:
            plot (bool, optional): Whether to plot the binned and unbinned light curves.
             Defaults to True.

        Returns:
            tuple: Arrays of bin centers, bin means, and bin standard deviations.
        """
        bin_centers, bin_means, bin_errors, _, _ = self.calculate_bins()

        if plot:
            self.plot_unbinned_light_curve()
            self.plot_binned_light_curve(bin_centers, bin_means, bin_errors)

        return bin_centers, bin_means, bin_errors
