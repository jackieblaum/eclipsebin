import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class EclipsingBinaryBinner:
    def __init__(self, phases, fluxes, fluxerrs, nbins=200, frac_in_ecl=0.2):
        self.phases = phases
        self.fluxes = fluxes
        self.fluxerrs = fluxerrs
        self.nbins = nbins
        self.frac_in_ecl = frac_in_ecl

        self.primary_min = self.find_minimum_flux()
        self.secondary_min = self.find_secondary_minimum()

        self.primary_eclipse = self.find_eclipse(self.primary_min)
        self.secondary_eclipse = self.find_eclipse(self.secondary_min)

    def find_minimum_flux(self):
        idx_min = np.argmin(self.fluxes)
        return self.phases[idx_min]

    def find_secondary_minimum(self):
        mask = np.abs(self.phases - self.primary_min) > 0.2
        idx_secondary_min = np.argmin(self.fluxes[mask])
        return self.phases[mask][idx_secondary_min]

    def find_eclipse(self, phase_min):
        idx_start, idx_end = self.find_eclipse_boundaries(phase_min)
        return (self.phases[idx_start], self.phases[idx_end])

    def find_eclipse_boundaries(self, phase_min):
        idx_start = self.find_eclipse_boundary(phase_min, direction='start')
        idx_end = self.find_eclipse_boundary(phase_min, direction='end')
        return idx_start, idx_end

    def find_eclipse_boundary(self, phase_min, direction):
        if direction == 'start':
            mask = (self.phases < phase_min)
        else:  # direction == 'end'
            mask = (self.phases > phase_min)

        idx_boundary = np.where(mask & np.isclose(self.fluxes, 1.0, atol=0.01))[0]

        if len(idx_boundary) == 0:
            if direction == 'start':
                return np.where(np.isclose(self.fluxes, 1.0, atol=0.01))[0][-1]
            else:
                return np.where(np.isclose(self.fluxes, 1.0, atol=0.01))[0][0]
        else:
            return idx_boundary[-1] if direction == 'start' else idx_boundary[0]

    def calculate_bins(self):
        bins_in_primary = int((self.nbins * self.frac_in_ecl) / 2)
        bins_in_secondary = int((self.nbins * self.frac_in_ecl) - bins_in_primary)

        primary_bin_edges = self.calculate_eclipse_bins(self.primary_eclipse, bins_in_primary)
        secondary_bin_edges = self.calculate_eclipse_bins(self.secondary_eclipse, bins_in_secondary)

        ooe1_bins, ooe2_bins = self.calculate_out_of_eclipse_bins(bins_in_primary, bins_in_secondary)

        all_bins = np.sort(np.concatenate((primary_bin_edges, secondary_bin_edges, ooe1_bins, ooe2_bins)))
        bin_means, bin_edges, binnumber = stats.binned_statistic(self.phases, self.fluxes, statistic='mean', bins=all_bins)
        bin_centers = (bin_edges[1:] - bin_edges[:-1]) / 2 + bin_edges[:-1]
        bin_stds, _, bin_number = stats.binned_statistic(self.phases, self.fluxes, statistic='std', bins=all_bins)

        return bin_centers, bin_means, bin_stds, bin_number, bin_edges

    def calculate_eclipse_bins(self, eclipse, bins_in_eclipse):
        idx_start, idx_end = np.searchsorted(self.phases, eclipse)
        eclipse_phases = np.concatenate((self.phases[idx_start:], self.phases[:idx_end + 1] + 1)) if idx_end < idx_start else self.phases[idx_start:idx_end + 1]
        bins = pd.qcut(eclipse_phases, q=bins_in_eclipse)
        return np.array([interval.right for interval in np.unique(bins)]) % 1

    def calculate_out_of_eclipse_bins(self, bins_in_primary, bins_in_secondary):
        bins_in_ooe1 = int((self.nbins - bins_in_primary - bins_in_secondary) / 2)
        bins_in_ooe2 = self.nbins - bins_in_primary - bins_in_secondary - bins_in_ooe1

        ooe1_bins = pd.qcut(np.concatenate((self.phases[np.searchsorted(self.phases, self.secondary_eclipse[1]):],
                                            self.phases[:np.searchsorted(self.phases, self.primary_eclipse[0])] + 1)), q=bins_in_ooe1)
        ooe1_edges = np.array([interval.right for interval in np.unique(ooe1_bins)])[:-1] % 1

        ooe2_bins = pd.qcut(np.concatenate((self.phases[np.searchsorted(self.phases, self.primary_eclipse[1]):],
                                            self.phases[:np.searchsorted(self.phases, self.secondary_eclipse[0])] + 1)), q=bins_in_ooe2 + 2)
        ooe2_edges = np.array([interval.right for interval in np.unique(ooe2_bins)])[:-1] % 1

        return ooe1_edges, ooe2_edges

    def plot_binned_light_curve(self, bin_centers, bin_means, bin_stds, bin_edges):
        plt.figure(figsize=(20, 5))
        plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='.', color='red')
        plt.scatter(self.phases, self.fluxes, s=20)
        plt.vlines(self.primary_eclipse, ymin=0.6, ymax=1.1, linestyle='--', color='red')
        plt.vlines(self.secondary_eclipse, ymin=0.6, ymax=1.1, linestyle='--', color='red')
        plt.vlines(bin_edges, ymin=0.6, ymax=1.1, linestyle='--', color='green')
        plt.ylim(0.8, 1.05)
        plt.xlim(0.97, 1.001)
        plt.show()

    def plot_unbinned_light_curve(self):
        plt.figure(figsize=(20, 5))
        plt.scatter(self.phases, self.fluxes, s=3)
        plt.scatter(self.primary_min, min(self.fluxes), c='red', s=5)
        plt.scatter(self.secondary_min, min(self.fluxes[np.abs(self.phases - self.secondary_min) > 0.2]), c='red', s=5)
        plt.vlines(self.primary_eclipse, ymin=0.6, ymax=1.1, linestyle='--', color='red')
        plt.vlines(self.secondary_eclipse, ymin=0.6, ymax=1.1, linestyle='--', color='red')
        plt.ylim(0.8, 1.05)
        plt.show()

    def bin_light_curve(self, plot=True):
        bin_centers, bin_means, bin_stds, bin_number, bin_edges = self.calculate_bins()
        
        if plot:
            self.plot_binned_light_curve(bin_centers, bin_means, bin_stds, bin_edges)
            self.plot_unbinned_light_curve()

        return bin_centers, bin_means, bin_stds
