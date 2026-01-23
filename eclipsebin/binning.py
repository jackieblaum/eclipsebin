# pylint: disable=too-many-arguments
"""
This module contains the EclipsingBinaryBinner class, which performs non-uniform binning
of eclipsing binary star light curves.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import trim_mean
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import warnings


def _detect_eclipse_edges_slope(
    phases,
    fluxes,
    sigmas=None,
    smoothing_window=None,
    slope_threshold_percentile=90.0,
    return_threshold_fraction=0.1,
    min_eclipse_depth=0.01,
):
    """
    Detect eclipse boundaries using slope/derivative-based edge detection.

    This method is robust to ellipsoidal variations because it detects edges
    based on local slope changes rather than absolute flux levels.

    Parameters
    ----------
    phases : array
        Phase values (should be sorted and in [0, 1))
    fluxes : array
        Flux values (normalized to median~1 recommended)
    sigmas : array, optional
        Per-point uncertainties
    smoothing_window : int, optional
        Window size for Savitzky-Golay smoothing (must be odd).
        If None, auto-selects based on data density (~5% of points, minimum 5)
    slope_threshold_percentile : float, default=90.0
        Percentile of absolute slopes to use as threshold for detecting
        steep ingress/egress. Higher = more selective.
    return_threshold_fraction : float, default=0.1
        Fraction of maximum slope to use for defining boundary.
        Lower = boundaries closer to eclipse center.
    min_eclipse_depth : float, default=0.01
        Minimum flux drop to consider as eclipse (relative to local baseline)

    Returns
    -------
    eclipse_boundaries : list of tuples
        List of (ingress_phase, egress_phase) for each detected eclipse.
        Empty list if no eclipses detected.
    diagnostics : dict
        Dictionary containing diagnostic information:
        - slopes: array of computed slopes at each phase point
        - smoothed_fluxes: array of smoothed flux values
        - threshold: slope threshold value used for detection
        - return_threshold: threshold for boundary refinement
        - ingress_candidates: list of ingress candidate indices
        - egress_candidates: list of egress candidate indices
        - detected_count: number of eclipses detected
    """
    phases = np.asarray(phases, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)

    if len(phases) < 10:
        # Return minimal diagnostics for early return
        diagnostics = {
            "slopes": np.array([]),
            "smoothed_fluxes": np.array([]),
            "threshold": 0.0,
            "return_threshold": 0.0,
            "smoothing_window": smoothing_window,
            "baseline_window": 5,  # Would use minimum
            "refinement_range": 10,  # Would use minimum
            "ingress_candidates": [],
            "egress_candidates": [],
            "detected_count": 0,
        }
        return [], diagnostics

    # Ensure sorted
    idx = np.argsort(phases)
    phases = phases[idx]
    fluxes = fluxes[idx]

    # Auto-select smoothing window if not provided
    if smoothing_window is None:
        n_points = len(phases)
        window = max(5, int(0.05 * n_points))
        # Must be odd for Savitzky-Golay
        if window % 2 == 0:
            window += 1
        smoothing_window = min(window, n_points - 2)
        if smoothing_window < 5:
            smoothing_window = 5

    # Smooth the light curve to reduce noise
    try:
        smoothed_fluxes = savgol_filter(fluxes, smoothing_window, 3)
    except Exception:
        # Fallback to simple moving average if Savitzky-Golay fails
        smoothed_fluxes = uniform_filter1d(fluxes, size=smoothing_window)

    # Compute derivative (slope) using finite differences
    dphase = np.diff(phases)
    dflux = np.diff(smoothed_fluxes)

    # Handle phase wrapping: if gap is large, don't compute slope across it
    median_dphase = np.median(dphase[dphase > 0])
    # Detect gaps that are significantly larger than typical spacing
    # Use 10x as threshold but ensure it's at least 0.05 phase units
    gap_threshold = max(10 * median_dphase, 0.05)
    large_gap = dphase > gap_threshold

    slopes = np.zeros_like(phases)
    # Compute slopes, avoiding division by zero with epsilon
    slopes[1:] = dflux / (dphase + 1e-10)
    # Set slopes to zero where there are large gaps (indexing matches slopes[1:])
    slopes[1:][large_gap] = 0.0

    # Compute absolute slopes for thresholding
    abs_slopes = np.abs(slopes)

    # Determine threshold based on percentile
    valid_slopes = abs_slopes[abs_slopes > 0]
    if len(valid_slopes) == 0:
        diagnostics = {
            "slopes": slopes,
            "smoothed_fluxes": smoothed_fluxes,
            "threshold": 0.0,
            "return_threshold": 0.0,
            "smoothing_window": smoothing_window,
            "baseline_window": max(5, int(0.02 * len(fluxes))),
            "refinement_range": max(10, int(0.05 * len(phases))),
            "ingress_candidates": [],
            "egress_candidates": [],
            "detected_count": 0,
        }
        return [], diagnostics

    slope_threshold = np.percentile(valid_slopes, slope_threshold_percentile)
    return_threshold = slope_threshold * return_threshold_fraction

    # Find regions with steep negative slopes (ingress) and positive slopes (egress)
    ingress_mask = slopes < -slope_threshold
    egress_mask = slopes > slope_threshold

    # Find eclipse candidates by looking for ingress-egress pairs
    eclipse_boundaries = []
    ingress_indices = np.where(ingress_mask)[0]
    egress_indices = np.where(egress_mask)[0]

    if len(ingress_indices) == 0 or len(egress_indices) == 0:
        diagnostics = {
            "slopes": slopes,
            "smoothed_fluxes": smoothed_fluxes,
            "threshold": slope_threshold,
            "return_threshold": return_threshold,
            "smoothing_window": smoothing_window,
            "baseline_window": max(5, int(0.02 * len(fluxes))),
            "refinement_range": max(10, int(0.05 * len(phases))),
            "ingress_candidates": ingress_indices.tolist(),
            "egress_candidates": egress_indices.tolist(),
            "detected_count": 0,
        }
        return [], diagnostics

    # Adaptive baseline window: 2% of data, minimum 5 points
    baseline_window = max(5, int(0.02 * len(fluxes)))

    # Adaptive refinement range: 5% of data, minimum 10 points
    refinement_range = max(10, int(0.05 * len(phases)))

    # Pair up ingress and egress points
    i = 0
    while i < len(ingress_indices):
        ingress_idx = ingress_indices[i]
        egress_candidates = egress_indices[egress_indices > ingress_idx]

        if len(egress_candidates) > 0:
            egress_idx = egress_candidates[0]

            # Check if this is a real eclipse (flux drops significantly)
            eclipse_region = smoothed_fluxes[ingress_idx : egress_idx + 1]
            if len(eclipse_region) > 0:
                # Calculate baseline with validation
                pre_window_start = max(0, ingress_idx - baseline_window)
                pre_window_end = ingress_idx
                post_window_start = egress_idx
                post_window_end = min(len(fluxes), egress_idx + baseline_window)

                pre_window = smoothed_fluxes[pre_window_start:pre_window_end]
                post_window = smoothed_fluxes[post_window_start:post_window_end]

                # Need at least 3 points for reliable baseline
                if len(pre_window) < 3 or len(post_window) < 3:
                    i += 1
                    continue

                # Use trimmed mean for robustness against outliers (trim 10% from each end)
                try:
                    pre_baseline = (
                        trim_mean(pre_window, 0.1)
                        if len(pre_window) >= 5
                        else np.median(pre_window)
                    )
                    post_baseline = (
                        trim_mean(post_window, 0.1)
                        if len(post_window) >= 5
                        else np.median(post_window)
                    )
                    local_baseline = np.mean([pre_baseline, post_baseline])
                except Exception:
                    # Fallback to median if trimmed mean fails
                    local_baseline = np.median(
                        [np.median(pre_window), np.median(post_window)]
                    )

                # Validate baseline is reasonable (not too close to zero)
                if local_baseline < 0.1:
                    # Baseline too faint for reliable depth calculation
                    i += 1
                    continue

                min_flux = np.min(eclipse_region)
                depth = (local_baseline - min_flux) / local_baseline

                if depth >= min_eclipse_depth:
                    # Refine boundaries: find where slope crosses return_threshold
                    ingress_refined = ingress_idx
                    for j in range(
                        ingress_idx, max(0, ingress_idx - refinement_range), -1
                    ):
                        if abs_slopes[j] < return_threshold:
                            ingress_refined = j
                            break

                    egress_refined = egress_idx
                    for j in range(
                        egress_idx, min(len(phases), egress_idx + refinement_range)
                    ):
                        if abs_slopes[j] < return_threshold:
                            egress_refined = j
                            break

                    eclipse_boundaries.append(
                        (phases[ingress_refined], phases[egress_refined])
                    )

                    # Skip to after this egress
                    i = np.searchsorted(ingress_indices, egress_idx, side="right")
                    continue

        i += 1

    # Before collecting diagnostics, validate smoothing window
    if len(eclipse_boundaries) > 0:
        eclipse_widths = [egress - ingress for ingress, egress in eclipse_boundaries]
        min_eclipse_width = min(eclipse_widths)

        # Estimate points per eclipse
        avg_dphase = np.median(dphase[dphase > 0]) if len(dphase) > 0 else 0.001
        points_per_eclipse = min_eclipse_width / avg_dphase if avg_dphase > 0 else 0

        # Warn if smoothing window is more than 1/3 of narrowest eclipse
        if points_per_eclipse > 0 and smoothing_window > points_per_eclipse / 3:
            warnings.warn(
                f"Smoothing window ({smoothing_window} points) may be too large "
                f"for narrow eclipses (~{points_per_eclipse:.0f} points wide). "
                f"Consider reducing edge_smoothing_window or let it auto-select. "
                f"This may cause missed or poorly-defined eclipse boundaries.",
                UserWarning,
            )

    # Build diagnostics dictionary
    diagnostics = {
        "slopes": slopes,
        "smoothed_fluxes": smoothed_fluxes,
        "threshold": slope_threshold,
        "return_threshold": return_threshold,
        "smoothing_window": smoothing_window,
        "baseline_window": baseline_window,
        "refinement_range": refinement_range,
        "ingress_candidates": ingress_indices.tolist(),
        "egress_candidates": egress_indices.tolist(),
        "detected_count": len(eclipse_boundaries),
    }

    return eclipse_boundaries, diagnostics


def _find_primary_and_secondary_from_edges(
    eclipse_boundaries, phases, fluxes, min_separation=0.2
):
    """
    Identify which detected eclipse is primary vs secondary based on depth and phase separation.

    Parameters
    ----------
    eclipse_boundaries : list of tuples
        List of (ingress, egress) phase pairs
    phases : array
        Phase values
    fluxes : array
        Flux values
    min_separation : float, optional
        Minimum phase separation between primary and secondary eclipses. Defaults to 0.2.

    Returns
    -------
    primary_boundaries : tuple or None
        (ingress, egress) for primary eclipse
    secondary_boundaries : tuple or None
        (ingress, egress) for secondary eclipse
    """
    if len(eclipse_boundaries) == 0:
        return None, None

    if len(eclipse_boundaries) == 1:
        # Only one eclipse detected - assume it's primary
        return eclipse_boundaries[0], None

    # Calculate depths and centers for each eclipse
    eclipse_info = []
    for ingress, egress in eclipse_boundaries:
        # Find indices
        ingress_idx = np.argmin(np.abs(phases - ingress))
        egress_idx = np.argmin(np.abs(phases - egress))

        # Calculate local baseline
        local_baseline = np.median(
            [
                np.median(fluxes[max(0, ingress_idx - 10) : ingress_idx]),
                np.median(fluxes[egress_idx : min(len(fluxes), egress_idx + 10)]),
            ]
        )

        # Find minimum in eclipse region
        eclipse_region = fluxes[
            min(ingress_idx, egress_idx) : max(ingress_idx, egress_idx) + 1
        ]
        min_flux = np.min(eclipse_region)
        depth = (local_baseline - min_flux) / local_baseline

        # Calculate center phase
        center_phase = (ingress + egress) / 2.0
        if egress < ingress:  # Wrapped eclipse
            center_phase = (ingress + egress + 1.0) / 2.0 % 1.0

        eclipse_info.append(
            {"boundaries": (ingress, egress), "depth": depth, "center": center_phase}
        )

    # Primary is the deeper eclipse
    primary_idx = np.argmax([e["depth"] for e in eclipse_info])
    primary_boundaries = eclipse_info[primary_idx]["boundaries"]
    primary_center = eclipse_info[primary_idx]["center"]

    # Secondary is the other one, but must be at least min_separation phase units away
    secondary_boundaries = None
    for i, info in enumerate(eclipse_info):
        if i != primary_idx:
            # Check phase separation
            phase_sep = abs(info["center"] - primary_center)
            if phase_sep > 0.5:
                phase_sep = 1.0 - phase_sep
            if (
                phase_sep >= min_separation
            ):  # Secondary should be ~0.5 phase away (or at least min_separation)
                secondary_boundaries = info["boundaries"]
                break

    # If no secondary found with proper separation, use the other eclipse anyway
    if secondary_boundaries is None and len(eclipse_boundaries) > 1:
        secondary_idx = 1 - primary_idx
        secondary_boundaries = eclipse_boundaries[secondary_idx]

    return primary_boundaries, secondary_boundaries


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

    def __init__(
        self,
        phases,
        fluxes,
        flux_errors,
        nbins=200,
        fraction_in_eclipse=0.2,
        atol_primary=None,
        atol_secondary=None,
        boundary_method="flux_return",
        edge_slope_threshold_percentile=90.0,
        edge_return_threshold_fraction=0.1,
        edge_min_eclipse_depth=0.01,
        edge_smoothing_window=None,
        min_eclipse_separation=0.2,
    ):
        """
        Initializes the EclipsingBinaryBinner with the given light curve data and parameters.

        Args:
            phases (np.ndarray): Array of phase values.
            fluxes (np.ndarray): Array of flux values.
            flux_errors (np.ndarray): Array of flux errors.
            nbins (int, optional): Number of bins to use. Defaults to 200.
            fraction_in_eclipse (float, optional): Fraction of bins within eclipses.
                Defaults to 0.2.
            atol_primary (float, optional): Tolerance for primary eclipse boundary detection.
                Defaults to None (auto-calculated).
            atol_secondary (float, optional): Tolerance for secondary eclipse boundary detection.
                Defaults to None (auto-calculated).
            boundary_method (str, optional): Method for detecting eclipse boundaries.
                'flux_return' (default): Traditional method finding where flux returns to 1.0.
                'edge_detection': Slope-based edge detection, robust to ellipsoidal variations.
            edge_slope_threshold_percentile (float, optional): For edge_detection method,
                percentile of absolute slopes to use as threshold. Defaults to 90.0.
            edge_return_threshold_fraction (float, optional): For edge_detection method,
                fraction of max slope for boundary definition. Defaults to 0.1.
            edge_min_eclipse_depth (float, optional): For edge_detection method,
                minimum flux drop to consider as eclipse. Defaults to 0.01.
            edge_smoothing_window (int, optional): For edge_detection method,
                window size for smoothing. If None, auto-selected. Defaults to None.
            min_eclipse_separation (float, optional): Minimum phase separation between
                primary and secondary eclipses. Defaults to 0.2 (suitable for most
                circular orbits). Adjust for eccentric systems or close eclipses.

        Raises:
            ValueError: If the number of data points is less than 10, or if the number of bins
                is less than 10, or if the number of data points is not greater than the number of bins.
        """
        if len(phases) < 10:
            raise ValueError("Number of data points must be at least 10.")
        if nbins < 10:
            raise ValueError("Number of bins must be at least 10.")
        if len(phases) <= nbins:
            raise ValueError(
                "Number of data points must be greater than the number of bins."
            )
        if np.any(flux_errors) <= 0:
            raise ValueError("Flux errors must be > 0.")
        sort_idx = np.argsort(phases)
        self.data = {
            "phases": phases[sort_idx],
            "fluxes": fluxes[sort_idx],
            "flux_errors": flux_errors[sort_idx],
        }
        self.params = {
            "nbins": nbins,
            "fraction_in_eclipse": fraction_in_eclipse,
            "atol_primary": None,
            "atol_secondary": None,
        }

        # Store edge detection parameters
        self.boundary_method = boundary_method
        self.edge_slope_threshold_percentile = edge_slope_threshold_percentile
        self.edge_return_threshold_fraction = edge_return_threshold_fraction
        self.edge_min_eclipse_depth = edge_min_eclipse_depth
        self.edge_smoothing_window = edge_smoothing_window
        self.min_eclipse_separation = min_eclipse_separation

        # Initialize diagnostics storage
        self._edge_diagnostics = None

        # Detect and store original phase range for denormalization later
        self._original_phase_min = np.min(phases)
        self._original_phase_max = np.max(phases)
        self._original_phase_range = self._original_phase_max - self._original_phase_min

        # Normalize phases to [0, 1] if not already
        if self._original_phase_min < 0 or self._original_phase_max > 1:
            self._needs_denormalization = True
            normalized_phases = self._normalize_phases(phases)
            sort_idx = np.argsort(normalized_phases)
            self.data["phases"] = normalized_phases[sort_idx]
            self.data["fluxes"] = fluxes[sort_idx]
            self.data["flux_errors"] = flux_errors[sort_idx]
        else:
            self._needs_denormalization = False

        self.set_atol(primary=atol_primary, secondary=atol_secondary)

        # Identify primary and secondary eclipse minima (in original phase space)
        self.primary_eclipse_min_phase = self.find_minimum_flux_phase()
        self.secondary_eclipse_min_phase = self.find_secondary_minimum_phase()

        # Determine start and end of each eclipse (in original phase space)
        self.primary_eclipse = self.get_eclipse_boundaries(primary=True)
        self.secondary_eclipse = self.get_eclipse_boundaries(primary=False)

        # Calculate shift needed to unwrap any wrapped eclipses
        self._phase_shift = self._calculate_unwrap_shift()

        # Apply unwrapping if needed
        if self._phase_shift != 0.0:
            self._unwrap_phases()
            # Recalculate eclipse boundaries in unwrapped space
            self.primary_eclipse = self.get_eclipse_boundaries(primary=True)
            self.secondary_eclipse = self.get_eclipse_boundaries(primary=False)

    def _normalize_phases(self, phases):
        """
        Normalize phases from original range to [0, 1].

        Args:
            phases (np.ndarray): Phases in original range

        Returns:
            np.ndarray: Phases normalized to [0, 1]
        """
        # Shift so minimum is at 0, then scale to [0, 1]
        return (phases - self._original_phase_min) / self._original_phase_range

    def _denormalize_phases(self, phases):
        """
        Convert phases from [0, 1] back to original range.

        Args:
            phases (np.ndarray): Phases in [0, 1] range

        Returns:
            np.ndarray: Phases in original range
        """
        if not self._needs_denormalization:
            return phases
        # Scale from [0, 1] back to original range
        return phases * self._original_phase_range + self._original_phase_min

    def find_minimum_flux_phase(self):
        """
        Finds the phase of the minimum flux, corresponding to the primary eclipse.

        Returns:
            float: Phase value of the primary eclipse minimum.
        """
        phases = self.data["phases"]
        idx_min = np.argmin(self.data["fluxes"])
        return phases[idx_min]

    def find_minimum_flux(self):
        """
        Finds the minimum flux value in the light curve data.

        Returns:
            float: The minimum flux value.
        """
        return np.min(self.data["fluxes"])

    def find_secondary_minimum_phase(self):
        """
        Finds the phase of the secondary eclipse by identifying the minimum flux
        at least min_eclipse_separation phase units away from the primary eclipse.

        Returns:
            float: Phase value of the secondary eclipse minimum.
        """
        phases, mask = self._helper_secondary_minimum_mask()
        idx_secondary_min = np.argmin(self.data["fluxes"][mask])
        return phases[mask][idx_secondary_min]

    def find_secondary_minimum(self):
        """
        Finds the minimum flux value in the secondary eclipse region.

        Returns:
            float: The minimum flux value in the secondary eclipse region.
        """
        _, mask = self._helper_secondary_minimum_mask()
        return np.min(self.data["fluxes"][mask])

    def _helper_secondary_minimum_mask(self):
        phases = self.data["phases"]
        primary_min_phase = self.primary_eclipse_min_phase
        mask = np.abs(phases - primary_min_phase) > self.min_eclipse_separation
        return phases, mask

    def _detect_wrapped_eclipse(self, eclipse_start, eclipse_end):
        """
        Detect if an eclipse wraps around the phase boundary.

        Args:
            eclipse_start (float): Start phase of eclipse
            eclipse_end (float): End phase of eclipse

        Returns:
            bool: True if eclipse wraps around boundary (end < start)
        """
        return eclipse_end < eclipse_start

    def _calculate_unwrap_shift(self):
        """
        Calculate the phase shift needed to unwrap any wrapped eclipses.

        Returns:
            float: Phase shift amount (0 if no wrapping detected)
        """
        # Check if either eclipse is wrapped
        primary_wrapped = self._detect_wrapped_eclipse(
            self.primary_eclipse[0], self.primary_eclipse[1]
        )
        secondary_wrapped = self._detect_wrapped_eclipse(
            self.secondary_eclipse[0], self.secondary_eclipse[1]
        )

        if not (primary_wrapped or secondary_wrapped):
            return 0.0

        # Calculate shift to unwrap the wrapped eclipse
        # Shift to place the wrapped eclipse midpoint away from the 0/1 boundary
        # while keeping the unwrapped eclipse unwrapped
        if primary_wrapped and not secondary_wrapped:
            # Shift so primary unwraps but secondary stays unwrapped
            # Place the shift point between secondary end and primary start
            shift = (
                1.0 - self.primary_eclipse[0] + 0.05
            )  # Small offset to move primary start away from 0
        elif secondary_wrapped and not primary_wrapped:
            # Shift so secondary unwraps but primary stays unwrapped
            # Place the shift point between primary end and secondary start
            shift = (
                1.0 - self.secondary_eclipse[0] + 0.05
            )  # Small offset to move secondary start away from 0
        else:
            # Both wrapped (rare) - use 0.5
            shift = 0.5

        return shift % 1.0

    def _unwrap_phases(self):
        """
        Unwrap phases by applying the calculated shift.
        This ensures no eclipse crosses the 0/1 boundary.
        """
        self.data["phases"] = (self.data["phases"] + self._phase_shift) % 1.0
        # Re-sort after shifting
        sort_idx = np.argsort(self.data["phases"])
        self.data["phases"] = self.data["phases"][sort_idx]
        self.data["fluxes"] = self.data["fluxes"][sort_idx]
        self.data["flux_errors"] = self.data["flux_errors"][sort_idx]

        # Recalculate eclipse minima in unwrapped space
        # (they should be at the same flux values, just different phases)
        self.primary_eclipse_min_phase = self.find_minimum_flux_phase()
        self.secondary_eclipse_min_phase = self.find_secondary_minimum_phase()

    def get_eclipse_boundaries(self, primary=True):
        """
        Finds the start and end phase of an eclipse.

        Args:
            primary (bool): If True, get primary eclipse boundaries, else secondary.

        Returns:
            tuple: Start and end phases of the eclipse.
        """
        if self.boundary_method == "edge_detection":
            # Use edge detection method
            boundaries, diagnostics = _detect_eclipse_edges_slope(
                self.data["phases"],
                self.data["fluxes"],
                self.data["flux_errors"],
                smoothing_window=self.edge_smoothing_window,
                slope_threshold_percentile=self.edge_slope_threshold_percentile,
                return_threshold_fraction=self.edge_return_threshold_fraction,
                min_eclipse_depth=self.edge_min_eclipse_depth,
            )

            # Store diagnostics
            self._edge_diagnostics = diagnostics

            if len(boundaries) > 0:
                # Identify primary and secondary
                primary_bounds, secondary_bounds = (
                    _find_primary_and_secondary_from_edges(
                        boundaries,
                        self.data["phases"],
                        self.data["fluxes"],
                        self.min_eclipse_separation,
                    )
                )

                if primary:
                    if primary_bounds is not None:
                        return primary_bounds
                else:
                    if secondary_bounds is not None:
                        return secondary_bounds

                # If requested eclipse not found, fall back to flux_return
                eclipse_type = "primary" if primary else "secondary"
                diag_str = f"detected {len(boundaries)} eclipse(s) total"
                warnings.warn(
                    f"Edge detection did not find {eclipse_type} eclipse ({diag_str}). "
                    f"This may occur if the {eclipse_type} eclipse is very shallow "
                    f"(depth < {self.edge_min_eclipse_depth}) or if slope threshold "
                    f"is too high (percentile: {self.edge_slope_threshold_percentile}). "
                    f"Falling back to flux_return method."
                )
                # Temporarily switch to flux_return for this call
                original_method = self.boundary_method
                self.boundary_method = "flux_return"
                result = self.get_eclipse_boundaries(primary=primary)
                self.boundary_method = original_method
                return result
            else:
                # No eclipses detected, fall back to flux_return
                diag_str = (
                    f"threshold={diagnostics['threshold']:.3e}, "
                    f"{len(diagnostics['ingress_candidates'])} ingress candidates, "
                    f"{len(diagnostics['egress_candidates'])} egress candidates"
                )
                warnings.warn(
                    f"Edge detection found no eclipses ({diag_str}). "
                    f"Consider lowering edge_slope_threshold_percentile "
                    f"(current: {self.edge_slope_threshold_percentile}) or "
                    f"edge_min_eclipse_depth (current: {self.edge_min_eclipse_depth}). "
                    f"Falling back to flux_return method."
                )
                original_method = self.boundary_method
                self.boundary_method = "flux_return"
                result = self.get_eclipse_boundaries(primary=primary)
                self.boundary_method = original_method
                return result

        # Original flux_return method
        phases = self.data["phases"]
        if primary:
            eclipse_min_phase = self.primary_eclipse_min_phase
        else:
            eclipse_min_phase = self.secondary_eclipse_min_phase
        start_idx, end_idx = self._find_eclipse_boundaries(eclipse_min_phase)
        return (phases[start_idx], phases[end_idx])

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

    def _find_boundary_index(self, idx_boundary, phases, direction, atol):
        nbins = 100
        # bins = np.linspace(
        #     min(phases[idx_boundary]), max(phases[idx_boundary]), nbins + 1
        # )
        unbinned_data = pd.DataFrame(
            {
                "phase": phases[idx_boundary],
                "flux": self.data["fluxes"][idx_boundary],
            }
        )
        unbinned_data["phase_bin"] = pd.cut(unbinned_data["phase"], bins=nbins)
        binned_data = unbinned_data.groupby("phase_bin", observed=False)[
            "flux"
        ].median()
        binned_data = binned_data.dropna()  # Drop any bins with NaN values
        medians_closest_to_1 = np.where(np.isclose(binned_data, 1, atol=atol))[0]
        phase_bin_idx = (
            max(medians_closest_to_1)
            if direction == "start"
            else min(medians_closest_to_1)
        )
        selected_bin = binned_data.index[phase_bin_idx]
        # selected_bin = unbinned_data["phase_bin"].cat.categories[phase_bin_idx]
        mid = (selected_bin.left + selected_bin.right) / 2
        boundary_index = np.argmin(np.abs(phases - mid))
        return boundary_index

    def _find_eclipse_boundary(self, eclipse_min_phase, direction):
        """
        Finds the boundary index of an eclipse either before (start) or after (end)
            the minimum flux.

        Args:
            eclipse_min_phase (float): Phase of the minimum flux.
            direction (str): Direction to search ('start' or 'end').

        Returns:
            int: Index of the boundary point.
        """
        phases = self.data["phases"]
        if direction == "start":
            mask = phases < eclipse_min_phase
        else:  # direction == 'end'
            mask = phases > eclipse_min_phase

        min_flux_idx = np.where(phases == eclipse_min_phase)[0][0]
        min_flux = self.data["fluxes"][min_flux_idx]
        atol = self.get_atol(min_flux)

        idx_boundary = np.where(mask & np.isclose(self.data["fluxes"], 1.0, atol=atol))[
            0
        ]

        if len(idx_boundary) == 0:
            # If no boundary found, use the closest point to 1.0 flux
            idx_boundary = np.where(np.isclose(self.data["fluxes"], 1.0, atol=atol))[0]

        if len(idx_boundary) > 100:
            boundary_index = self._find_boundary_index(
                idx_boundary, phases, direction, atol
            )
            return boundary_index

        # Return the last or first index depending on direction
        boundary_phase = (
            max(phases[idx_boundary])
            if direction == "start"
            else min(phases[idx_boundary])
        )
        boundary_index = np.argmin(np.abs(phases - boundary_phase))
        return boundary_index

    def set_atol(self, primary=None, secondary=None):
        """
        Set atol for closeness to 1 in detecting eclipse boundaries.
        """
        if primary is not None:
            self.params["atol_primary"] = primary
        if secondary is not None:
            self.params["atol_secondary"] = secondary
        return 0

    def get_atol(self, min_flux):
        """
        Get atol for closeness to 1 in detecting eclipse boundaries.
        """
        proximity_to_one = 1 - min_flux
        if min_flux == min(self.data["fluxes"]):
            if self.params["atol_primary"] is None:
                return proximity_to_one * 0.05
            return self.params["atol_primary"]
        if self.params["atol_secondary"] is None:
            return proximity_to_one * 0.05
        return self.params["atol_secondary"]

    def calculate_eclipse_bins_distribution(self):
        """
        Calculates the number of bins to allocate to the primary and secondary eclipses.

        Returns:
            tuple: Number of bins in the primary eclipse, number of bins in the secondary eclipse.
        """
        bins_in_primary = int(
            (self.params["nbins"] * self.params["fraction_in_eclipse"]) / 2
        )
        start_idx, end_idx = np.searchsorted(self.data["phases"], self.primary_eclipse)
        eclipse_phases = self.data["phases"][start_idx : end_idx + 1]
        bins_in_primary = min(bins_in_primary, len(np.unique(eclipse_phases)))

        bins_in_secondary = int(
            (self.params["nbins"] * self.params["fraction_in_eclipse"])
            - bins_in_primary
        )
        start_idx, end_idx = np.searchsorted(
            self.data["phases"], self.secondary_eclipse
        )
        eclipse_phases = self.data["phases"][start_idx : end_idx + 1]
        bins_in_secondary = min(bins_in_secondary, len(np.unique(eclipse_phases)))

        return bins_in_primary, bins_in_secondary

    def find_bin_edges(self):
        """
        Finds the bin edges within the light curve.
        """

        bins_in_primary, bins_in_secondary = self.calculate_eclipse_bins_distribution()

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

        # Check for duplicate edges
        if len(np.unique(all_bins)) != len(all_bins):
            if self.params["fraction_in_eclipse"] > 0.1:
                new_fraction_in_eclipse = self.params["fraction_in_eclipse"] - 0.1
                print(
                    f"Binning resulted in repeat edges; trying again with "
                    f"fraction_in_eclipse={new_fraction_in_eclipse}"
                )
                self.params["fraction_in_eclipse"] = new_fraction_in_eclipse
                return self.find_bin_edges()
            raise ValueError(
                "There may not be enough data to bin these eclipses. Try "
                "changing the atol values for detecting eclipse boundaries with set_atol()."
            )

        # Check if we have significantly different number of bins than requested
        # Allow small differences (< 2%) due to duplicates='drop' rounding
        bin_count_diff = abs(len(all_bins) - self.params["nbins"])
        if bin_count_diff > max(1, 0.02 * self.params["nbins"]):
            if self.params["fraction_in_eclipse"] > 0.1:
                new_fraction_in_eclipse = self.params["fraction_in_eclipse"] - 0.1
                print(
                    f"Requested {self.params['nbins']} bins but got {len(all_bins)} "
                    f"due to data distribution; trying fraction_in_eclipse={new_fraction_in_eclipse}"
                )
                self.params["fraction_in_eclipse"] = new_fraction_in_eclipse
                return self.find_bin_edges()
            raise ValueError(
                "Cannot create the requested number of bins. Try "
                "reducing nbins or changing the atol values for detecting eclipse boundaries."
            )

        return all_bins

    def _rewrap_to_original_phase(self, phases_array):
        """
        Rewrap phases back to original phase space before unwrapping,
        then denormalize if original input had non-standard range.

        Args:
            phases_array (np.ndarray): Array of phases in unwrapped [0, 1] space

        Returns:
            np.ndarray: Phases shifted back to original space
        """
        result = phases_array
        if self._phase_shift != 0.0:
            result = (result - self._phase_shift) % 1.0
        # Denormalize back to original range (e.g., [-0.5, 0.5])
        return self._denormalize_phases(result)

    def calculate_bins(self, return_in_original_phase=True):
        """
        Calculates the bin centers, means, and standard deviations for the binned light curve.

        Args:
            return_in_original_phase (bool): If True, return results in original phase space
                (before unwrapping). If False, return in unwrapped space. Defaults to True.

        Returns:
            tuple: Arrays of bin centers, bin means, bin standard deviations, bin numbers,
                and bin edges.
        """
        all_bins = self.find_bin_edges()

        # Add phase 0 and 1 as boundaries for binned_statistic
        bin_edges = np.concatenate([[0], all_bins, [1]])

        # Ensure no duplicate edges (can occur at region boundaries even with duplicates='drop')
        bin_edges = np.unique(bin_edges)

        bin_means, _, bin_number = stats.binned_statistic(
            self.data["phases"],
            self.data["fluxes"],
            statistic="mean",
            bins=bin_edges,
        )
        bin_centers = (bin_edges[1:] - bin_edges[:-1]) / 2 + bin_edges[:-1]
        bin_errors = np.zeros(len(bin_means))

        # Calculate the propagated errors for each bin
        bincounts = np.bincount(bin_number, minlength=len(bin_edges))[1:]
        for i in range(len(bin_means)):
            # Get the indices of the data points in this bin
            bin_mask = (self.data["phases"] >= bin_edges[i]) & (
                self.data["phases"] < bin_edges[i + 1]
            )
            # Get the errors for these data points
            flux_errors_in_bin = self.data["flux_errors"][bin_mask]
            if len(flux_errors_in_bin) != bincounts[i]:
                raise ValueError("Incorrect bin masking.")
            # Calculate the propagated error for the bin
            n = bincounts[i]
            if n > 0:
                bin_errors[i] = np.sqrt(np.sum(flux_errors_in_bin**2)) / n

        if np.any(bincounts <= 0) or np.any(bin_errors <= 0):
            # Only retry if we have room to reduce fraction_in_eclipse
            if self.params["fraction_in_eclipse"] > 0.1:
                new_fraction_in_eclipse = self.params["fraction_in_eclipse"] - 0.1
                print(
                    f"Requested fraction of bins in eclipse regions results in empty bins; "
                    f"trying fraction_in_eclipse={new_fraction_in_eclipse}"
                )
                self.params["fraction_in_eclipse"] = new_fraction_in_eclipse
                return self.calculate_bins(
                    return_in_original_phase=return_in_original_phase
                )
            # If we can't reduce further, this combination of parameters is invalid
            raise ValueError(
                "Not enough data to bin these eclipses with the requested parameters. "
                "Try reducing nbins or increasing fraction_in_eclipse."
            )

        # Rewrap to original phase space if requested
        if return_in_original_phase:
            bin_centers = self._rewrap_to_original_phase(bin_centers)
            bin_edges = self._rewrap_to_original_phase(bin_edges)

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

        # Since phases are now unwrapped, we can directly slice
        eclipse_phases = self.data["phases"][start_idx : end_idx + 1]

        # Ensure there are enough unique phases for the number of bins requested
        if len(np.unique(eclipse_phases)) < bins_in_eclipse:
            raise ValueError(
                "Not enough unique phase values to create the requested number of bins."
            )

        bins = pd.qcut(eclipse_phases, q=bins_in_eclipse, duplicates="drop")
        return np.array([interval.right for interval in np.unique(bins)])

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

        # OOE1: between end of secondary eclipse and start of primary eclipse
        end_idx_secondary_eclipse = np.searchsorted(
            self.data["phases"], self.secondary_eclipse[1]
        )
        start_idx_primary_eclipse = np.searchsorted(
            self.data["phases"], self.primary_eclipse[0]
        )

        # Eclipses are unwrapped, but OOE regions may still wrap
        if end_idx_secondary_eclipse <= start_idx_primary_eclipse:
            # No wrapping in OOE1
            ooe1_phases = self.data["phases"][
                end_idx_secondary_eclipse : start_idx_primary_eclipse + 1
            ]
        else:
            # OOE1 wraps around
            ooe1_phases = np.concatenate(
                (
                    self.data["phases"][end_idx_secondary_eclipse:],
                    self.data["phases"][: start_idx_primary_eclipse + 1] + 1,
                )
            )

        ooe1_bins = pd.qcut(ooe1_phases, q=bins_in_ooe1, duplicates="drop")
        ooe1_edges = np.array([interval.right for interval in np.unique(ooe1_bins)]) % 1

        # OOE2: between end of primary eclipse and start of secondary eclipse
        end_idx_primary_eclipse = np.searchsorted(
            self.data["phases"], self.primary_eclipse[1]
        )
        start_idx_secondary_eclipse = np.searchsorted(
            self.data["phases"], self.secondary_eclipse[0]
        )

        if end_idx_primary_eclipse <= start_idx_secondary_eclipse:
            # No wrapping in OOE2
            ooe2_phases = self.data["phases"][
                end_idx_primary_eclipse : start_idx_secondary_eclipse + 1
            ]
        else:
            # OOE2 wraps around
            ooe2_phases = np.concatenate(
                (
                    self.data["phases"][end_idx_primary_eclipse:],
                    self.data["phases"][: start_idx_secondary_eclipse + 1] + 1,
                )
            )

        ooe2_bins = pd.qcut(ooe2_phases, q=bins_in_ooe2, duplicates="drop")
        ooe2_edges = np.array([interval.right for interval in np.unique(ooe2_bins)]) % 1

        return ooe1_edges, ooe2_edges

    def plot_binned_light_curve(self, bin_centers, bin_means, bin_stds):
        """
        Plots the binned light curve and the bin edges.

        Args:
            bin_centers (np.ndarray): Array of bin centers (in original phase space).
            bin_means (np.ndarray): Array of bin means.
            bin_stds (np.ndarray): Array of bin standard deviations.
        """
        plt.figure(figsize=(20, 5))
        plt.title("Binned Light Curve")
        plt.errorbar(
            bin_centers, bin_means, yerr=bin_stds, linestyle="none", marker="."
        )
        plt.xlabel("Phases", fontsize=14)
        plt.ylabel("Normalized Flux", fontsize=14)
        if self._needs_denormalization:
            plt.xlim(self._original_phase_min, self._original_phase_max)
        else:
            plt.xlim(0, 1)
        ylims = plt.ylim()

        # Get eclipse boundaries in original phase space
        primary_bounds = self._rewrap_to_original_phase(np.array(self.primary_eclipse))
        secondary_bounds = self._rewrap_to_original_phase(
            np.array(self.secondary_eclipse)
        )

        plt.vlines(
            primary_bounds,
            ymin=ylims[0],
            ymax=ylims[1],
            linestyle="--",
            color="red",
            label="Primary Eclipse",
        )
        plt.vlines(
            secondary_bounds,
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

        # Get data in original phase space for plotting
        original_phases = self._rewrap_to_original_phase(self.data["phases"])

        plt.errorbar(
            original_phases,
            self.data["fluxes"],
            yerr=self.data["flux_errors"],
            linestyle="none",
            marker=".",
        )
        ylims = plt.ylim()

        # Get eclipse boundaries in original phase space
        primary_bounds = self._rewrap_to_original_phase(np.array(self.primary_eclipse))
        secondary_bounds = self._rewrap_to_original_phase(
            np.array(self.secondary_eclipse)
        )

        plt.vlines(
            primary_bounds,
            ymin=ylims[0],
            ymax=ylims[1],
            linestyle="--",
            color="red",
            label="Primary Eclipse",
        )
        plt.vlines(
            secondary_bounds,
            ymin=ylims[0],
            ymax=ylims[1],
            linestyle="--",
            color="blue",
            label="Secondary Eclipse",
        )
        plt.ylim(ylims)
        if self._needs_denormalization:
            plt.xlim(self._original_phase_min, self._original_phase_max)
        else:
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
