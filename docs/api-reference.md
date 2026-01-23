# API Reference

## EclipsingBinaryBinner

The main class for binning eclipsing binary light curves.

### Constructor

```python
EclipsingBinaryBinner(
    phases,
    fluxes,
    flux_errors,
    nbins=200,
    fraction_in_eclipse=0.2,
    atol_primary=None,
    atol_secondary=None,
    boundary_method='flux_return',
    edge_slope_threshold_percentile=90.0,
    edge_return_threshold_fraction=0.1,
    edge_min_eclipse_depth=0.01,
    edge_smoothing_window=None,
    min_eclipse_separation=0.2
)
```

### Parameters

#### Required Parameters

- **phases** : array-like
  - Phase values in the range [0, 1)
  - Must have at least 10 data points

- **fluxes** : array-like
  - Normalized flux values
  - Should ideally have out-of-eclipse baseline near 1.0

- **flux_errors** : array-like
  - Flux uncertainties/errors
  - Must be same length as phases and fluxes

#### Basic Parameters

- **nbins** : int, default=200
  - Total number of bins to create
  - Must be at least 10
  - More bins = higher resolution but requires more data

- **fraction_in_eclipse** : float, default=0.2
  - Fraction of total bins allocated to eclipse regions
  - Range: [0, 1]
  - 0.2 means 20% of bins in eclipses, 80% out-of-eclipse
  - Higher values give more resolution in eclipses

#### Eclipse Boundary Detection Parameters

- **boundary_method** : str, default='flux_return'
  - Method for detecting eclipse boundaries
  - Options:
    - `'flux_return'`: Traditional method, finds where flux returns to baseline
    - `'edge_detection'`: Slope-based method, robust to ellipsoidal variations
  - See [Edge Detection Guide](edge_detection_guide.md) for detailed comparison

- **atol_primary** : float or None, default=None
  - Absolute tolerance for primary eclipse boundary detection (flux_return method)
  - If None, automatically calculated as `0.05 * primary_depth`
  - Lower values = stricter boundary definition

- **atol_secondary** : float or None, default=None
  - Absolute tolerance for secondary eclipse boundary detection (flux_return method)
  - If None, automatically calculated as `0.05 * secondary_depth`
  - Lower values = stricter boundary definition

- **min_eclipse_separation** : float, default=0.2
  - Minimum phase separation between primary and secondary eclipses
  - Range: typically 0.1 to 0.4
  - Adjust for eccentric orbits or unusual eclipse spacing

#### Edge Detection Parameters

These parameters only apply when `boundary_method='edge_detection'`. See [Edge Detection Guide](edge_detection_guide.md) for detailed usage.

- **edge_slope_threshold_percentile** : float, default=90.0
  - Percentile of absolute slopes to use as threshold for edge detection
  - Higher = more selective (only very steep slopes count as edges)
  - Lower = more sensitive (detects gentler slopes)
  - **Typical range**: 80-95

- **edge_return_threshold_fraction** : float, default=0.1
  - Fraction of max slope for boundary refinement
  - Controls where eclipse boundaries are placed relative to ingress/egress
  - Lower = boundaries closer to eclipse center
  - Higher = boundaries farther from center
  - **Typical range**: 0.05-0.2

- **edge_min_eclipse_depth** : float, default=0.01
  - Minimum flux drop required to classify as eclipse
  - Filters out noise and minor flux variations
  - Lower = detect shallower eclipses
  - Higher = more strict eclipse definition
  - **Typical range**: 0.005-0.05

- **edge_smoothing_window** : int or None, default=None
  - Size of smoothing window for Savitzky-Golay filter (must be odd)
  - If None, automatically set to ~5% of data points
  - Larger values = more smoothing (reduce noise but may blur features)
  - **Typical range**: 5-101 (odd numbers only)

### Attributes

After initialization, the binner object has these key attributes:

- **primary_eclipse** : tuple of (float, float) or None
  - Phase boundaries of primary eclipse (start, end)
  - None if no primary eclipse detected

- **secondary_eclipse** : tuple of (float, float) or None
  - Phase boundaries of secondary eclipse (start, end)
  - None if no secondary eclipse detected

- **_edge_diagnostics** : dict (when using edge_detection)
  - Diagnostic information from edge detection algorithm
  - Keys include:
    - `'threshold'`: Slope threshold used
    - `'return_threshold'`: Boundary refinement threshold
    - `'smoothing_window'`: Window size used for smoothing
    - `'detected_count'`: Number of eclipse candidates found
    - `'ingress_candidates'`: List of ingress candidate indices
    - `'egress_candidates'`: List of egress candidate indices
    - `'smoothed_fluxes'`: Smoothed flux array
    - `'slopes'`: Computed slope array

### Methods

#### bin_light_curve()

```python
bin_light_curve(plot=False)
```

Bins the light curve data.

**Parameters:**
- **plot** : bool, default=False
  - If True, plots the binned and unbinned light curves

**Returns:**
- **bin_centers** : numpy.ndarray
  - Phase values at bin centers
- **bin_means** : numpy.ndarray
  - Mean flux in each bin
- **bin_stds** : numpy.ndarray
  - Standard deviation of flux in each bin (propagated errors)

#### find_minimum_flux()

```python
find_minimum_flux()
```

Returns the minimum flux value in the primary eclipse.

**Returns:**
- **min_flux** : float
  - Minimum flux value

#### find_secondary_minimum()

```python
find_secondary_minimum()
```

Returns the minimum flux value in the secondary eclipse.

**Returns:**
- **min_flux** : float
  - Minimum flux value in secondary eclipse

#### plot_unbinned_light_curve()

```python
plot_unbinned_light_curve()
```

Plots the unbinned light curve with eclipse boundaries marked.

Creates a matplotlib figure showing the raw phase-folded light curve with vertical lines indicating the detected eclipse boundaries.

### Examples

#### Basic Usage

```python
import numpy as np
from eclipsebin.binning import EclipsingBinaryBinner

# Your light curve data
phases = np.array([...])
fluxes = np.array([...])
flux_errors = np.array([...])

# Create binner
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    fraction_in_eclipse=0.2
)

# Bin the light curve
bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=True)

# Access eclipse information
print(f"Primary eclipse: {binner.primary_eclipse}")
print(f"Secondary eclipse: {binner.secondary_eclipse}")
```

#### Using Edge Detection

```python
# For systems with ellipsoidal variations
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection',
    edge_slope_threshold_percentile=90,
    edge_min_eclipse_depth=0.01
)

# Bin and check diagnostics
bin_centers, bin_means, bin_stds = binner.bin_light_curve()
print(f"Detected {binner._edge_diagnostics['detected_count']} eclipses")
print(f"Slope threshold: {binner._edge_diagnostics['threshold']:.3e}")
```

See [Edge Detection Guide](edge_detection_guide.md) for comprehensive examples and tuning instructions.

### Notes

- The binner automatically handles phase-wrapped eclipses by internally unwrapping phases during detection and binning, then rewrapping results for output.
- If edge detection fails to find eclipses, the binner will automatically fall back to the flux_return method.
- At least 5x as many data points as bins are recommended for stable binning.
- The fraction_in_eclipse may be automatically reduced if the requested number of bins cannot be achieved.

### See Also

- [Edge Detection Guide](edge_detection_guide.md) - Comprehensive guide to using edge detection method
- [Getting Started](getting-started.md) - Tutorial for first-time users
- [Example Scripts](../examples/) - Working example code
