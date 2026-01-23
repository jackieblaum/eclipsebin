# Edge Detection Method for Eclipse Boundaries

## Overview

The edge detection method identifies eclipse boundaries using slope/derivative analysis rather than absolute flux levels. This makes it robust to **ellipsoidal variations** and other systematic baseline changes.

## When to Use Edge Detection

Use `boundary_method='edge_detection'` when:

- Your system has significant ellipsoidal variations (tidally deformed stars)
- Out-of-eclipse baseline is not constant
- Traditional flux_return method misidentifies boundaries
- You want more robust boundary detection

Use `boundary_method='flux_return'` (default) when:

- Clean light curve with flat baseline
- You want faster computation
- System is well-separated (minimal tidal effects)

## Basic Usage

```python
from eclipsebin.binning import EclipsingBinaryBinner
import numpy as np

# Your light curve data
phases = np.array([...])  # Phase values in [0, 1)
fluxes = np.array([...])  # Normalized flux
flux_errors = np.array([...])  # Uncertainties

# Use edge detection
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection'
)

# Access eclipse boundaries
print(f"Primary: {binner.primary_eclipse}")
print(f"Secondary: {binner.secondary_eclipse}")
```

## Parameters

### Core Parameters

- **edge_slope_threshold_percentile** (default: 90.0)
  - Controls sensitivity to steep slopes
  - Higher = more selective (only very steep slopes count as edges)
  - Lower = more sensitive (detects gentler slopes)
  - **Adjust if**: Too many/few eclipses detected
  - **Typical range**: 80-95

- **edge_return_threshold_fraction** (default: 0.1)
  - Fraction of max slope for boundary definition
  - Lower = boundaries closer to eclipse center
  - Higher = boundaries farther from center
  - **Adjust if**: Eclipse boundaries seem too narrow/wide
  - **Typical range**: 0.05-0.2

- **edge_min_eclipse_depth** (default: 0.01)
  - Minimum flux drop to consider as eclipse
  - Filters out noise and minor dips
  - **Adjust if**: Missing shallow eclipses or detecting noise
  - **Typical range**: 0.005-0.05

- **edge_smoothing_window** (default: None = auto)
  - Smoothing window size (must be odd)
  - Auto-selects as ~5% of data points
  - **Adjust if**: Very noisy or very clean data
  - **Typical range**: 5-101 (odd numbers only)

- **min_eclipse_separation** (default: 0.2)
  - Minimum phase separation between primary and secondary
  - **Adjust if**: Eccentric orbit or unusual eclipse spacing
  - **Typical range**: 0.1-0.4

## Tuning Guide

### Problem: No Eclipses Detected

**Symptoms**: Warning "Edge detection found no eclipses"

**Solutions**:
1. Lower `edge_slope_threshold_percentile` (try 80 or 85)
2. Lower `edge_min_eclipse_depth` (try 0.005)
3. Check data quality and phase folding

```python
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection',
    edge_slope_threshold_percentile=85,  # More sensitive
    edge_min_eclipse_depth=0.005  # Detect shallower eclipses
)
```

### Problem: Secondary Eclipse Not Found

**Symptoms**: Warning "did not find secondary eclipse"

**Solutions**:
1. Secondary may be very shallow - lower `edge_min_eclipse_depth`
2. Check `min_eclipse_separation` - may be too large for your system
3. Inspect diagnostics to see if secondary was detected but not classified

```python
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection',
    edge_min_eclipse_depth=0.01,
    min_eclipse_separation=0.15  # Allow closer eclipses
)

# Check diagnostics
print(f"Detected {binner._edge_diagnostics['detected_count']} eclipses")
```

### Problem: Eclipse Boundaries Too Wide/Narrow

**Symptoms**: Visual inspection shows poor boundary placement

**Solutions**:
1. Adjust `edge_return_threshold_fraction`
   - Increase (e.g., 0.15) for wider boundaries
   - Decrease (e.g., 0.05) for narrower boundaries
2. Check smoothing - may be over/under-smoothing

```python
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection',
    edge_return_threshold_fraction=0.15  # Wider boundaries
)
```

### Problem: Too Noisy

**Symptoms**: False detections, unstable boundaries

**Solutions**:
1. Increase smoothing window
2. Increase `edge_slope_threshold_percentile`
3. Increase `edge_min_eclipse_depth`

```python
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection',
    edge_smoothing_window=51,  # More smoothing
    edge_slope_threshold_percentile=95  # Stricter
)
```

## Diagnostics

Access diagnostic information after binning:

```python
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection'
)

# Access diagnostics
diag = binner._edge_diagnostics

print(f"Slope threshold: {diag['threshold']:.3e}")
print(f"Return threshold: {diag['return_threshold']:.3e}")
print(f"Detected eclipses: {diag['detected_count']}")
print(f"Ingress candidates: {len(diag['ingress_candidates'])}")
print(f"Egress candidates: {len(diag['egress_candidates'])}")
print(f"Smoothing window: {diag['smoothing_window']}")

# Plot smoothed curve
import matplotlib.pyplot as plt
plt.plot(phases, fluxes, 'k.', alpha=0.3, label='Raw')
plt.plot(phases, diag['smoothed_fluxes'], 'r-', label='Smoothed')
plt.legend()
plt.show()
```

## Algorithm Details

### How It Works

1. **Smoothing**: Apply Savitzky-Golay filter to reduce noise
2. **Slope Calculation**: Compute derivatives using finite differences
3. **Threshold Detection**: Find steep negative (ingress) and positive (egress) slopes
4. **Pairing**: Match ingress-egress pairs as eclipse candidates
5. **Validation**: Check depth requirements and reject shallow dips
6. **Refinement**: Adjust boundaries to where slope returns to baseline
7. **Classification**: Identify primary (deeper) and secondary eclipses

### Advantages Over Flux Return

- **Robust to ellipsoidal variations**: Doesn't assume flat baseline
- **Local behavior**: Uses slope, not absolute flux
- **Adaptive**: Thresholds based on data statistics
- **Handles noise**: Smoothing reduces false positives

### Limitations

- **Requires clear ingress/egress**: Grazing eclipses may be challenging
- **More parameters**: More tuning knobs than flux_return
- **Computationally slower**: Smoothing and derivative calculations add overhead

## Examples

See `examples/edge_detection_example.py` for complete working examples.

## References

- Original flux_return method: Based on Gaia eclipsing binary pipeline
- Edge detection concept: Inspired by change-point detection in signal processing
- Ellipsoidal variations: See Wilson-Devinney model and tidally distorted stars
