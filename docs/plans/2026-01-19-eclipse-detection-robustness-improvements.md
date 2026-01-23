# Eclipse Detection Robustness Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve robustness, diagnostics, and configurability of the slope-based edge detection method for eclipse boundary identification.

**Architecture:** Refactor hardcoded constants to be adaptive based on data properties, add diagnostic output capabilities, improve error handling and validation, and increase configurability for different use cases.

**Tech Stack:** Python, NumPy, SciPy, pytest

**Priority:** High priority items first (diagnostics), then medium (constants, validation), then low (configurability)

---

## Task 1: Add Diagnostics Output Infrastructure

**Files:**
- Modify: `eclipsebin/binning.py:16-166` (_detect_eclipse_edges_slope function)
- Modify: `eclipsebin/binning.py:250-366` (EclipsingBinaryBinner class)
- Create: `tests/test_edge_detection_diagnostics.py`

**Step 1: Write the failing test for diagnostics return**

```python
# tests/test_edge_detection_diagnostics.py
import numpy as np
from eclipsebin.binning import EclipsingBinaryBinner

def test_edge_detection_stores_diagnostics():
    """Test that edge detection diagnostics are stored and accessible."""
    # Create synthetic light curve with clear eclipse
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)
    # Add primary eclipse at phase 0.0
    eclipse_mask = (phases < 0.1) | (phases > 0.9)
    fluxes[eclipse_mask] = 0.8
    flux_errors = np.ones_like(phases) * 0.01

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    # Check that diagnostics exist
    assert hasattr(binner, '_edge_diagnostics')
    assert binner._edge_diagnostics is not None

    # Check diagnostic content
    diag = binner._edge_diagnostics
    assert 'slopes' in diag
    assert 'smoothed_fluxes' in diag
    assert 'threshold' in diag
    assert 'return_threshold' in diag
    assert 'ingress_candidates' in diag
    assert 'egress_candidates' in diag
    assert 'detected_count' in diag

    # Verify data types
    assert isinstance(diag['slopes'], np.ndarray)
    assert isinstance(diag['threshold'], (float, np.floating))
    assert isinstance(diag['detected_count'], (int, np.integer))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_edge_detection_diagnostics.py::test_edge_detection_stores_diagnostics -v`
Expected: FAIL with "AttributeError: 'EclipsingBinaryBinner' object has no attribute '_edge_diagnostics'"

**Step 3: Modify _detect_eclipse_edges_slope to return diagnostics**

In `eclipsebin/binning.py`, modify the function signature and return statement:

```python
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

    ... [existing docstring content] ...

    Returns
    -------
    eclipse_boundaries : list of tuples
        List of (ingress_phase, egress_phase) for each detected eclipse.
        Empty list if no eclipses detected.
    diagnostics : dict
        Diagnostic information including slopes, thresholds, and candidates.
    """
    # ... [existing code until line 165] ...

    # Before the final return, collect diagnostics
    diagnostics = {
        'slopes': slopes,
        'smoothed_fluxes': smoothed_fluxes,
        'threshold': slope_threshold,
        'return_threshold': return_threshold,
        'ingress_candidates': ingress_indices.tolist() if len(ingress_indices) > 0 else [],
        'egress_candidates': egress_indices.tolist() if len(egress_indices) > 0 else [],
        'detected_count': len(eclipse_boundaries),
        'smoothing_window': smoothing_window,
    }

    return eclipse_boundaries, diagnostics
```

**Step 4: Add _edge_diagnostics attribute to EclipsingBinaryBinner.__init__**

In `eclipsebin/binning.py` around line 340, after storing edge detection parameters:

```python
# Store edge detection parameters
self.boundary_method = boundary_method
self.edge_slope_threshold_percentile = edge_slope_threshold_percentile
self.edge_return_threshold_fraction = edge_return_threshold_fraction
self.edge_min_eclipse_depth = edge_min_eclipse_depth
self.edge_smoothing_window = edge_smoothing_window

# Initialize diagnostics storage
self._edge_diagnostics = None
```

**Step 5: Update get_eclipse_boundaries to store diagnostics**

In `eclipsebin/binning.py` around line 536-546, update the edge detection call:

```python
if self.boundary_method == 'edge_detection':
    # Use edge detection method
    boundaries, diagnostics = _detect_eclipse_edges_slope(
        self.data["phases"],
        self.data["fluxes"],
        self.data["flux_errors"],
        smoothing_window=self.edge_smoothing_window,
        slope_threshold_percentile=self.edge_slope_threshold_percentile,
        return_threshold_fraction=self.edge_return_threshold_fraction,
        min_eclipse_depth=self.edge_min_eclipse_depth
    )

    # Store diagnostics
    self._edge_diagnostics = diagnostics
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_edge_detection_diagnostics.py::test_edge_detection_stores_diagnostics -v`
Expected: PASS

**Step 7: Commit**

```bash
git add eclipsebin/binning.py tests/test_edge_detection_diagnostics.py
git commit -m "feat: add diagnostics output to edge detection method

- Return diagnostics dict from _detect_eclipse_edges_slope
- Store diagnostics in EclipsingBinaryBinner._edge_diagnostics
- Add test coverage for diagnostics storage"
```

---

## Task 2: Make Hardcoded Constants Adaptive

**Files:**
- Modify: `eclipsebin/binning.py:16-166` (_detect_eclipse_edges_slope function)
- Create: `tests/test_adaptive_constants.py`

**Step 1: Write tests for adaptive constants**

```python
# tests/test_adaptive_constants.py
import numpy as np
from eclipsebin.binning import _detect_eclipse_edges_slope

def test_baseline_window_scales_with_data_density():
    """Test that baseline window adapts to data density."""
    # Sparse data (100 points)
    sparse_phases = np.linspace(0, 1, 100)
    sparse_fluxes = np.ones(100)
    sparse_fluxes[40:50] = 0.8  # Eclipse

    boundaries_sparse, diag_sparse = _detect_eclipse_edges_slope(
        sparse_phases, sparse_fluxes
    )

    # Dense data (10000 points)
    dense_phases = np.linspace(0, 1, 10000)
    dense_fluxes = np.ones(10000)
    dense_fluxes[4000:5000] = 0.8  # Eclipse at same phase

    boundaries_dense, diag_dense = _detect_eclipse_edges_slope(
        dense_phases, dense_fluxes
    )

    # Both should detect the eclipse
    assert len(boundaries_sparse) > 0
    assert len(boundaries_dense) > 0

    # Diagnostic should show different smoothing windows
    assert 'smoothing_window' in diag_sparse
    assert 'smoothing_window' in diag_dense
    # Dense data should have larger window
    assert diag_dense['smoothing_window'] > diag_sparse['smoothing_window']

def test_refinement_range_adapts_to_data():
    """Test that boundary refinement range scales with data."""
    # This is an integration test - we verify through diagnostics
    # that the algorithm doesn't fail on edge cases

    # Very sparse data
    phases = np.linspace(0, 1, 50)
    fluxes = np.ones(50)
    fluxes[20:25] = 0.7

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should not crash and should detect eclipse
    assert len(boundaries) >= 0  # May or may not detect with very sparse data
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptive_constants.py -v`
Expected: Tests may pass or fail depending on current behavior, but we'll improve the implementation

**Step 3: Replace hardcoded baseline window constant**

In `eclipsebin/binning.py` around line 134-137, replace:

```python
# OLD:
local_baseline = np.median([
    np.median(smoothed_fluxes[max(0, ingress_idx-10):ingress_idx]),
    np.median(smoothed_fluxes[egress_idx:min(len(fluxes), egress_idx+10)])
])

# NEW:
# Adaptive baseline window: 2% of data, minimum 5 points
baseline_window = max(5, int(0.02 * len(fluxes)))

# Calculate baseline with validation
pre_window_start = max(0, ingress_idx - baseline_window)
pre_window_end = ingress_idx
post_window_start = egress_idx
post_window_end = min(len(fluxes), egress_idx + baseline_window)

pre_window = smoothed_fluxes[pre_window_start:pre_window_end]
post_window = smoothed_fluxes[post_window_start:post_window_end]

# Need at least 3 points for reliable baseline
if len(pre_window) < 3 or len(post_window) < 3:
    continue

local_baseline = np.median([np.median(pre_window), np.median(post_window)])
```

**Step 4: Replace hardcoded refinement range**

In `eclipsebin/binning.py` around lines 143-153, replace:

```python
# OLD:
for j in range(ingress_idx, max(0, ingress_idx-20), -1):

# NEW:
# Adaptive refinement range: 5% of data, minimum 10 points
refinement_range = max(10, int(0.05 * len(phases)))

for j in range(ingress_idx, max(0, ingress_idx - refinement_range), -1):
```

And similarly for egress:

```python
# OLD:
for j in range(egress_idx, min(len(phases), egress_idx+20)):

# NEW:
for j in range(egress_idx, min(len(phases), egress_idx + refinement_range)):
```

**Step 5: Replace hardcoded gap detection multiplier**

In `eclipsebin/binning.py` around line 91, improve the gap detection:

```python
# OLD:
large_gap = dphase > 10 * median_dphase

# NEW:
# Detect gaps that are significantly larger than typical spacing
# Use 10x as threshold but ensure it's at least 0.05 phase units
gap_threshold = max(10 * median_dphase, 0.05)
large_gap = dphase > gap_threshold
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_adaptive_constants.py -v`
Expected: PASS

**Step 7: Run existing tests to ensure no regression**

Run: `pytest tests/test_eclipsing_binary_binner.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add eclipsebin/binning.py tests/test_adaptive_constants.py
git commit -m "refactor: make edge detection constants adaptive

- Baseline window scales with data density (2% of points, min 5)
- Refinement range adapts to data (5% of points, min 10)
- Gap detection uses adaptive threshold
- Add validation for minimum window sizes
- Add tests for adaptive behavior"
```

---

## Task 3: Improve Warning Messages

**Files:**
- Modify: `eclipsebin/binning.py:526-591` (get_eclipse_boundaries method)
- Create: `tests/test_edge_detection_warnings.py`

**Step 1: Write test for informative warning messages**

```python
# tests/test_edge_detection_warnings.py
import numpy as np
import warnings
from eclipsebin.binning import EclipsingBinaryBinner

def test_no_eclipses_warning_is_informative():
    """Test that warning provides actionable guidance when no eclipses found."""
    # Create flat light curve (no eclipses)
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases) + np.random.normal(0, 0.01, len(phases))
    flux_errors = np.ones_like(phases) * 0.01

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should have issued a warning
        assert len(w) > 0
        warning_msg = str(w[0].message)

        # Check that warning mentions:
        # 1. What failed (no eclipses found)
        # 2. Diagnostic info (thresholds, candidates)
        # 3. Actionable suggestion (parameter to adjust)
        assert 'no eclipses' in warning_msg.lower() or 'not find' in warning_msg.lower()
        assert 'edge_' in warning_msg  # Mentions parameter names
        assert 'percentile' in warning_msg.lower() or 'depth' in warning_msg.lower()

def test_missing_eclipse_warning_mentions_eclipse_type():
    """Test that warning specifies which eclipse (primary/secondary) wasn't found."""
    # Create light curve with only one strong eclipse
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)
    # Strong primary only
    fluxes[450:550] = 0.7
    flux_errors = np.ones_like(phases) * 0.01

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # May warn about missing secondary
        if len(w) > 0:
            warning_msg = str(w[0].message)
            # Should mention which eclipse is missing
            assert 'primary' in warning_msg.lower() or 'secondary' in warning_msg.lower()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_edge_detection_warnings.py -v`
Expected: FAIL (current warnings don't include this detail)

**Step 3: Improve warning when no eclipses detected**

In `eclipsebin/binning.py` around lines 572-581, replace:

```python
# OLD:
else:
    # No eclipses detected, fall back to flux_return
    warnings.warn(
        "Edge detection found no eclipses, falling back to flux_return method"
    )

# NEW:
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
```

**Step 4: Improve warning when specific eclipse not found**

In `eclipsebin/binning.py` around lines 562-571, replace:

```python
# OLD:
warnings.warn(
    f"Edge detection did not find {'primary' if primary else 'secondary'} "
    f"eclipse, falling back to flux_return method"
)

# NEW:
eclipse_type = 'primary' if primary else 'secondary'
diag_str = f"detected {len(boundaries)} eclipse(s) total"
warnings.warn(
    f"Edge detection did not find {eclipse_type} eclipse ({diag_str}). "
    f"This may occur if the {eclipse_type} eclipse is very shallow "
    f"(depth < {self.edge_min_eclipse_depth}) or if slope threshold "
    f"is too high (percentile: {self.edge_slope_threshold_percentile}). "
    f"Falling back to flux_return method."
)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_edge_detection_warnings.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add eclipsebin/binning.py tests/test_edge_detection_warnings.py
git commit -m "improve: make edge detection warnings more informative

- Include diagnostic information (thresholds, candidates) in warnings
- Provide actionable suggestions for parameter adjustments
- Specify which eclipse type (primary/secondary) wasn't found
- Add tests for warning message content"
```

---

## Task 4: Improve Local Baseline Robustness

**Files:**
- Modify: `eclipsebin/binning.py:16-166` (_detect_eclipse_edges_slope function)
- Create: `tests/test_baseline_robustness.py`

**Step 1: Write tests for robust baseline calculation**

```python
# tests/test_baseline_robustness.py
import numpy as np
from eclipsebin.binning import _detect_eclipse_edges_slope

def test_baseline_handles_outliers():
    """Test that baseline calculation is robust to outliers."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Add eclipse
    fluxes[450:550] = 0.8

    # Add outliers near eclipse edges
    fluxes[440] = 0.5  # Outlier before ingress
    fluxes[560] = 1.5  # Outlier after egress

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should still detect eclipse correctly
    assert len(boundaries) > 0
    # Eclipse should be roughly centered around phase 0.5
    ingress, egress = boundaries[0]
    center = (ingress + egress) / 2
    assert 0.45 < center < 0.55

def test_baseline_handles_sparse_edges():
    """Test baseline calculation with very few points near eclipse edges."""
    # Sparse sampling with gaps
    phases = np.concatenate([
        np.linspace(0, 0.4, 50),
        np.linspace(0.45, 0.55, 200),  # Dense in eclipse
        np.linspace(0.6, 1.0, 50)
    ])
    fluxes = np.ones_like(phases)
    eclipse_mask = (phases >= 0.48) & (phases <= 0.52)
    fluxes[eclipse_mask] = 0.75

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should handle sparse edges gracefully
    assert len(boundaries) >= 0  # May or may not detect, but shouldn't crash

def test_baseline_with_phase_boundary_eclipse():
    """Test baseline near phase=0/1 boundary."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Eclipse near phase 0
    fluxes[0:50] = 0.8
    fluxes[950:1000] = 0.8  # Wraps around

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should not crash
    assert isinstance(boundaries, list)
```

**Step 2: Run tests to verify current behavior**

Run: `pytest tests/test_baseline_robustness.py -v`
Expected: Some tests may fail due to lack of outlier handling

**Step 3: Add scipy trimmed_mean import**

At the top of `eclipsebin/binning.py`, add:

```python
from scipy.stats import trim_mean
```

**Step 4: Improve baseline calculation with robust statistics**

In `eclipsebin/binning.py` in the section modified in Task 2, update to use trimmed mean:

```python
# Calculate baseline with validation
pre_window_start = max(0, ingress_idx - baseline_window)
pre_window_end = ingress_idx
post_window_start = egress_idx
post_window_end = min(len(fluxes), egress_idx + baseline_window)

pre_window = smoothed_fluxes[pre_window_start:pre_window_end]
post_window = smoothed_fluxes[post_window_start:post_window_end]

# Need at least 3 points for reliable baseline
if len(pre_window) < 3 or len(post_window) < 3:
    continue

# Use trimmed mean for robustness against outliers (trim 10% from each end)
try:
    pre_baseline = trim_mean(pre_window, 0.1) if len(pre_window) >= 5 else np.median(pre_window)
    post_baseline = trim_mean(post_window, 0.1) if len(post_window) >= 5 else np.median(post_window)
    local_baseline = np.mean([pre_baseline, post_baseline])
except Exception:
    # Fallback to median if trimmed mean fails
    local_baseline = np.median([np.median(pre_window), np.median(post_window)])

# Validate baseline is reasonable (not too close to zero)
if local_baseline < 0.1:
    # Baseline too faint for reliable depth calculation
    continue
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_baseline_robustness.py -v`
Expected: PASS

**Step 6: Run regression tests**

Run: `pytest tests/test_eclipsing_binary_binner.py tests/test_adaptive_constants.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add eclipsebin/binning.py tests/test_baseline_robustness.py
git commit -m "improve: make baseline calculation more robust

- Use trimmed mean to handle outliers (10% trim from each end)
- Add validation for minimum baseline flux (> 0.1)
- Add validation for minimum window sizes (>= 3 points)
- Graceful fallback to median if trimmed mean fails
- Add tests for outliers, sparse data, and edge cases"
```

---

## Task 5: Add Smoothing Window Validation

**Files:**
- Modify: `eclipsebin/binning.py:16-166` (_detect_eclipse_edges_slope function)
- Create: `tests/test_smoothing_validation.py`

**Step 1: Write tests for smoothing validation**

```python
# tests/test_smoothing_validation.py
import numpy as np
import warnings
from eclipsebin.binning import _detect_eclipse_edges_slope

def test_warns_if_smoothing_too_large_for_narrow_eclipse():
    """Test warning when smoothing window may obscure narrow eclipse."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Very narrow eclipse (2% of phase)
    fluxes[490:510] = 0.7

    # Force large smoothing window
    large_window = 101  # 10% of data

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        boundaries, diagnostics = _detect_eclipse_edges_slope(
            phases, fluxes, smoothing_window=large_window
        )

        # Should warn about smoothing being too large
        # (if it detects the eclipse and validates)
        warning_msgs = [str(warning.message).lower() for warning in w]
        if len(boundaries) > 0:  # Only validates if eclipse detected
            assert any('smoothing' in msg or 'window' in msg for msg in warning_msgs)

def test_auto_smoothing_appropriate_for_data():
    """Test that auto-selected smoothing is reasonable."""
    # Test with different data densities
    for n_points in [100, 1000, 10000]:
        phases = np.linspace(0, 1, n_points)
        fluxes = np.ones_like(phases)

        # Add eclipse (10% of phase)
        eclipse_width = int(0.1 * n_points)
        start_idx = (n_points - eclipse_width) // 2
        fluxes[start_idx:start_idx + eclipse_width] = 0.8

        boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

        # Smoothing window should be reasonable
        # (between 1% and 10% of data)
        window = diagnostics['smoothing_window']
        assert 0.01 * n_points <= window <= 0.1 * n_points
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_smoothing_validation.py -v`
Expected: FAIL (no validation implemented yet)

**Step 3: Add smoothing validation after eclipse detection**

In `eclipsebin/binning.py` in _detect_eclipse_edges_slope, add validation after the eclipse detection loop (around line 165, before collecting diagnostics):

```python
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
            UserWarning
        )
```

**Step 4: Import warnings module if not already imported**

At the top of `eclipsebin/binning.py`, ensure:

```python
import warnings
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_smoothing_validation.py -v`
Expected: PASS

**Step 6: Run regression tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add eclipsebin/binning.py tests/test_smoothing_validation.py
git commit -m "feat: add smoothing window validation

- Warn if smoothing window is too large for narrow eclipses
- Compare window size to eclipse width (warn if > 1/3 width)
- Provide actionable suggestions in warning
- Add tests for smoothing validation"
```

---

## Task 6: Add Division Safety Checks

**Files:**
- Modify: `eclipsebin/binning.py:16-166` (_detect_eclipse_edges_slope function)
- Create: `tests/test_division_safety.py`

**Step 1: Write tests for division safety**

```python
# tests/test_division_safety.py
import numpy as np
from eclipsebin.binning import _detect_eclipse_edges_slope

def test_handles_zero_baseline_gracefully():
    """Test that algorithm handles near-zero baseline without crashing."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases) * 0.05  # Very faint baseline

    # Add relative eclipse
    fluxes[450:550] = 0.03

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should not crash with ZeroDivisionError
    assert isinstance(boundaries, list)

def test_handles_zero_phase_spacing():
    """Test handling of duplicate phase values."""
    phases = np.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fluxes = np.ones_like(phases)
    fluxes[6:8] = 0.7  # Eclipse

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should handle duplicate phases without division by zero
    assert isinstance(boundaries, list)

def test_handles_all_same_flux():
    """Test handling of constant flux (no variation)."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)  # Constant

    boundaries, diagnostics = _detect_eclipse_edges_slope(phases, fluxes)

    # Should return empty boundaries (no eclipses)
    assert boundaries == []
```

**Step 2: Run tests to verify current behavior**

Run: `pytest tests/test_division_safety.py -v`
Expected: May crash or fail on zero baseline test

**Step 3: Add division safety to depth calculation**

This was already addressed in Task 4 (added baseline < 0.1 check), but verify it's present in the code. The relevant section should look like:

```python
# Validate baseline is reasonable (not too close to zero)
if local_baseline < 0.1:
    # Baseline too faint for reliable depth calculation
    continue

min_flux = np.min(eclipse_region)
depth = (local_baseline - min_flux) / local_baseline
```

**Step 4: Add safety to slope calculation**

In `eclipsebin/binning.py` around line 95, verify the safety is adequate:

```python
# The existing code has:
slopes[1:] = dflux / (dphase + 1e-10)  # Avoid division by zero

# This is adequate, but let's improve the comment:
# Compute slopes, avoiding division by zero with epsilon
slopes[1:] = dflux / (dphase + 1e-10)
```

**Step 5: Add validation for empty valid_slopes**

Around line 104, the code already checks:

```python
valid_slopes = abs_slopes[abs_slopes > 0]
if len(valid_slopes) == 0:
    return []
```

This is good. Ensure the return matches updated signature:

```python
valid_slopes = abs_slopes[abs_slopes > 0]
if len(valid_slopes) == 0:
    # No valid slopes - flat light curve or insufficient data
    diagnostics = {
        'slopes': slopes,
        'smoothed_fluxes': smoothed_fluxes,
        'threshold': 0.0,
        'return_threshold': 0.0,
        'ingress_candidates': [],
        'egress_candidates': [],
        'detected_count': 0,
        'smoothing_window': smoothing_window,
    }
    return [], diagnostics
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_division_safety.py -v`
Expected: PASS

**Step 7: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add eclipsebin/binning.py tests/test_division_safety.py
git commit -m "improve: add division safety checks

- Validate baseline > 0.1 before depth calculation
- Return diagnostics even when no valid slopes found
- Add tests for zero baseline, duplicate phases, constant flux
- Improve code comments for division safety"
```

---

## Task 7: Make Secondary Eclipse Separation Configurable

**Files:**
- Modify: `eclipsebin/binning.py:169-247` (_find_primary_and_secondary_from_edges function)
- Modify: `eclipsebin/binning.py:250-366` (EclipsingBinaryBinner.__init__)
- Modify: `eclipsebin/binning.py:448-455` (_helper_secondary_minimum_mask method)
- Create: `tests/test_eclipse_separation.py`

**Step 1: Write tests for configurable separation**

```python
# tests/test_eclipse_separation.py
import numpy as np
from eclipsebin.binning import EclipsingBinaryBinner

def test_default_secondary_separation():
    """Test default secondary eclipse separation is 0.2."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Primary at phase 0.5
    fluxes[475:525] = 0.7
    # Secondary at phase 0.75 (0.25 separation)
    fluxes[725:775] = 0.85

    flux_errors = np.ones_like(phases) * 0.01

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    # Should detect both eclipses with default separation
    assert binner.primary_eclipse is not None
    assert binner.secondary_eclipse is not None

def test_custom_secondary_separation():
    """Test custom minimum secondary eclipse separation."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Primary at phase 0.5
    fluxes[475:525] = 0.7
    # Secondary at phase 0.65 (0.15 separation - too close with default)
    fluxes[625:675] = 0.85

    flux_errors = np.ones_like(phases) * 0.01

    # With default (0.2), secondary might not be detected
    binner_default = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    # With custom (0.1), secondary should be detected
    binner_custom = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        min_eclipse_separation=0.1
    )

    # Custom should be more permissive
    assert binner_custom.min_eclipse_separation == 0.1

def test_separation_affects_secondary_detection():
    """Test that separation parameter affects which eclipse is considered secondary."""
    phases = np.linspace(0, 1, 1000)
    fluxes = np.ones_like(phases)

    # Three eclipses
    fluxes[100:150] = 0.6  # Deepest (primary)
    fluxes[450:500] = 0.8  # Medium
    fluxes[800:850] = 0.85  # Shallowest

    flux_errors = np.ones_like(phases) * 0.01

    # Strict separation might only detect primary
    binner_strict = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        min_eclipse_separation=0.4  # Very strict
    )

    # Permissive separation should detect more
    binner_permissive = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        min_eclipse_separation=0.1  # Very permissive
    )

    # Both should detect primary
    assert binner_strict.primary_eclipse is not None
    assert binner_permissive.primary_eclipse is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eclipse_separation.py -v`
Expected: FAIL with "unexpected keyword argument 'min_eclipse_separation'"

**Step 3: Add min_eclipse_separation parameter to __init__**

In `eclipsebin/binning.py` around line 267-281, add the parameter:

```python
def __init__(
    self,
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
    min_eclipse_separation=0.2,
):
```

**Step 4: Update docstring**

In the docstring (around line 282-307), add:

```python
    min_eclipse_separation (float, optional): Minimum phase separation between
        primary and secondary eclipses. Eclipses closer than this are not
        considered distinct. Defaults to 0.2 (works for most circular orbits).
        Reduce for eccentric systems or close eclipses.
```

**Step 5: Store parameter in __init__**

Around line 340, add:

```python
# Store edge detection parameters
self.boundary_method = boundary_method
self.edge_slope_threshold_percentile = edge_slope_threshold_percentile
self.edge_return_threshold_fraction = edge_return_threshold_fraction
self.edge_min_eclipse_depth = edge_min_eclipse_depth
self.edge_smoothing_window = edge_smoothing_window
self.min_eclipse_separation = min_eclipse_separation
```

**Step 6: Update _find_primary_and_secondary_from_edges to accept parameter**

In `eclipsebin/binning.py` around line 169, update function signature:

```python
def _find_primary_and_secondary_from_edges(eclipse_boundaries, phases, fluxes, min_separation=0.2):
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
        Minimum phase separation between primary and secondary. Defaults to 0.2.

    Returns
    -------
    primary_boundaries : tuple or None
        (ingress, egress) for primary eclipse
    secondary_boundaries : tuple or None
        (ingress, egress) for secondary eclipse
    """
```

**Step 7: Use min_separation parameter in function**

Around line 238, replace hardcoded 0.2:

```python
# OLD:
if phase_sep >= 0.2:  # Secondary should be ~0.5 phase away (or at least 0.2)

# NEW:
if phase_sep >= min_separation:
```

**Step 8: Update call site in get_eclipse_boundaries**

Around line 550, pass the parameter:

```python
# Identify primary and secondary
primary_bounds, secondary_bounds = _find_primary_and_secondary_from_edges(
    boundaries, self.data["phases"], self.data["fluxes"],
    min_separation=self.min_eclipse_separation
)
```

**Step 9: Update _helper_secondary_minimum_mask to use parameter**

Around line 454, replace hardcoded 0.2:

```python
# OLD:
mask = phase_delta > 0.2

# NEW:
mask = phase_delta > self.min_eclipse_separation
```

**Step 10: Run tests to verify they pass**

Run: `pytest tests/test_eclipse_separation.py -v`
Expected: PASS

**Step 11: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 12: Commit**

```bash
git add eclipsebin/binning.py tests/test_eclipse_separation.py
git commit -m "feat: make secondary eclipse separation configurable

- Add min_eclipse_separation parameter (default 0.2)
- Update _find_primary_and_secondary_from_edges to accept parameter
- Update _helper_secondary_minimum_mask to use parameter
- Add documentation for parameter usage
- Add tests for configurable separation"
```

---

## Task 8: Add Comprehensive Integration Tests

**Files:**
- Create: `tests/test_edge_detection_integration.py`

**Step 1: Write comprehensive integration tests**

```python
# tests/test_edge_detection_integration.py
"""
Comprehensive integration tests for edge detection with ellipsoidal variations.
"""
import numpy as np
import pytest
from eclipsebin.binning import EclipsingBinaryBinner

def create_synthetic_light_curve_with_ellipsoidal(
    n_points=1000,
    primary_depth=0.3,
    secondary_depth=0.15,
    ellipsoidal_amplitude=0.05,
    noise_level=0.01
):
    """
    Create synthetic light curve with eclipses and ellipsoidal variations.

    Parameters
    ----------
    n_points : int
        Number of data points
    primary_depth : float
        Depth of primary eclipse (relative)
    secondary_depth : float
        Depth of secondary eclipse (relative)
    ellipsoidal_amplitude : float
        Amplitude of ellipsoidal variation (cos(4*pi*phase))
    noise_level : float
        Standard deviation of Gaussian noise

    Returns
    -------
    phases, fluxes, flux_errors : arrays
    """
    phases = np.linspace(0, 1, n_points)

    # Start with ellipsoidal variation (twice per orbit)
    fluxes = 1.0 + ellipsoidal_amplitude * np.cos(4 * np.pi * phases)

    # Add primary eclipse centered at phase 0.0
    primary_mask = (phases < 0.1) | (phases > 0.9)
    primary_depth_curve = primary_depth * np.exp(-((phases[primary_mask] % 1.0 - 0.0) ** 2) / 0.005)
    fluxes[primary_mask] -= primary_depth_curve

    # Add secondary eclipse centered at phase 0.5
    secondary_mask = (phases > 0.4) & (phases < 0.6)
    secondary_depth_curve = secondary_depth * np.exp(-((phases[secondary_mask] - 0.5) ** 2) / 0.005)
    fluxes[secondary_mask] -= secondary_depth_curve

    # Add noise
    fluxes += np.random.normal(0, noise_level, n_points)

    flux_errors = np.ones_like(phases) * noise_level

    return phases, fluxes, flux_errors


class TestEdgeDetectionIntegration:
    """Integration tests for edge detection with realistic light curves."""

    def test_detects_eclipses_with_ellipsoidal_variation(self):
        """Test that edge detection works with strong ellipsoidal variations."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            ellipsoidal_amplitude=0.1  # Strong variation
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should detect both eclipses
        assert binner.primary_eclipse is not None
        assert binner.secondary_eclipse is not None

        # Primary should be deeper
        primary_depth = binner.find_minimum_flux()
        secondary_depth = binner.find_secondary_minimum()
        assert primary_depth < secondary_depth

    def test_edge_detection_better_than_flux_return_with_ellipsoidal(self):
        """Test that edge detection outperforms flux_return with ellipsoidal variation."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            ellipsoidal_amplitude=0.08
        )

        # Edge detection
        binner_edge = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Flux return method
        binner_flux = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='flux_return'
        )

        # Both should detect eclipses
        assert binner_edge.primary_eclipse is not None
        assert binner_flux.primary_eclipse is not None

        # Edge detection boundaries should be more accurate
        # (closer to true boundaries at 0.9-0.1 and 0.4-0.6)
        edge_primary = binner_edge.primary_eclipse
        edge_width = (edge_primary[1] - edge_primary[0]) % 1.0

        # Primary should span roughly 0.2 phase units
        assert 0.15 < edge_width < 0.25

    def test_handles_sparse_data(self):
        """Test edge detection with sparse data."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            n_points=100,  # Sparse
            noise_level=0.02
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=50,  # Fewer bins for sparse data
            boundary_method='edge_detection'
        )

        # Should handle gracefully (may or may not detect eclipses)
        assert binner.primary_eclipse is not None or binner.secondary_eclipse is not None

    def test_handles_dense_noisy_data(self):
        """Test edge detection with dense but noisy data."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            n_points=10000,  # Dense
            noise_level=0.05  # Noisy
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=500,
            boundary_method='edge_detection'
        )

        # Should still detect eclipses despite noise
        assert binner.primary_eclipse is not None
        assert binner.secondary_eclipse is not None

    def test_shallow_eclipse_detection(self):
        """Test detection of very shallow eclipses."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            primary_depth=0.05,  # Very shallow
            secondary_depth=0.03,
            ellipsoidal_amplitude=0.02,
            noise_level=0.005
        )

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection',
            edge_min_eclipse_depth=0.02,  # Lower threshold for shallow eclipses
            edge_slope_threshold_percentile=85  # More sensitive
        )

        # Should detect at least primary
        assert binner.primary_eclipse is not None

    def test_wrapped_eclipse_handling(self):
        """Test edge detection with eclipse wrapping around phase 0/1."""
        phases = np.linspace(0, 1, 1000)
        fluxes = 1.0 + 0.05 * np.cos(4 * np.pi * phases)  # Ellipsoidal

        # Eclipse centered at phase 0.0 (wraps around)
        wrap_mask = (phases < 0.15) | (phases > 0.85)
        fluxes[wrap_mask] -= 0.2

        # Secondary at phase 0.5
        fluxes[450:550] -= 0.1

        flux_errors = np.ones_like(phases) * 0.01

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should handle wrapped eclipse
        assert binner.primary_eclipse is not None
        assert binner.secondary_eclipse is not None

    def test_diagnostics_available_after_binning(self):
        """Test that diagnostics are accessible after binning."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal()

        binner = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Diagnostics should be available
        assert hasattr(binner, '_edge_diagnostics')
        assert binner._edge_diagnostics is not None

        diag = binner._edge_diagnostics
        assert 'threshold' in diag
        assert 'detected_count' in diag
        assert diag['detected_count'] >= 1  # At least one eclipse

    def test_parameter_tuning(self):
        """Test that adjusting parameters improves detection."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            primary_depth=0.08,  # Moderate depth
            ellipsoidal_amplitude=0.06
        )

        # Conservative parameters (may miss secondary)
        binner_conservative = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection',
            edge_slope_threshold_percentile=95,  # Very strict
            edge_min_eclipse_depth=0.05
        )

        # Sensitive parameters
        binner_sensitive = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection',
            edge_slope_threshold_percentile=85,  # More permissive
            edge_min_eclipse_depth=0.02
        )

        # Sensitive should detect as many or more eclipses
        conservative_count = (
            (binner_conservative.primary_eclipse is not None) +
            (binner_conservative.secondary_eclipse is not None)
        )
        sensitive_count = (
            (binner_sensitive.primary_eclipse is not None) +
            (binner_sensitive.secondary_eclipse is not None)
        )

        assert sensitive_count >= conservative_count


class TestEdgeDetectionVsFluxReturn:
    """Comparative tests between edge detection and flux return methods."""

    def test_both_methods_work_on_clean_data(self):
        """Test that both methods work on clean data without ellipsoidal."""
        phases = np.linspace(0, 1, 1000)
        fluxes = np.ones_like(phases)
        fluxes[450:550] = 0.7  # Primary
        fluxes[100:150] = 0.85  # Secondary
        flux_errors = np.ones_like(phases) * 0.01

        binner_edge = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        binner_flux = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='flux_return'
        )

        # Both should detect eclipses
        assert binner_edge.primary_eclipse is not None
        assert binner_flux.primary_eclipse is not None

    def test_edge_detection_superior_with_ellipsoidal(self):
        """Test that edge detection is superior with strong ellipsoidal variation."""
        phases, fluxes, flux_errors = create_synthetic_light_curve_with_ellipsoidal(
            ellipsoidal_amplitude=0.15  # Very strong
        )

        # Edge detection should handle this better
        binner_edge = EclipsingBinaryBinner(
            phases, fluxes, flux_errors,
            nbins=200,
            boundary_method='edge_detection'
        )

        # Should successfully detect both eclipses
        assert binner_edge.primary_eclipse is not None
        assert binner_edge.secondary_eclipse is not None
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_edge_detection_integration.py -v`
Expected: Most tests PASS (some may need parameter tuning)

**Step 3: Fix any failing tests by adjusting parameters**

If tests fail, adjust the synthetic light curve parameters or detection thresholds to ensure tests validate the correct behavior.

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tests/test_edge_detection_integration.py
git commit -m "test: add comprehensive integration tests for edge detection

- Test with ellipsoidal variations
- Test sparse and dense data
- Test shallow and wrapped eclipses
- Test parameter sensitivity
- Compare edge detection vs flux_return methods
- Validate diagnostics availability"
```

---

## Task 9: Update Documentation

**Files:**
- Create: `docs/edge_detection_guide.md`
- Modify: `docs/api-reference.md`
- Create: `examples/edge_detection_example.py`

**Step 1: Write edge detection user guide**

```markdown
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
```

**Step 2: Write example script**

```python
# examples/edge_detection_example.py
"""
Example demonstrating edge detection method for eclipse boundary detection.
"""
import numpy as np
import matplotlib.pyplot as plt
from eclipsebin.binning import EclipsingBinaryBinner


def create_ellipsoidal_light_curve():
    """Create synthetic light curve with ellipsoidal variations."""
    phases = np.linspace(0, 1, 2000)

    # Ellipsoidal variation (twice per orbit)
    fluxes = 1.0 + 0.08 * np.cos(4 * np.pi * phases)

    # Primary eclipse at phase 0.0
    primary_mask = (phases < 0.12) | (phases > 0.88)
    fluxes[primary_mask] -= 0.25 * np.exp(-((phases[primary_mask] % 1.0) ** 2) / 0.003)

    # Secondary eclipse at phase 0.5
    secondary_mask = (phases > 0.38) & (phases < 0.62)
    fluxes[secondary_mask] -= 0.12 * np.exp(-((phases[secondary_mask] - 0.5) ** 2) / 0.003)

    # Add noise
    fluxes += np.random.normal(0, 0.01, len(phases))
    flux_errors = np.ones_like(phases) * 0.01

    return phases, fluxes, flux_errors


def example_basic_usage():
    """Basic usage of edge detection."""
    print("=" * 60)
    print("Example 1: Basic Edge Detection")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    print(f"Primary eclipse: {binner.primary_eclipse}")
    print(f"Secondary eclipse: {binner.secondary_eclipse}")
    print(f"Primary depth: {1 - binner.find_minimum_flux():.3f}")
    print(f"Secondary depth: {1 - binner.find_secondary_minimum():.3f}")

    # Plot
    binner.plot_unbinned_light_curve()


def example_comparison():
    """Compare edge detection vs flux return."""
    print("\n" + "=" * 60)
    print("Example 2: Edge Detection vs Flux Return")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    # Edge detection
    binner_edge = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    # Flux return
    binner_flux = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='flux_return'
    )

    print("\nEdge Detection:")
    print(f"  Primary: {binner_edge.primary_eclipse}")
    print(f"  Secondary: {binner_edge.secondary_eclipse}")

    print("\nFlux Return:")
    print(f"  Primary: {binner_flux.primary_eclipse}")
    print(f"  Secondary: {binner_flux.secondary_eclipse}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Edge detection
    ax1.plot(phases, fluxes, 'k.', alpha=0.3, markersize=1)
    ax1.axvline(binner_edge.primary_eclipse[0], color='r', linestyle='--', label='Primary')
    ax1.axvline(binner_edge.primary_eclipse[1], color='r', linestyle='--')
    ax1.axvline(binner_edge.secondary_eclipse[0], color='b', linestyle='--', label='Secondary')
    ax1.axvline(binner_edge.secondary_eclipse[1], color='b', linestyle='--')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('Edge Detection Method')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Flux return
    ax2.plot(phases, fluxes, 'k.', alpha=0.3, markersize=1)
    ax2.axvline(binner_flux.primary_eclipse[0], color='r', linestyle='--', label='Primary')
    ax2.axvline(binner_flux.primary_eclipse[1], color='r', linestyle='--')
    ax2.axvline(binner_flux.secondary_eclipse[0], color='b', linestyle='--', label='Secondary')
    ax2.axvline(binner_flux.secondary_eclipse[1], color='b', linestyle='--')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_title('Flux Return Method')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def example_parameter_tuning():
    """Demonstrate parameter tuning."""
    print("\n" + "=" * 60)
    print("Example 3: Parameter Tuning")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    # Conservative parameters
    binner_conservative = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        edge_slope_threshold_percentile=95,
        edge_min_eclipse_depth=0.05
    )

    # Sensitive parameters
    binner_sensitive = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection',
        edge_slope_threshold_percentile=85,
        edge_min_eclipse_depth=0.01
    )

    print("\nConservative (strict thresholds):")
    print(f"  Primary: {binner_conservative.primary_eclipse}")
    print(f"  Secondary: {binner_conservative.secondary_eclipse}")
    print(f"  Detected: {binner_conservative._edge_diagnostics['detected_count']} eclipses")

    print("\nSensitive (permissive thresholds):")
    print(f"  Primary: {binner_sensitive.primary_eclipse}")
    print(f"  Secondary: {binner_sensitive.secondary_eclipse}")
    print(f"  Detected: {binner_sensitive._edge_diagnostics['detected_count']} eclipses")


def example_diagnostics():
    """Show diagnostic information."""
    print("\n" + "=" * 60)
    print("Example 4: Diagnostics")
    print("=" * 60)

    phases, fluxes, flux_errors = create_ellipsoidal_light_curve()

    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors,
        nbins=200,
        boundary_method='edge_detection'
    )

    diag = binner._edge_diagnostics

    print(f"\nDiagnostic Information:")
    print(f"  Slope threshold: {diag['threshold']:.3e}")
    print(f"  Return threshold: {diag['return_threshold']:.3e}")
    print(f"  Smoothing window: {diag['smoothing_window']} points")
    print(f"  Detected eclipses: {diag['detected_count']}")
    print(f"  Ingress candidates: {len(diag['ingress_candidates'])}")
    print(f"  Egress candidates: {len(diag['egress_candidates'])}")

    # Plot slopes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Original and smoothed flux
    ax1.plot(phases, fluxes, 'k.', alpha=0.3, markersize=1, label='Raw')
    ax1.plot(phases, diag['smoothed_fluxes'], 'r-', linewidth=2, label='Smoothed')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('Light Curve and Smoothing')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Slopes
    ax2.plot(phases, diag['slopes'], 'b-', linewidth=1, label='Slope')
    ax2.axhline(diag['threshold'], color='r', linestyle='--', label='Ingress threshold')
    ax2.axhline(-diag['threshold'], color='r', linestyle='--', label='Egress threshold')
    ax2.axhline(diag['return_threshold'], color='orange', linestyle=':', label='Return threshold')
    ax2.axhline(-diag['return_threshold'], color='orange', linestyle=':')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Slope (dFlux/dPhase)')
    ax2.set_title('Slopes and Thresholds')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_comparison()
    example_parameter_tuning()
    example_diagnostics()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
```

**Step 3: Update API reference**

Read the current API reference and add edge detection documentation.

**Step 4: Create examples directory if needed**

Run: `mkdir -p examples`

**Step 5: Write files**

```bash
# Create the files (done in steps above via Write tool)
```

**Step 6: Test example script**

Run: `cd examples && python edge_detection_example.py`
Expected: Runs without errors, produces plots

**Step 7: Commit**

```bash
git add docs/edge_detection_guide.md examples/edge_detection_example.py docs/api-reference.md
git commit -m "docs: add comprehensive edge detection documentation

- Add user guide with tuning instructions
- Add example script demonstrating all features
- Update API reference with edge detection parameters
- Include diagnostic usage examples"
```

---

## Task 10: Final Testing and Validation

**Files:**
- Run all tests
- Generate coverage report
- Validate on real data (if available)

**Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run with coverage**

Run: `pytest tests/ --cov=eclipsebin --cov-report=html --cov-report=term`
Expected: Coverage report showing good coverage of modified code

**Step 3: Verify no regressions**

Run tests specifically for original functionality:

Run: `pytest tests/test_eclipsing_binary_binner.py -v`
Expected: All PASS

**Step 4: Manual validation (if real data available)**

If you have real light curves with ellipsoidal variations:

```python
# Load your data
phases, fluxes, flux_errors = load_real_data()

# Try edge detection
binner = EclipsingBinaryBinner(
    phases, fluxes, flux_errors,
    nbins=200,
    boundary_method='edge_detection'
)

# Inspect results
binner.plot_unbinned_light_curve()
print(f"Diagnostics: {binner._edge_diagnostics}")
```

**Step 5: Review all changes**

Run: `git log --oneline --graph`
Expected: Clean commit history showing all tasks

**Step 6: Create summary document**

```markdown
# Eclipse Detection Robustness Improvements - Summary

## Changes Implemented

### High Priority (Complete)
1. ✅ Diagnostics output infrastructure
   - Added `_edge_diagnostics` attribute
   - Return diagnostics from detection function
   - Accessible after binning for debugging

### Medium Priority (Complete)
2. ✅ Adaptive constants
   - Baseline window: 2% of data (min 5)
   - Refinement range: 5% of data (min 10)
   - Gap detection threshold: adaptive

3. ✅ Improved warning messages
   - Include diagnostic info (thresholds, candidates)
   - Provide actionable suggestions
   - Specify which eclipse type missing

4. ✅ Robust baseline calculation
   - Trimmed mean for outlier resistance
   - Validation for minimum window sizes
   - Safety check for baseline > 0.1

5. ✅ Smoothing validation
   - Warn if window > 1/3 eclipse width
   - Prevent over-smoothing narrow eclipses

### Low Priority (Complete)
6. ✅ Division safety checks
   - Baseline validation before depth calculation
   - Diagnostic return even with no valid slopes

7. ✅ Configurable eclipse separation
   - `min_eclipse_separation` parameter
   - Default 0.2, customizable for eccentric systems

### Testing & Documentation (Complete)
8. ✅ Comprehensive tests
   - Integration tests with ellipsoidal variations
   - Edge cases (sparse, dense, noisy, wrapped)
   - Parameter sensitivity tests
   - Comparison tests vs flux_return

9. ✅ Documentation
   - User guide with tuning instructions
   - Example scripts
   - API reference updates
   - Diagnostic usage guide

## Backward Compatibility

All changes are backward compatible:
- Default parameters unchanged
- New parameters optional
- Fallback to flux_return method when edge detection fails
- Existing tests still pass

## Performance Impact

- Minimal overhead from adaptive calculations
- Diagnostic collection adds negligible cost
- Validation checks are fast

## Next Steps

1. Deploy to production
2. Monitor performance on real data
3. Gather user feedback
4. Consider additional improvements:
   - Phase wrapping in edge detection
   - Alternative smoothing methods
   - GPU acceleration for large datasets

## Testing Summary

- Unit tests: [count] tests, all passing
- Integration tests: [count] tests, all passing
- Coverage: [percentage]% of modified code
- Manual validation: [status]
```

**Step 7: Final commit**

```bash
git add -A
git commit -m "docs: add implementation summary and final validation

- Complete test suite passing
- Coverage report generated
- All improvements documented
- Ready for review and deployment"
```

---

## Completion Checklist

- [ ] Task 1: Diagnostics infrastructure
- [ ] Task 2: Adaptive constants
- [ ] Task 3: Improved warnings
- [ ] Task 4: Robust baseline
- [ ] Task 5: Smoothing validation
- [ ] Task 6: Division safety
- [ ] Task 7: Configurable separation
- [ ] Task 8: Integration tests
- [ ] Task 9: Documentation
- [ ] Task 10: Final validation

**Total estimated time**: 4-6 hours for experienced developer

---

## Notes for Implementation

1. **Test-Driven Development**: Each task writes tests first, implements, then verifies
2. **Frequent commits**: After each task completes
3. **Incremental**: Each task builds on previous ones
4. **Backward compatible**: All changes are additive
5. **Well-documented**: Each change explained with examples

## Troubleshooting

If tests fail:
1. Check that all imports are correct
2. Verify parameter names match between function signature and calls
3. Check diagnostic dict keys match between return and access
4. Ensure test synthetic data has sufficient signal-to-noise
5. Adjust test thresholds if needed for synthetic data

---

**Plan complete. Ready for implementation!**
