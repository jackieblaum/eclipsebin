# Fix Phase-Wrapped Eclipse Binning

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix binning issues for eclipses near phase boundaries by unwrapping phases before binning

**Architecture:** Detect phase-wrapped eclipses at initialization, shift all phases so no eclipse crosses the 0/1 boundary, perform all binning in unwrapped phase space, then optionally re-wrap results to [0, 1] range at the end.

**Tech Stack:** NumPy for array operations, pandas for binning (qcut), pytest for testing

---

## Task 1: Add phase unwrapping detection

**Files:**
- Modify: `eclipsebin/binning.py:30-87` (__init__ method)
- Test: `tests/test_eclipsing_binary_binner.py`

**Step 1: Write failing test for phase unwrapping detection**

Add to `tests/test_eclipsing_binary_binner.py` after line 479:

```python
def test_detect_phase_wrapping(wrapped_light_curve):
    """Test that phase wrapping is correctly detected"""
    phases, fluxes, flux_errors = wrapped_light_curve
    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=100, fraction_in_eclipse=0.2
    )
    # For wrapped_light_curve fixture, secondary eclipse wraps around 0/1
    # Check that phases were unwrapped (no eclipse crosses boundary)
    assert binner.primary_eclipse[0] < binner.primary_eclipse[1]
    assert binner.secondary_eclipse[0] < binner.secondary_eclipse[1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eclipsing_binary_binner.py::test_detect_phase_wrapping -v`

Expected: FAIL because secondary_eclipse boundaries are not properly ordered when wrapped

**Step 3: Add _detect_wrapped_eclipse method**

Add to `eclipsebin/binning.py` after line 147:

```python
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
```

**Step 4: Add _calculate_unwrap_shift method**

Add to `eclipsebin/binning.py` after the method from step 3:

```python
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

    # Shift so the wrapped eclipse is centered away from boundaries
    # Use midpoint of the eclipse that's NOT wrapped as reference
    if primary_wrapped and not secondary_wrapped:
        # Shift so primary is unwrapped - place it opposite secondary
        secondary_mid = (self.secondary_eclipse[0] + self.secondary_eclipse[1]) / 2
        shift = 0.5 - secondary_mid
    elif secondary_wrapped and not primary_wrapped:
        # Shift so secondary is unwrapped - place it opposite primary
        primary_mid = (self.primary_eclipse[0] + self.primary_eclipse[1]) / 2
        shift = 0.5 - primary_mid
    else:
        # Both wrapped (rare) - shift by 0.5
        shift = 0.5

    return shift % 1.0
```

**Step 5: Run test to verify it still fails**

Run: `pytest tests/test_eclipsing_binary_binner.py::test_detect_phase_wrapping -v`

Expected: FAIL - detection methods exist but not yet used in __init__

**Step 6: Commit detection methods**

```bash
git add eclipsebin/binning.py tests/test_eclipsing_binary_binner.py
git commit -m "feat: add phase wrap detection methods"
```

---

## Task 2: Unwrap phases at initialization

**Files:**
- Modify: `eclipsebin/binning.py:30-87` (__init__ method)
- Modify: `eclipsebin/binning.py:380-388` (shift_bin_edges method - remove, replace with rewrap)

**Step 1: Modify __init__ to unwrap phases immediately**

Replace `eclipsebin/binning.py:64-86` with:

```python
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
```

**Step 2: Add _unwrap_phases method**

Add to `eclipsebin/binning.py` after _calculate_unwrap_shift:

```python
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

    # Update eclipse minima in unwrapped space
    self.primary_eclipse_min_phase = (self.primary_eclipse_min_phase + self._phase_shift) % 1.0
    self.secondary_eclipse_min_phase = (self.secondary_eclipse_min_phase + self._phase_shift) % 1.0
```

**Step 3: Run test**

Run: `pytest tests/test_eclipsing_binary_binner.py::test_detect_phase_wrapping -v`

Expected: PASS - phases now unwrapped at initialization

**Step 4: Run all existing tests**

Run: `pytest tests/test_eclipsing_binary_binner.py -v`

Expected: Some failures in wrapped light curve tests because they expect wrapped behavior

**Step 5: Commit unwrapping implementation**

```bash
git add eclipsebin/binning.py
git commit -m "feat: unwrap phases at initialization to fix binning"
```

---

## Task 3: Remove old wrapping logic from binning methods

**Files:**
- Modify: `eclipsebin/binning.py:435-464` (calculate_eclipse_bins)
- Modify: `eclipsebin/binning.py:466-528` (calculate_out_of_eclipse_bins)

**Step 1: Simplify calculate_eclipse_bins**

Replace `eclipsebin/binning.py:435-464` with:

```python
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

    bins = pd.qcut(eclipse_phases, q=bins_in_eclipse)
    return np.array([interval.right for interval in np.unique(bins)])
```

**Step 2: Simplify calculate_out_of_eclipse_bins**

Replace `eclipsebin/binning.py:466-528` with:

```python
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

    # Since phases are unwrapped, we can directly slice between eclipses
    # OOE1: between end of secondary eclipse and start of primary eclipse
    end_idx_secondary_eclipse = np.searchsorted(
        self.data["phases"], self.secondary_eclipse[1]
    )
    start_idx_primary_eclipse = np.searchsorted(
        self.data["phases"], self.primary_eclipse[0]
    )
    ooe1_phases = self.data["phases"][
        end_idx_secondary_eclipse : start_idx_primary_eclipse + 1
    ]
    ooe1_bins = pd.qcut(ooe1_phases, q=bins_in_ooe1)
    ooe1_edges = np.array([interval.right for interval in np.unique(ooe1_bins)])

    # OOE2: between end of primary eclipse and start of secondary eclipse
    end_idx_primary_eclipse = np.searchsorted(
        self.data["phases"], self.primary_eclipse[1]
    )
    start_idx_secondary_eclipse = np.searchsorted(
        self.data["phases"], self.secondary_eclipse[0]
    )
    ooe2_phases = self.data["phases"][
        end_idx_primary_eclipse : start_idx_secondary_eclipse + 1
    ]
    ooe2_bins = pd.qcut(ooe2_phases, q=bins_in_ooe2)
    ooe2_edges = np.array([interval.right for interval in np.unique(ooe2_bins)])

    return ooe1_edges, ooe2_edges
```

**Step 3: Simplify calculate_eclipse_bins_distribution**

Replace `eclipsebin/binning.py:299-340` with:

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_eclipsing_binary_binner.py::test_unwrapped_light_curves -v`

Expected: PASS - unwrapped curves should work fine

**Step 5: Commit simplified binning logic**

```bash
git add eclipsebin/binning.py
git commit -m "refactor: remove wrapping logic from binning methods"
```

---

## Task 4: Replace shift_bin_edges with rewrap functionality

**Files:**
- Modify: `eclipsebin/binning.py:380-388` (shift_bin_edges)
- Modify: `eclipsebin/binning.py:390-433` (calculate_bins)

**Step 1: Replace shift_bin_edges with _rewrap_to_original_phase**

Replace `eclipsebin/binning.py:380-388` with:

```python
def _rewrap_to_original_phase(self, phases_array):
    """
    Rewrap phases back to original phase space before unwrapping.

    Args:
        phases_array (np.ndarray): Array of phases in unwrapped space

    Returns:
        np.ndarray: Phases shifted back to original space
    """
    if self._phase_shift == 0.0:
        return phases_array
    return (phases_array - self._phase_shift) % 1.0
```

**Step 2: Update calculate_bins to use rewrapping**

Replace `eclipsebin/binning.py:390-433` with:

```python
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
        if self.params["fraction_in_eclipse"] > 0.1:
            new_fraction_in_eclipse = self.params["fraction_in_eclipse"] - 0.1
            print(
                f"Requested fraction of bins in eclipse regions results in empty bins; "
                f"trying fraction_in_eclipse={new_fraction_in_eclipse}"
            )
            self.params["fraction_in_eclipse"] = new_fraction_in_eclipse
            return self.calculate_bins(return_in_original_phase=return_in_original_phase)
        raise ValueError("Not enough data to bin these eclipses.")

    # Rewrap to original phase space if requested
    if return_in_original_phase:
        bin_centers = self._rewrap_to_original_phase(bin_centers)
        bin_edges = self._rewrap_to_original_phase(bin_edges)

    return bin_centers, bin_means, bin_errors, bin_number, bin_edges
```

**Step 3: Run tests**

Run: `pytest tests/test_eclipsing_binary_binner.py -v`

Expected: Multiple failures - need to update tests and plotting

**Step 4: Commit rewrapping implementation**

```bash
git add eclipsebin/binning.py
git commit -m "feat: add rewrapping to return results in original phase"
```

---

## Task 5: Update plotting methods to handle unwrapped phases

**Files:**
- Modify: `eclipsebin/binning.py:530-567` (plot_binned_light_curve)
- Modify: `eclipsebin/binning.py:569-604` (plot_unbinned_light_curve)

**Step 1: Update plot_binned_light_curve**

Replace `eclipsebin/binning.py:530-567` with:

```python
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
    plt.xlim(0, 1)
    ylims = plt.ylim()

    # Get eclipse boundaries in original phase space
    primary_bounds = self._rewrap_to_original_phase(
        np.array(self.primary_eclipse)
    )
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
```

**Step 2: Update plot_unbinned_light_curve**

Replace `eclipsebin/binning.py:569-604` with:

```python
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
    primary_bounds = self._rewrap_to_original_phase(
        np.array(self.primary_eclipse)
    )
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
    plt.xlim(0, 1)
    plt.ylabel("Normalized Flux", fontsize=14)
    plt.xlabel("Phases", fontsize=14)
    plt.legend()
    plt.show()
```

**Step 3: Remove use_shifted_phases parameter**

The parameter `use_shifted_phases` is no longer needed. Remove it from:
- `find_minimum_flux_phase` (line 88)
- `find_secondary_minimum_phase` (line 115)
- `get_eclipse_boundaries` (line 149)
- `_find_eclipse_boundaries` (line 180)
- `_find_eclipse_boundary` (line 226)
- `_helper_secondary_minimum_mask` (line 139)

Simplify these methods to only work with unwrapped phases.

**Step 4: Run tests**

Run: `pytest tests/test_eclipsing_binary_binner.py -v`

Expected: Some failures - tests need updating

**Step 5: Commit plotting updates**

```bash
git add eclipsebin/binning.py
git commit -m "feat: update plotting to use original phase space"
```

---

## Task 6: Update tests to match new behavior

**Files:**
- Modify: `tests/test_eclipsing_binary_binner.py:282-318` (helper_eclipse_detection)
- Modify: `tests/test_eclipsing_binary_binner.py:249-280` (helper_find_eclipse_minima)

**Step 1: Update helper_eclipse_detection**

Replace `tests/test_eclipsing_binary_binner.py:282-318` with:

```python
def helper_eclipse_detection(
    phases, fluxes, flux_errors, nbins, fraction_in_eclipse, wrapped
):
    """
    Test the eclipse detection capabilities of EclipsingBinaryBinner.
    With unwrapping, all eclipses should have proper ordering.
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )

    # In unwrapped space, eclipses should always have proper ordering
    primary_min = binner.primary_eclipse_min_phase
    primary_eclipse = binner.primary_eclipse
    assert 0 <= primary_min <= 1
    assert 0 <= primary_eclipse[0] <= 1
    assert 0 <= primary_eclipse[1] <= 1
    # After unwrapping, boundaries should be properly ordered
    assert primary_eclipse[0] < primary_min < primary_eclipse[1]

    secondary_min = binner.secondary_eclipse_min_phase
    secondary_eclipse = binner.secondary_eclipse
    assert 0 <= secondary_min <= 1
    assert 0 <= secondary_eclipse[0] <= 1
    assert 0 <= secondary_eclipse[1] <= 1
    # After unwrapping, boundaries should be properly ordered
    assert secondary_eclipse[0] < secondary_min < secondary_eclipse[1]
```

**Step 2: Simplify helper_find_eclipse_minima**

Replace `tests/test_eclipsing_binary_binner.py:249-280` with:

```python
def helper_find_eclipse_minima(phases, fluxes, flux_errors, nbins, fraction_in_eclipse):
    """
    Test the find_minimum_flux method of EclipsingBinaryBinner.
    """
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=nbins,
        fraction_in_eclipse=fraction_in_eclipse,
    )
    primary_minimum_phase = binner.primary_eclipse_min_phase
    assert 0 <= primary_minimum_phase <= 1.0

    secondary_minimum_phase = binner.secondary_eclipse_min_phase
    assert 0 <= secondary_minimum_phase <= 1.0
```

**Step 3: Remove obsolete test helper calls**

In `test_unwrapped_light_curves` and `test_secondary_wrapped_light_curves`, the wrapped parameter is no longer needed. Update calls to:

```python
helper_eclipse_detection(
    phases,
    fluxes,
    flux_errors,
    nbins,
    fraction_in_eclipse,
    wrapped=None,  # No longer used
)
```

Or better yet, remove the wrapped parameter entirely from the function signature in step 1.

**Step 4: Run tests**

Run: `pytest tests/test_eclipsing_binary_binner.py -v`

Expected: PASS for most tests

**Step 5: Commit test updates**

```bash
git add tests/test_eclipsing_binary_binner.py
git commit -m "test: update tests for unwrapped phase behavior"
```

---

## Task 7: Add comprehensive tests for edge cases

**Files:**
- Test: `tests/test_eclipsing_binary_binner.py`

**Step 1: Write test for primary eclipse wrapping**

Add to `tests/test_eclipsing_binary_binner.py` after line 479:

```python
@pytest.fixture
def primary_wrapped_light_curve():
    """
    Fixture for light curve with primary eclipse wrapping around phase boundary.
    """
    np.random.seed(42)
    phases = np.linspace(0, 0.999, 10000)
    fluxes = np.ones_like(phases)
    # Primary eclipse wraps: 0.95-1.0 and 0.0-0.05
    fluxes[9500:10000] = np.linspace(0.95, 0.8, 500)
    fluxes[0:500] = np.linspace(0.8, 0.95, 500)
    # Secondary eclipse at 0.5
    fluxes[4800:5200] = np.linspace(0.95, 0.9, 400)
    flux_errors = np.random.normal(0.01, 0.001, 10000)
    random_indices = np.random.choice(range(len(phases)), size=5000, replace=False)
    return phases[random_indices], fluxes[random_indices], flux_errors[random_indices]


def test_primary_wrapped_eclipse(primary_wrapped_light_curve):
    """Test binning with primary eclipse wrapping around boundary"""
    phases, fluxes, flux_errors = primary_wrapped_light_curve
    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=100, fraction_in_eclipse=0.2
    )

    # Verify unwrapping detected and applied
    assert binner._phase_shift != 0.0

    # Verify eclipses are properly ordered in unwrapped space
    assert binner.primary_eclipse[0] < binner.primary_eclipse[1]
    assert binner.secondary_eclipse[0] < binner.secondary_eclipse[1]

    # Verify binning works
    bin_centers, bin_means, bin_errors = binner.bin_light_curve(plot=False)
    assert len(bin_centers) == 100
    assert np.all(bin_errors > 0)
    assert np.all((bin_centers >= 0) & (bin_centers <= 1))
```

**Step 2: Run test**

Run: `pytest tests/test_eclipsing_binary_binner.py::test_primary_wrapped_eclipse -v`

Expected: PASS

**Step 3: Write test for both eclipses near boundary**

Add test:

```python
@pytest.fixture
def both_near_boundary_light_curve():
    """
    Fixture with both eclipses near phase boundaries.
    """
    np.random.seed(123)
    phases = np.linspace(0, 0.999, 10000)
    fluxes = np.ones_like(phases)
    # Primary at 0.05
    fluxes[400:600] = np.linspace(0.95, 0.8, 200)
    # Secondary at 0.95
    fluxes[9400:9600] = np.linspace(0.95, 0.9, 200)
    flux_errors = np.random.normal(0.01, 0.001, 10000)
    random_indices = np.random.choice(range(len(phases)), size=5000, replace=False)
    return phases[random_indices], fluxes[random_indices], flux_errors[random_indices]


def test_both_eclipses_near_boundary(both_near_boundary_light_curve):
    """Test binning when both eclipses are near phase boundaries"""
    phases, fluxes, flux_errors = both_near_boundary_light_curve
    binner = EclipsingBinaryBinner(
        phases, fluxes, flux_errors, nbins=100, fraction_in_eclipse=0.2
    )

    # Verify binning succeeds
    bin_centers, bin_means, bin_errors = binner.bin_light_curve(plot=False)
    assert len(bin_centers) == 100
    assert np.all(bin_errors > 0)
```

**Step 4: Run all tests**

Run: `pytest tests/test_eclipsing_binary_binner.py -v`

Expected: All tests PASS

**Step 5: Commit edge case tests**

```bash
git add tests/test_eclipsing_binary_binner.py
git commit -m "test: add edge cases for phase wrapping"
```

---

## Task 8: Update CLAUDE.md with new architecture

**Files:**
- Modify: `CLAUDE.md:9-23` (Architecture section)

**Step 1: Update architecture description**

Replace `CLAUDE.md:9-23` with:

```markdown
## Architecture

### Core Module: `eclipsebin/binning.py`

The `EclipsingBinaryBinner` class is the entire public API. Key workflow:

1. **Eclipse Detection**: Finds primary eclipse (global flux minimum) and secondary eclipse (minimum ≥0.2 phase away)
2. **Boundary Detection**: Locates where flux returns to ~1.0 using adaptive tolerance based on eclipse depth
3. **Phase Unwrapping**: Detects if any eclipse wraps around the phase boundary (0/1) and shifts all phases to unwrap if needed
4. **Bin Allocation**: Distributes bins between eclipse regions (configurable fraction, default 20%) and out-of-eclipse regions
5. **Binning**: Uses `pandas.qcut()` to ensure equal data points per bin; propagates flux errors
6. **Rewrapping**: Returns results in original phase space (before unwrapping) by default

Key design decisions:
- All eclipse detection and binning performed in unwrapped phase space (no eclipses cross boundaries)
- Results automatically rewrapped to original [0, 1] phase space for output
- Graceful degradation: reduces `fraction_in_eclipse` if binning fails
- Minimum requirements: 10 data points, 10 bins, data points ≥ 5× nbins
```

**Step 2: Commit documentation update**

```bash
git add CLAUDE.md
git commit -m "docs: update architecture for phase unwrapping"
```

---

## Task 9: Run full test suite and fix any remaining issues

**Files:**
- All test files

**Step 1: Run complete test suite**

Run: `pytest tests/ -v --cov=eclipsebin --cov-report=term-missing`

Expected: All tests pass with high coverage

**Step 2: Run linting**

Run: `black --check --verbose .`

Expected: All files properly formatted

**Step 3: Format if needed**

Run: `black .`

**Step 4: Run tests again**

Run: `pytest tests/ -v`

Expected: All tests PASS

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: format code with black"
```

---

## Task 10: Test with real data

**Files:**
- Manual testing

**Step 1: Create test script**

Create `test_real_data.py`:

```python
#!/usr/bin/env python3
"""Manual test with real ASAS-SN and TESS data"""
from pathlib import Path
import numpy as np
from eclipsebin import EclipsingBinaryBinner

# Test ASAS-SN unwrapped
data_path = Path("tests/data/lc_asas_sn_unwrapped.npy")
phases, fluxes, flux_errors = np.load(data_path)
print("Testing ASAS-SN unwrapped...")
binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=200, fraction_in_eclipse=0.2)
bin_centers, bin_means, bin_errors = binner.bin_light_curve(plot=False)
print(f"  ✓ Binned to {len(bin_centers)} bins")
print(f"  ✓ Phase shift: {binner._phase_shift}")

# Test TESS unwrapped
data_path = Path("tests/data/lc_tess_unwrapped.npy")
phases, fluxes, flux_errors = np.load(data_path)
print("Testing TESS unwrapped...")
binner = EclipsingBinaryBinner(phases, fluxes, flux_errors, nbins=200, fraction_in_eclipse=0.2)
bin_centers, bin_means, bin_errors = binner.bin_light_curve(plot=False)
print(f"  ✓ Binned to {len(bin_centers)} bins")
print(f"  ✓ Phase shift: {binner._phase_shift}")

print("\n✓ All real data tests passed!")
```

**Step 2: Run test script**

Run: `python test_real_data.py`

Expected: Success messages for both datasets

**Step 3: Remove test script**

Run: `rm test_real_data.py`

**Step 4: Verify with plotting (optional)**

If desired, manually test with `plot=True` to visually verify binning quality

**Step 5: Done**

Implementation complete!
