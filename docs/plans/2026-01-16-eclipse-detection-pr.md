# Eclipse Detection PR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get `feature/edge-detection-eclipse-boundaries` ready for a GitHub PR by adding pytest coverage, removing the ad-hoc integration script, updating user docs, and verifying tests/formatting.

**Architecture:** The new edge-based boundary detection lives in `eclipsebin/binning.py` and is invoked via `boundary_method="edge_detection"`, falling back to the existing flux-return method when needed. The plan adds focused pytest coverage using a synthetic ellipsoidal light curve and exercises the fallback path. Docs are updated to advertise the new boundary method and parameters.

**Tech Stack:** Python 3.9+, numpy, scipy, pandas, pytest, matplotlib.

### Task 1: Add pytest coverage for edge detection (@superpowers:test-driven-development)

**Files:**
- Create: `tests/test_edge_detection.py`
- Modify: `eclipsebin/binning.py` (only if tests reveal gaps)

**Step 1: Write the failing test file**

Create `tests/test_edge_detection.py` with:

```python
import numpy as np
import pytest

from eclipsebin import EclipsingBinaryBinner


@pytest.fixture
def ellipsoidal_light_curve():
    np.random.seed(42)
    n_points = 1000
    phases = np.linspace(0, 1, n_points, endpoint=False)
    fluxes = np.ones(n_points)

    # Ellipsoidal variation (2x orbital frequency)
    ellipsoidal_amplitude = 0.05
    fluxes += ellipsoidal_amplitude * np.cos(2 * np.pi * 2 * phases)

    # Primary eclipse
    primary_phase = 0.25
    primary_width = 0.05
    primary_depth = 0.2
    fluxes[np.abs(phases - primary_phase) < primary_width / 2] -= primary_depth

    # Secondary eclipse
    secondary_phase = 0.75
    secondary_width = 0.03
    secondary_depth = 0.1
    fluxes[np.abs(phases - secondary_phase) < secondary_width / 2] -= secondary_depth

    # Normalize to median = 1
    flux_median = np.median(fluxes)
    fluxes = fluxes / flux_median
    flux_errors = np.ones(n_points) * 0.01 / flux_median

    return phases, fluxes, flux_errors


def test_edge_detection_boundaries_with_ellipsoidal_variation(ellipsoidal_light_curve):
    phases, fluxes, flux_errors = ellipsoidal_light_curve
    binner = EclipsingBinaryBinner(
        phases,
        fluxes,
        flux_errors,
        nbins=200,
        boundary_method="edge_detection",
        edge_slope_threshold_percentile=90.0,
        edge_return_threshold_fraction=0.1,
    )

    primary_start, primary_end = binner.primary_eclipse
    secondary_start, secondary_end = binner.secondary_eclipse

    assert 0.20 <= primary_start <= 0.25
    assert 0.25 <= primary_end <= 0.30
    assert 0.72 <= secondary_start <= 0.76
    assert 0.74 <= secondary_end <= 0.78


def test_edge_detection_falls_back_to_flux_return(ellipsoidal_light_curve):
    phases, fluxes, flux_errors = ellipsoidal_light_curve
    with pytest.warns(UserWarning):
        binner = EclipsingBinaryBinner(
            phases,
            fluxes,
            flux_errors,
            nbins=200,
            boundary_method="edge_detection",
            edge_min_eclipse_depth=1.0,
        )

    assert len(binner.primary_eclipse) == 2
    assert len(binner.secondary_eclipse) == 2
```

**Step 2: Run the new tests to confirm they fail first (or document unexpected pass)**

Run: `pytest tests/test_edge_detection.py -v`
Expected: FAIL if the edge detection does not match expected boundary ranges or fallback behavior.

**Step 3: Make the minimal code change if needed**

If the test fails due to edge detection thresholds, adjust the algorithm in `eclipsebin/binning.py` to meet the assertions. Keep changes minimal and focused on returning boundary ranges within the expected windows. Re-run the test after each adjustment.

**Step 4: Re-run the test to confirm pass**

Run: `pytest tests/test_edge_detection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_edge_detection.py eclipsebin/binning.py
git commit -m "test: cover edge detection boundaries"
```

### Task 2: Remove the ad-hoc integration script

**Files:**
- Delete: `test_edge_detection_integration.py`

**Step 1: Delete the file**

Run: `rm test_edge_detection_integration.py`
Expected: file removed

**Step 2: Confirm repository references**

Run: `rg -n "edge_detection_integration" -S .`
Expected: no matches

**Step 3: Run the edge detection tests**

Run: `pytest tests/test_edge_detection.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add -u
git commit -m "chore: remove edge detection integration script"
```

### Task 3: Update README and docs landing page for edge detection option

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`

**Step 1: Update the "How it Works" eclipse boundary bullet**

Replace the existing eclipse boundary bullet with:

```markdown
- **Eclipse Boundaries**: By default, boundaries are where the flux returns to ~1.0. For light curves with strong ellipsoidal variations, set `boundary_method="edge_detection"` to use slope-based edge detection; tune it with `edge_slope_threshold_percentile`, `edge_return_threshold_fraction`, and `edge_min_eclipse_depth`.
```

**Step 2: Expand the Usage example to show edge detection**

Update the usage example block in both `README.md` and `docs/index.md` to:

```bash
import eclipsebin as ebin

# Default boundary method
binner = ebin.EclipsingBinaryBinner(
    phases, fluxes, fluxerrs,
    nbins=200,
    fraction_in_eclipse=0.5,
    atol_primary=0.001,
    atol_secondary=0.05,
)

# Edge detection boundary method (robust to ellipsoidal variations)
edge_binner = ebin.EclipsingBinaryBinner(
    phases,
    fluxes,
    fluxerrs,
    nbins=200,
    boundary_method="edge_detection",
    edge_slope_threshold_percentile=90.0,
    edge_return_threshold_fraction=0.1,
    edge_min_eclipse_depth=0.01,
)
```

**Step 3: Ensure doc blocks are identical between README and docs**

Run: `diff -u README.md docs/index.md | head -n 40`
Expected: only intentional differences outside the updated sections

**Step 4: Commit**

```bash
git add README.md docs/index.md
git commit -m "docs: describe edge detection boundaries"
```

### Task 4: Final verification and PR prep (@superpowers:verification-before-completion)

**Files:**
- Verify: `eclipsebin/binning.py`
- Verify: `tests/test_edge_detection.py`
- Verify: `README.md`
- Verify: `docs/index.md`

**Step 1: Run formatting and test suite**

Run: `black --check --verbose .`
Expected: no formatting changes required

Run: `pytest --cov`
Expected: PASS

**Step 2: Confirm clean status and collect PR notes**

Run: `git status -sb`
Expected: clean working tree on `feature/edge-detection-eclipse-boundaries`

Run: `git log --oneline main..HEAD`
Expected: list of new commits to summarize in PR

**Step 3: Draft PR summary/testing notes**

Prepare a PR description with:
- Summary: "Add slope-based edge detection option for eclipse boundaries, add pytest coverage, update docs"
- Testing: `pytest tests/test_edge_detection.py -v`, `pytest --cov`
