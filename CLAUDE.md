# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**eclipsebin** is a Python package for non-uniform binning of eclipsing binary star light curves. It concentrates more bins in eclipse regions to better capture eclipse features compared to traditional uniform binning.

## Commands

```bash
# Install in development mode
pip install .

# Run all tests with coverage
pytest --cov

# Run a single test file
pytest tests/test_eclipsing_binary_binner.py

# Run specific test
pytest tests/test_eclipsing_binary_binner.py::test_unwrapped_light_curves -v

# Lint (check only)
black --check --verbose .

# Format code
black .
```

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

### Test Data

`tests/data/` contains real astronomical light curves (ASAS-SN and TESS missions) as `.npy` files used for integration testing.

## Code Style

- Formatter: Black
- Max line length: 100 (from .pylintrc)
- Python 3.9+ required
- Do NOT add emojis or the "Generated with Claude Code" tag to git commit message. It adds so much noise to every message. Keep them clean and concise
