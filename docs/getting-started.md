# Installation

Install the package using `pip`:

```bash
pip install eclipsebin
```

# Prepare your Light Curve

* **Phase your light curve.** Phases must fall between 0 and 1.
* **Convert magnitude to flux.** 
* **Normalize the fluxes.** The out-of-eclipse normalized flux should be close to 1.


# Create an EclipsingBinaryBinner Object

Assuming you already have your phases, fluxes, and flux uncertainties stored as `phases`, `fluxes`, and `fluxerrs`, respectively, initialize your `EclipsingBinaryBinner`:

```bash
import eclipsebin as ebin

# Example usage
binner = ebin.EclipsingBinaryBinner(phases, fluxes, fluxerrs, nbins=200, fraction_in_eclipse=0.5, atol_primary=0.001, atol_secondary=0.05)
```
Here `nbins` indicates the desired total number of bins, and `fraction_in_eclipse` indicates the fraction of that total number of bins that you wish to place within the eclipse regions. 

You can also optionally set `atol_primary` and/or `atol_secondary`, which specify the absolute tolerance of the corresponding eclipse ingress and egress regarding their proximity to one. These arguments are typically only necessary for systems with significant ellipsoidal variations, as the out-of-eclipse regions are more variable. By default, these values are calculated within the code by taking `proximity_to_one * 0.05`, where `proximity_to_one` indicates the distance of the corresponding eclipse minimum* from unity.

*Note that this is a rough approximation and may be inaccurate for some light curves. See [Binning Scheme](docs/binning-scheme.md) for more details.

Now just call one simple function to bin your light curve, and plot it if you wish.

```bash
bin_centers, bin_means, bin_stds = binner.bin_light_curve(plot=True)
```

And voila, you have your binned light curve! Do with it as you wish.