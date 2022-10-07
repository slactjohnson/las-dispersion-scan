import numpy as np
import pypret
import pypret.frequencies
import scipy
import scipy.interpolate


def pulse_from_spectrum(wavelength, spectrum, pulse):
    """Generates a pulse instance from a measured spectrum."""
    # scale to intensity over frequency, convert to amplitude and normalize
    # spectrum = spectrum * wavelength * wavelength
    spectrum[spectrum < 0.0] = 0.0
    spectrum = np.sqrt(spectrum + 0.0j)
    spectrum /= spectrum.max()
    # calculate angular frequencies
    w = pypret.frequencies.convert(wavelength, "wl", "om")
    pulse.spectrum = scipy.interpolate.interp1d(
        w - pulse.w0, spectrum, bounds_error=False, fill_value=0.0
    )(pulse.w)
    return pulse


def preprocess(trace, signal_range=None, dark_signal_range=None):
    dark_signal = None
    if dark_signal_range is not None:
        dark_signal = trace.copy()
        dark_signal.limit(dark_signal_range, axes=1)
        dark_signal = np.median(dark_signal.data, axis=1)
    if signal_range is not None:
        trace.limit(signal_range, axes=1)
    if dark_signal is not None and dark_signal_range is not None:
        # subtract dark counts for every spectrum separately
        trace.data -= dark_signal[:, None]
    # normalize
    trace.normalize()
    return trace


def preprocess2(trace, pnps):
    if trace.units[1] == "m":
        # scaled in wavelength -> has to be corrected
        wavelength = trace.axes[1]
        frequency = pypret.frequencies.convert(wavelength, "wl", "om")
        trace.scale(wavelength * wavelength)
        trace.normalize()
        trace.axes[1] = frequency
        trace.units[1] = "Hz"
    trace.interpolate(axis2=pnps.process_w)
