from typing import Optional, Tuple

import numpy as np
import pypret
import pypret.frequencies
import scipy
import scipy.interpolate


def get_pulse_spectrum(
    wavelength: np.ndarray, spectrum: np.ndarray, pulse: pypret.Pulse
) -> np.ndarray:
    """
    From a measured spectrum, generate the pypret.Pulse spectrum.

    Modifies ``pulse.spectrum`` in-place.
    """
    # scale to intensity over frequency, convert to amplitude and normalize
    # spectrum = spectrum * wavelength * wavelength
    spectrum[spectrum < 0.0] = 0.0
    spectrum = np.sqrt(spectrum + 0.0j)
    spectrum /= spectrum.max()
    # calculate angular frequencies
    w = pypret.frequencies.convert(wavelength, "wl", "om")
    return scipy.interpolate.interp1d(
        w - pulse.w0, spectrum, bounds_error=False, fill_value=0.0
    )(pulse.w)


def preprocess(
    trace: pypret.MeshData,
    signal_range: Optional[Tuple[float, float]] = None,
    dark_signal_range: Optional[Tuple[float, float]] = None,
) -> pypret.MeshData:
    """
    Preprocess the mesh scan data.

    Parameters
    ----------
    trace : pypret.MeshData
        The mesh data, including scanned positions, wavelengths, and
        intensities.
    signal_range : Tuple[float, float], optional
        The low/high range of the signal.
    dark_signal_range : Tuple[float, float], optional
        The low/high range of the dark signal.

    Returns
    -------
    pypret.MeshData
        A new instance of pre-processed mesh data.
    """
    trace = trace.copy()
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
