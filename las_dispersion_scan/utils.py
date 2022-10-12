import functools
import pathlib
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import happi
import numpy as np
import ophyd
import pypret
import pypret.frequencies
import scipy
import scipy.interpolate
from qtpy import QtCore

SOURCE_PATH = pathlib.Path(__file__).resolve().parent


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


class RetrievalOptionsStandin(SimpleNamespace):
    """
    The retriever returns a SimpleNamespace with ``.options`` matching these
    attributes.

    This class exists for type hinting purposes only.
    """

    alpha: float
    maxfev: Optional[float]
    maxiter: int


class RetrievalResultStandin(SimpleNamespace):
    """
    The retriever returns a SimpleNamespace that contains (likely) all of
    these attributes.

    This class exists for type hinting purposes only.
    """

    parameter: np.ndarray
    options: RetrievalOptionsStandin
    logging: bool
    measurement: pypret.MeshData
    pnps: Optional[pypret.pnps.BasePNPS]
    # the pulse spectra
    # 1 - the retrieved pulse
    pulse_retrieved: np.ndarray
    # 2 - the original test pulse, optional
    pulse_original: Optional[np.ndarray]
    # 3 - the initial guess
    pulse_initial: np.ndarray

    # the measurement traces
    # 1 - the original data used for retrieval
    trace_input: np.ndarray
    # 2 - the trace error and the trace calculated from the retrieved pulse
    trace_error: float
    trace_retrieved: np.ndarray
    response_function: float
    # the weights
    weights: np.ndarray

    # this is set if the original spectrum is provided
    # the trace error of the test pulse (non-zero for noisy input)
    trace_error_optimal: float
    # 3 - the optimal trace calculated from the test pulse
    trace_original: float
    pulse_error: float
    # the logged trace errors
    trace_errors: np.ndarray
    # the running minimum of the trace errors (for plotting)
    rm_trace_errors: np.ndarray


_happi_client = None


def get_happi_client() -> happi.Client:
    """
    Get the global happi client instance.

    Returns
    -------
    happi.Client
        The client instance.
    """
    global _happi_client
    if _happi_client is None:
        _happi_client = happi.Client.from_config()
    return _happi_client


def get_device_from_happi(name: str, client: Optional[happi.Client] = None) -> object:
    """
    Get an ophyd Device from happi.

    Parameters
    ----------
    name : str
        The device name.
    client : happi.Client, optional
        The happi client, if available.  Otherwise, determined from environment
        settings.
    """
    if client is None:
        client = get_happi_client()
    return client[name].get()


def run_in_gui_thread(func: Callable, *args, _start_delay_ms: int = 0, **kwargs):
    """Run the provided function in the GUI thread."""
    QtCore.QTimer.singleShot(_start_delay_ms, functools.partial(func, *args, **kwargs))


def channel_from_device(device: ophyd.Device) -> str:
    """PyDM-compatible PV name URIs from a given ophyd Device."""
    return f"ca://{device.prefix}"


def channel_from_signal(signal: ophyd.signal.EpicsSignalBase) -> str:
    """PyDM-compatible PV name URIs from a given EpicsSignal."""
    return f"ca://{signal.pvname}"
