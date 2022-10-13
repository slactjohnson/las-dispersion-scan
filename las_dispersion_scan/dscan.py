from __future__ import annotations

import copy
import dataclasses
import logging
import os
import pathlib
from typing import List, Optional, Protocol, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pypret
import pypret.frequencies
import scipy.interpolate
from matplotlib.ticker import EngFormatter
from scipy.ndimage import gaussian_filter

from . import plotting
from .options import Material, NonlinearProcess, PulseAnalysisMethod, RetrieverSolver
from .plotting import RetrievalResultPlot
from .utils import RetrievalResultStandin, get_pulse_spectrum, preprocess

logger = logging.getLogger(__name__)


def get_fundamental_spectrum(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    range_low: float,
    range_high: float,
) -> np.ndarray:
    """
    Column-stack the wavelengths/intensities for the fundamental spectrum.

    Parameters
    ----------
    wavelengths : np.ndarray
        Fundamental wavelengths.
    intensities : np.ndarray
        Fundamental intensities.
    range_low : float
        Wavelength lower limit.
    range_high : float
        Wavelength upper limit.

    Returns
    -------
    np.ndarray
        Column-stacked wavelengths/intensities with the range applied.
    """
    assert wavelengths.shape == intensities.shape

    wavelength_fund = wavelengths[
        (wavelengths > range_low) & (wavelengths < range_high)
    ]
    intensities_fund = intensities[
        (wavelengths > range_low) & (wavelengths < range_high)
    ]
    return np.column_stack((wavelength_fund, intensities_fund))


@dataclasses.dataclass
class SpectrumData:
    wavelengths: np.ndarray
    intensities: np.ndarray

    def _get_pulse(self, ft: pypret.FourierTransform) -> Tuple[pypret.Pulse, float]:
        """
        Get a pypret.Pulse instance for this SpectrumData.

        Parameters
        ----------
        ft : pypret.FourierTransform
            The fourier transform used for the retrieval process, based on the
            scan data.

        Returns
        -------
        pypret.Pulse
            The pulse instance.
        float
            The fourier transform limit.
        """
        pulse = pypret.Pulse(ft, self.raw_center)
        _, fund_intensities_bkg_sub = self.subtract_background()
        pulse.spectrum = get_pulse_spectrum(
            wavelength=self.wavelengths,
            spectrum=fund_intensities_bkg_sub,
            pulse=pulse,
        )
        fourier_transform_limit = pulse.fwhm(dt=pulse.dt / 100)
        logger.info(
            f"Fourier Transform Limit (FTL): {fourier_transform_limit * 1e15:.1f} fs"
        )
        return pulse, fourier_transform_limit

    @property
    def raw_center(self) -> float:
        """Wavelength raw center."""
        return sum(np.multiply(self.wavelengths, self.intensities)) / sum(
            self.intensities
        )
        # wavelength_raw_center = self.wavelength_fund * 1E-9

    def truncate_wavelength(
        self, range_low: float, range_high: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Truncating wavelength given the provided range."""
        idx = (self.wavelengths > range_low) & (self.wavelengths < range_high)
        return self.wavelengths[idx].copy(), self.intensities[idx].copy()

    @classmethod
    def from_txt_file(cls, filename: str) -> SpectrumData:
        """
        Load fundamental spectrum data from the old .txt file format.

        Parameters
        ----------
        filename : str
            Directory where the old files are to be found.

        Returns
        -------
        SpectrumData
            The loaded data
        """
        return cls.from_file(filename)

    @classmethod
    def from_dat_file(cls, filename: str) -> SpectrumData:
        """
        Load fundamental spectrum data from the old .txt file format or the new
        .dat format.

        Parameters
        ----------
        filename : str
            Directory where the old files are to be found.

        Returns
        -------
        SpectrumData
            The loaded data
        """
        return cls.from_file(filename)

    @classmethod
    def from_path(cls, path: Union[pathlib.Path, str]) -> SpectrumData:
        """
        Load spectra from either the old .txt file format or a the new .dat
        file format, given a path.

        Parameters
        ----------
        path : str
            Directory where the old files are to be found.

        Returns
        -------
        ScanData
            The data from the scan.
        """
        try:
            return cls.from_dat_file(os.path.join(path, "fund.dat"))
        except FileNotFoundError:
            ...

        try:
            return cls.from_txt_file(os.path.join(path, "fund_conv.txt"))
        except FileNotFoundError:
            ...

        raise FileNotFoundError(f"No supported .dat or .txt file found in {path}")

    @classmethod
    def from_file(cls, path: Union[pathlib.Path, str]) -> SpectrumData:
        """
        Load fundamental spectrum data from the old .txt file format or the new
        .dat format.

        Parameters
        ----------
        path : str
            Filename that contains the data.

        Returns
        -------
        SpectrumData
            The loaded data
        """
        fund_data = np.loadtxt(path)
        wavelengths = fund_data[:, 0] * 1e-9
        intensities = fund_data[:, 1]
        intensities -= np.average(intensities[:7])
        # intensities *= wavelengths * wavelengths
        return cls(wavelengths, intensities)

    def get_background(self, *, count: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the background, based on the first and last ``count`` items.

        Parameters
        ----------
        count : int, optional
            The size of the slice.

        Returns
        -------
        np.ndarray
            Wavelength background
        np.ndarray
            Intensity background
        """
        wavelength_bkg = np.hstack(
            (self.wavelengths[:count], self.wavelengths[-count:])
        )
        intensities_bkg = np.hstack(
            (self.intensities[:count], self.intensities[-count:])
        )
        return wavelength_bkg, intensities_bkg

    def subtract_background(self, *, count: int = 15, threshold: float = 0.0025):
        """
        Subtract the background from both

        Parameters
        ----------
        count : int, optional
            The size of the slice to be used for background subtraction.
        threshold : float, optional
            Normalized intensities under ``threshold`` are zeroed.

        Returns
        -------
        np.ndarray
            Background-subtracted wavelengths based on a 1D polyfit
            of wavelength/intensity background.
        np.ndarray
            Background-subtracted and normalized intensity.  Intensities under
            ``threshold`` are zeroed.
        """
        wavelength_bkg, intensities_bkg = self.get_background(count=count)
        fit = np.polyfit(wavelength_bkg, intensities_bkg, 1)
        wavelength_fit = self.wavelengths * fit[0] + fit[1]
        intensities = self.intensities - wavelength_fit
        intensities /= np.max(intensities)
        intensities[intensities < threshold] = 0
        return wavelength_fit, intensities

    def plot(self, pulse: Optional[pypret.Pulse] = None):
        """
        Plot the spectrum data.

        Parameters
        ----------
        pulse : pypret.Pulse or None, optional
            Pulse, if available, to show FTL.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plot figure.
        matplotlib.pyplot.Axis
            The plot axis.
        """
        fig, ax = plt.subplots()
        ax = cast(plt.Axes, ax)
        wavelength_bkg, intensities_bkg = self.get_background(count=15)
        intensities_bkg_fit, _ = self.subtract_background(count=15)

        ax.plot(self.wavelengths * 1e9, self.intensities, "k", label="All data")
        ax.plot(
            wavelength_bkg * 1e9,
            intensities_bkg,
            "xb",
            label="Points for fit",
        )
        ax.plot(
            self.wavelengths * 1e9,
            intensities_bkg_fit,
            "r",
            label="Background fit",
        )
        plt.legend()
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Counts (arb.)")

        if pulse is not None:
            ftl = pulse.fwhm(dt=pulse.dt / 100)
            ax.set_title(f"Fundamental Spectrum (FTL) = {ftl * 1e15:.1f} fs")

        return fig, ax


@dataclasses.dataclass
class ScanData:
    #: Scan positions
    positions: np.ndarray
    #: Wavelengths
    wavelengths: np.ndarray
    #: Normalized intensities
    intensities: np.ndarray

    def subtract_background_for_all_positions(self, count: int = 15) -> None:
        """
        Clean scan by subtracting linear background for each stage position.

        Works in-place.
        """
        wavelength_bkg = np.hstack(
            (self.wavelengths[:count], self.wavelengths[-count:])
        )
        for i in range(len(self.positions)):
            intensities_bkg = np.hstack(
                (self.intensities[i, :count], self.intensities[i, -count:])
            )
            fit = np.polyfit(wavelength_bkg, intensities_bkg, 1)
            self.intensities[i, :] -= self.wavelengths * fit[0] + fit[1]

    def truncate_wavelength(
        self, range_low: float, range_high: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Truncating wavelength given the provided range."""
        idx = (self.wavelengths > range_low) & (self.wavelengths < range_high)
        return self.wavelengths[idx].copy(), self.intensities[:, idx].copy()

    @classmethod
    def from_txt_file(cls, filename: str) -> ScanData:
        """
        Load scan spectra from the "old" (original?) .txt file format.

        Parameters
        ----------
        filename : str
            Directory where the old files are to be found.

        Returns
        -------
        ScanData
            The data from the scan.
        """
        dscan_conv = np.loadtxt(filename)
        result = np.empty([len(dscan_conv[0, 1:]), len(dscan_conv[:, 0])])
        result[0, 1:] = dscan_conv[1:, 0].transpose()
        result[1:, 0] = dscan_conv[0, 2:].transpose()
        result[1:, 1:] = dscan_conv[1:, 2:].transpose()
        # np.savetxt(path_full_scan, result)
        return cls(
            positions=result[0],
            wavelengths=result[1],
            intensities=result[2],
        )

    @classmethod
    def from_dat_file(cls, filename: str) -> ScanData:
        """
        Load scan positions and spectra from the new .dat format.

        Parameters
        ----------
        filename : str
            Directory where the old files are to be found.

        Returns
        -------
        ScanData
            The data from the scan.
        """
        scan_data = np.loadtxt(filename)
        positions = scan_data[0, 1:] * 1e-3
        wavelengths = scan_data[1:, 0] * 1e-9
        intensities = scan_data[1:, 1:].transpose()
        intensities /= np.amax(intensities)
        return cls(positions, wavelengths, intensities)

    @classmethod
    def from_path(cls, path: Union[pathlib.Path, str]) -> ScanData:
        """
        Load scan positions and spectra from either the old .txt file format
        or a the new .dat file format, given a path.

        Parameters
        ----------
        path : str
            Directory where the old files are to be found.

        Returns
        -------
        ScanData
            The data from the scan.
        """
        try:
            return cls.from_dat_file(os.path.join(path, "scan.dat"))
        except FileNotFoundError:
            ...

        try:
            return cls.from_txt_file(os.path.join(path, "dscan_conv.txt"))
        except FileNotFoundError:
            ...

        raise FileNotFoundError(f"No supported .dat or .txt file found in {path}")


@dataclasses.dataclass
class Acquisition:
    fundamental: SpectrumData
    scan: ScanData

    @classmethod
    def from_path(cls, path: Union[pathlib.Path, str]) -> Acquisition:
        """
        Load fundamental spectrum and scan data from a path.

        Parameters
        ----------
        path : str
            Directory where the old files are to be found.

        Returns
        -------
        Acquisition
            The data from the scan.
        """
        return cls(
            fundamental=SpectrumData.from_path(path),
            scan=ScanData.from_path(path),
        )


def _default_ndarray():
    """Helper for optional ndarray values in dataclass fields."""
    return np.zeros(0)


class Callback(Protocol):
    def __call__(self, data: List[np.ndarray]) -> None:
        # mu: np.ndarray
        # parameter: np.ndarray
        # process_w: np.ndarray
        # new_spectrum: np.ndarray
        ...


@dataclasses.dataclass
class PypretResult:
    #: The fundamental spectrum data.
    fund: SpectrumData
    #: The data acquired when performing the d-scan.
    scan: ScanData
    #: The material used in the d-scan.
    material: Material
    #: The pulse analysis method to use
    method: PulseAnalysisMethod
    #: The non-linear process to use
    nlin_process: NonlinearProcess
    #: The wedge angle (for traditional d-scan, not grating)
    wedge_angle: float = 8.0

    #: Strength of Gaussian blur applied to raw data (standard deviations).
    blur_sigma: int = 0
    #: Number of grid points in frequency and time axes.
    num_grid_points: int = 3000
    #: Bandwidth around center wavelength for frequency and time axes (nm).
    freq_bandwidth_wl: int = 950
    #: Maximum number of iterations
    max_iter: int = 30

    #: Grating stage position for pypret plot OR glass insertion stage position
    #: (mm, use None for shortest duration)
    plot_position: Optional[float] = None
    #: The retriever solver to use.  Defaults to "copra".
    solver: RetrieverSolver = RetrieverSolver.copra
    #: Fundamental spectrum window
    spec_fund_range: Tuple[float, float] = (400, 600)
    #: Scan spectrum window
    spec_scan_range: Tuple[float, float] = (200, 300)
    #: The plot instance that can be used to inspect the retrieval result.
    plot: Optional[RetrievalResultPlot] = None
    #: The result of the retrieval process.
    retrieval: RetrievalResultStandin = dataclasses.field(
        default_factory=RetrievalResultStandin
    )
    rms_error: float = 0.0
    #: A model of the femtosecond pulses by their envelope
    pulse: Optional[pypret.Pulse] = None
    #: The raw (not pre-processed) mesh data.
    trace_raw: Optional[pypret.MeshData] = None
    #: The pre-processed mesh data.
    trace: Optional[pypret.MeshData] = None
    #: The per-parameter full-width half-max (FWHM).
    fwhm: np.ndarray = dataclasses.field(default_factory=_default_ndarray)
    #: The per-parameter resulting profile.
    result_profile: np.ndarray = dataclasses.field(default_factory=_default_ndarray)
    #: The fourier transform limit (FTL) of the pulse.
    fourier_transform_limit: float = 0.0

    def _get_fourier_transform(self) -> pypret.FourierTransform:
        """
        Get the fourier transform helper from pypret.

        This is based on:
        * freq_bandwidth_wl: Bandwidth around center wavelength for frequency
          and time axes (nm)
        * fund.raw_center: The fundamental spectrum raw center position
        * num_grid_points: the number of grid points in frequency and time
           axes.

        Returns
        -------
        pypret.FourierTransform
        """
        # Create frequency-time grid
        freq_bandwidth = (
            self.freq_bandwidth_wl
            * 1e-9
            * 2
            * np.pi
            * 2.998e8
            / self.fund.raw_center**2
        )
        fund_frequency_step = np.round(freq_bandwidth / (self.num_grid_points - 1), 0)
        return pypret.FourierTransform(
            self.num_grid_points, dw=fund_frequency_step, w0=-freq_bandwidth / 2
        )

    def _get_mesh_data(self) -> pypret.MeshData:
        """
        Get the MeshData instance based on:

        * The selected material (and wedge angle)
        * The scan positions
        * The intensities/wavelengths acquired during the scan

        Returns
        -------
        pypret.MeshData
        """
        coef = self.material.get_coefficient(self.wedge_angle)
        return pypret.MeshData(
            self.scan.intensities,
            coef * (self.scan.positions - min(self.scan.positions)),
            self.scan.wavelengths,
            labels=["Insertion", "Wavelength"],
            units=["m", "m"],
        )

    def plot_mesh_data(
        self, data: Optional[pypret.MeshData] = None, scan_padding_nm: int = 75
    ) -> pypret.MeshDataPlot:
        """
        Plot the mesh scan data.

        Parameters
        ----------
        data : pypret.MeshData, optional
            The mesh data to plot, if available.  Re-calculated if not.
        scan_padding_nm : int, optional
            Padding around the scan range to use.

        Returns
        -------
        pypret.MeshDataPlot
        """
        if data is None:
            data = self._get_mesh_data()

        md = pypret.MeshDataPlot(data, show=False)
        ax = cast(plt.Axes, md.ax)
        ax.set_title("Cropped scan")
        ax.set_xlim(
            (self.spec_scan_range[0] + scan_padding_nm) * 1e-9,
            (self.spec_scan_range[1] - scan_padding_nm) * 1e-9,
        )
        return md

    def plot_processed_scan(
        self, *, fig: Optional[plt.Figure] = None, scan_padding_nm: int = 75
    ) -> pypret.MeshDataPlot:
        """
        Plot the processed mesh scan data from ``self.trace``.

        Parameters
        ----------
        scan_padding_nm : int, optional
            Padding around the scan range to use.

        Returns
        -------
        pypret.MeshDataPlot
        """
        factor = 2 * np.pi * 2.99792 * 1e17
        md = pypret.MeshDataPlot(self.trace, show=False)
        ax = cast(plt.Axes, md.ax)
        ax.set_title("Processed scan")
        ax.set_xlabel("Frequency")
        ax.set_xlim(
            factor / (self.spec_scan_range[1] - scan_padding_nm),
            factor / (self.spec_scan_range[0] + scan_padding_nm),
        )
        return md

    @property
    def pulse_width_fs(self) -> float:
        """Get the reconstructed pulse width in femtoseconds."""
        pulse = self._get_retrieval_pulse()
        return pulse.fwhm(dt=pulse.dt / 100) / 1e-15

    def _get_retrieval_pulse(self) -> pypret.Pulse:
        pulse = pypret.Pulse(self.retrieval.pnps.ft, self.retrieval.pnps.w0, unit="om")
        pulse.spectrum = self.retrieval.pulse_retrieved * self.retrieval.pnps.mask(
            self._plot_param
        )
        return pulse

    def plot_time_domain_retrieval(
        self,
        fig: Optional[plt.Figure] = None,
        yaxis: plotting.PlotYAxis = plotting.PlotYAxis.intensity,
        limit: bool = True,
        oversampling: int = 8,
        phase_blanking: bool = True,
        phase_blanking_threshold: float = 1e-3,
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Plot the retrieval result in the time domain.

        Parameters
        ----------
        fig : plt.Figure or None, optional
            An optional figure to use for the plot.
        yaxis : plotting.PlotYAxis, optional
            The Y axis setting.
        limit : bool, optional
            Determine and apply a limit using pypret.
        oversampling : int, optional
            Oversampling count
        phase_blanking : bool, optional
            Enable phase blanking with pypret masking.
        phase_blanking_threshold : float, optional
            Phase blanking threshold.

        Returns
        -------
        plt.Figure
            The figure used.
        plt.Axes
            The left axis.
        plt.Axes
            The right axis.
        """
        assert self.plot is not None
        assert self.retrieval is not None

        # construct the figure
        if fig is None:
            fig = cast(plt.Figure, plt.figure())

        ax1 = cast(plt.Axes, fig.subplots(nrows=1, ncols=1))
        ax12 = cast(plt.Axes, ax1.twinx())

        pulse = self._get_retrieval_pulse()
        if oversampling:
            t = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * oversampling)
            field2 = pulse.field_at(t).copy()
        else:
            t = pulse.t.copy()
            field2 = pulse.field.copy()
        field2 /= np.abs(field2).max()

        result_parameter_mid_idx = np.floor(len(field2) / 2) + 1
        profile_max_idx = np.abs(field2).argmax()
        field3 = np.roll(field2, -round(profile_max_idx - result_parameter_mid_idx))

        li11, li12, _, _ = pypret.graphics.plot_complex(
            t,
            field3,
            ax1,
            ax12,
            yaxis=yaxis.value,
            phase_blanking=phase_blanking,
            limit=limit,
            phase_blanking_threshold=phase_blanking_threshold,
        )
        li11.set_linewidth(3.0)
        li11.set_color("#1f77b4")
        li11.set_alpha(0.6)
        li12.set_linewidth(3.0)
        li12.set_color("#ff7f0e")
        li12.set_alpha(0.6)

        fwhm = np.round(pulse.fwhm(dt=pulse.dt / 100) / 1e-15, 2)

        fx = EngFormatter(unit="s")
        ax1.xaxis.set_major_formatter(fx)
        ax1.set_title(
            f"time domain @ {self._final_plot_position:.3f} mm (FWHM = {fwhm} fs)"
        )
        ax1.set_xlabel("time")
        ax1.set_ylabel(yaxis.value)
        ax12.set_ylabel("phase (rad)")
        ax1.legend([li11, li12], [yaxis.value, "phase"])
        ax1.set_xlim([-10 * 1e-15 * np.round(fwhm, 0), 10 * 1e-15 * np.round(fwhm, 0)])
        return fig, ax1, ax12

    def plot_frequency_domain_retrieval(
        self,
        fig: Optional[plt.Figure] = None,
        xaxis: plotting.PlotXAxis = plotting.PlotXAxis.wavelength,
        yaxis: plotting.PlotYAxis = plotting.PlotYAxis.intensity,
        limit: bool = True,
        oversampling: int = 8,
        phase_blanking: bool = True,
        phase_blanking_threshold: float = 1e-3,
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Plot the retrieval result in the time domain.

        Parameters
        ----------
        fig : plt.Figure or None, optional
            An optional figure to use for the plot.
        xaxis : plotting.PlotXAxis, optional
            The X axis setting.
        yaxis : plotting.PlotYAxis, optional
            The Y axis setting.
        limit : bool, optional
            Determine and apply a limit using pypret.
        oversampling : int, optional
            Oversampling count
        phase_blanking : bool, optional
            Enable phase blanking with pypret masking.
        phase_blanking_threshold : float, optional
            Phase blanking threshold.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes, plt.Axes]

        """
        assert self.plot is not None
        assert self.retrieval is not None

        # construct the figure
        if fig is None:
            fig = cast(plt.Figure, plt.figure())

        ax2 = cast(plt.Axes, fig.subplots(nrows=1, ncols=1))
        ax22 = cast(plt.Axes, ax2.twinx())

        pulse = self._get_retrieval_pulse()
        if oversampling:
            w = np.linspace(pulse.w[0], pulse.w[-1], pulse.N * oversampling)
            spectrum2 = pulse.spectrum_at(w).copy()
            pulse.spectrum = self.retrieval.pulse_retrieved.copy()
        else:
            w = pulse.w.copy()
            spectrum2 = self.retrieval.pulse_retrieved.copy()
        fund_w = (
            pypret.frequencies.convert(self.fund.wavelengths, "wl", "om") - pulse.w0
        )
        spectrum2 /= np.abs(spectrum2).max()
        if self.fund is None:
            fundamental = None
        else:
            fundamental = self.get_fund_intensities_bkg_sub(
                use_pulse_spectral_intensity=True
            )
            fundamental /= np.abs(fundamental).max()

        if xaxis == plotting.PlotXAxis.wavelength:
            w = pypret.frequencies.convert(w + pulse.w0, "om", "wl")
            fund_w = self.fund.wavelengths
            unit = "m"
            label = "wavelength"
        elif xaxis == plotting.PlotXAxis.frequency:
            unit = " rad Hz"
            label = "frequency"
        else:
            raise ValueError(f"Unsupported x-axis for plotting: {xaxis}")

        # Plot in spectral domain
        li21, li22, _, _ = plotting.plot_complex_phase(
            w,
            spectrum2,
            ax2,
            ax22,
            yaxis=yaxis.value,
            phase_blanking=phase_blanking,
            limit=limit,
            phase_blanking_threshold=phase_blanking_threshold,
        )
        lines = [li21, li22]
        labels = ["intensity", "phase"]
        if fundamental is not None:
            (li31,) = ax2.plot(fund_w, fundamental, "r", ms=4.0, mew=1.0, zorder=0)
            lines.append(cast(plt.Line2D, li31))
            labels.append("measurement")
        li21.set_linewidth(3.0)
        li21.set_color("#1f77b4")
        li21.set_alpha(0.6)
        li22.set_linewidth(3.0)
        li22.set_color("#ff7f0e")
        li22.set_alpha(0.6)

        fx = EngFormatter(unit=unit)
        ax2.xaxis.set_major_formatter(fx)
        ftl = self.fourier_transform_limit * 1e15
        ax2.set_title(f"frequency domain (FTL = {ftl:.2f} fs)")
        ax2.set_xlabel(label)
        ax2.set_ylabel(yaxis.value)
        ax22.set_ylabel("phase (rad)")
        ax2.legend(lines, labels)
        ax2.set_xlim([self.spec_fund_range[0] * 1e-9, self.spec_fund_range[1] * 1e-9])
        return fig, ax2, ax22

    def plot_trace(
        self,
        fig: Optional[plt.Figure] = None,
        option: plotting.PlotTrace = plotting.PlotTrace.retrieved,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the retrieval result in the time domain.

        Parameters
        ----------
        fig : plt.Figure or None, optional
            An optional figure to use for the plot.
        option : plotting.PlotTrace, optional
            The type of plot to create.

        Returns
        -------
        plt.Figure
        plt.Axes
        """
        assert self.plot is not None
        assert self.retrieval is not None

        # construct the figure
        if fig is None:
            fig = cast(plt.Figure, plt.figure())

        ax = cast(plt.Axes, fig.subplots(nrows=1, ncols=1))
        sc = 1.0 / self.retrieval.trace_input.max()

        if option == plotting.PlotTrace.measured:
            trace = self.retrieval.trace_input * sc
            cmap = "nipy_spectral"
            vmin = 0
            vmax = 1
            title = "Measured"
        elif option == plotting.PlotTrace.retrieved:
            trace = self.retrieval.trace_retrieved * sc
            cmap = "nipy_spectral"
            vmin = 0
            vmax = 1
            title = "Retrieved"
        elif option == plotting.PlotTrace.difference:
            diff = self.retrieval.trace_input - self.retrieval.trace_retrieved
            trace = diff * self.retrieval.weights * sc
            cmap = "RdBu"
            vmin = "auto"
            vmax = "auto"
            title = "Difference"
            if np.any(self.retrieval.weights != 1.0):
                title = "Weighted difference"
        else:
            raise ValueError(f"Unsupported option: {option}")

        md = self.retrieval.measurement
        x, y = pypret.lib.edges(
            self.retrieval.pnps.process_w / (2 * np.pi)
        ), pypret.lib.edges(self.scan.positions)
        if vmin == "auto":
            vmin = -np.amax(abs(trace))
        if vmax == "auto":
            vmax = np.amax(abs(trace))

        im = ax.pcolormesh(x, y, trace, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        # plt.xticks(fontsize=8)
        # ax.set_xlabel(md.labels[1])
        ax.set_xlabel("frequency")
        if md.labels is not None:
            ax.set_ylabel(md.labels[0])

        if md.units is not None:
            fx = EngFormatter(unit=md.units[1])
            ax.xaxis.set_major_formatter(fx)
            fy = EngFormatter(unit=md.units[0])
            ax.yaxis.set_major_formatter(fy)

        ax.set_title(title)
        scan_padding = 75  # (nm)
        ax.set_xlim(
            2.99792 * 1e17 / (self.spec_scan_range[1] - scan_padding),
            2.99792 * 1e17 / (self.spec_scan_range[0] + scan_padding),
        )  # No factor of 2*pi
        return fig, ax

    def _calculate_fwhm_and_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates per-retrieval-parameter 'result_profile' and 'fwhm'.

        Side-effect: modifies self.pulse.spectrum

        Returns
        -------
        np.ndarray
            The FWHM array.
        np.ndarray
            The result profile.
        """
        pulse = self.pulse
        assert pulse is not None
        result_parameter_mid_idx = np.floor(len(pulse.field) / 2) + 1
        result_profile = np.zeros((len(pulse.field), len(self.retrieval.parameter)))
        fwhm = np.zeros((len(self.retrieval.parameter), 1))
        for idx, param in enumerate(self.retrieval.parameter):
            # Updating the spectrum property -> field gets updated
            pulse.spectrum = self.retrieval.pulse_retrieved * self.retrieval.pnps.mask(
                param
            )
            profile = np.power(np.abs(pulse.field), 2)[:]
            profile_max_idx = np.argmax(profile)
            result_profile[:, idx] = np.roll(
                profile, -round(profile_max_idx - result_parameter_mid_idx)
            )
            try:
                fwhm[idx] = pulse.fwhm(dt=pulse.dt / 100)
            except Exception:
                fwhm[idx] = np.nan

        return fwhm, result_profile

    @property
    def optimum_fwhm_idx(self) -> np.ndarray:
        """Optimum full-width half-max (FWHM) index."""
        return np.nanargmin(self.fwhm)

    @property
    def optimum_fwhm(self) -> float:
        """Optimum full-width half-max (FWHM) value."""
        return self.fwhm[self.optimum_fwhm_idx][0]

    def plot_fwhm_vs_grating_position(self):
        """
        Plot FWHM vs grating position.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plot figure.
        matplotlib.pyplot.Axis
            The plot axis.
        """
        fig = plt.figure()
        ax = cast(plt.Axes, fig.add_subplot(111))
        ax.plot(self.scan.positions * 1e3, self.fwhm * 1e15)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("FWHM (fs)")

        result_optimum_fwhm = self.optimum_fwhm
        fwhm0 = result_optimum_fwhm * 1e15
        pos = self.scan.positions[self.optimum_fwhm_idx] * 1e3
        ax.set_title(f"Shortest: {fwhm0:.1f} fs @ {pos:.3f} mm")
        ax.set_ylim(
            0,
            min([np.nanmax(self.fwhm * 1e15), 4 * result_optimum_fwhm * 1e15]),
        )
        return fig, ax

    def plot_temporal_profile_vs_grating_position(self):
        """
        Plot temporal profile vs grating position.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plot figure.
        matplotlib.pyplot.Axis
            The plot axis.
        """
        assert self.pulse is not None
        fig = plt.figure()
        ax = cast(plt.Axes, fig.add_subplot(111))
        fig = plt.contourf(
            self.pulse.t * 1e15,
            self.scan.positions * 1e3,
            self.result_profile.transpose(),
            200,
            cmap="nipy_spectral",
        )
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Position (mm)")
        ax.set_title("Dscan Temporal Profile")
        ax.set_xlim(-8 * self.optimum_fwhm * 1e15, 8 * self.optimum_fwhm * 1e15)
        return fig, ax

    def get_fund_intensities_bkg_sub(
        self, use_pulse_spectral_intensity: bool = False
    ) -> np.ndarray:
        """
        Fundamental intensities with background subtracted.
        """
        assert self.pulse is not None
        _, intensities = self.fund.subtract_background()
        if not use_pulse_spectral_intensity:
            return intensities

        result_spec = scipy.interpolate.interp1d(
            self.pulse.wl,
            self.pulse.spectral_intensity,
            bounds_error=False,
            fill_value=0.0,
        )(self.fund.wavelengths)
        return intensities * pypret.lib.best_scale(intensities, result_spec)

    def _get_rms_error(self) -> float:
        """
        RMS error of the final result, held in ``self.pulse``.

        Returns
        -------
        float
        """
        assert self.pulse is not None
        result_spec = scipy.interpolate.interp1d(
            self.pulse.wl,
            self.pulse.spectral_intensity,
            bounds_error=False,
            fill_value=0.0,
        )(self.fund.wavelengths)
        return pypret.lib.nrms(
            self.get_fund_intensities_bkg_sub(use_pulse_spectral_intensity=True),
            result_spec,
        )

    def _retrieve(self, callback: Optional[Callback] = None) -> RetrievalResultStandin:
        assert self.pulse is not None

        retriever = self._get_retriever(callback)
        pypret.random_gaussian(self.pulse, self.fourier_transform_limit, phase_max=0.1)
        retriever.retrieve(self.trace, self.pulse.spectrum, weights=None)
        return cast(RetrievalResultStandin, retriever.result())

    def _get_retriever(
        self, callback: Optional[Callback] = None
    ) -> pypret.retrieval.retriever.BaseRetriever:
        """
        Get the pypret Retriever instance for the pulse.

        Side-effects:

        1. Sets self.trace_raw
        2. Sets self.trace

        Returns
        -------
        pypret.Retriever
        """
        assert self.pulse is not None
        self.trace_raw = self._get_mesh_data()
        pnps = pypret.PNPS(
            self.pulse,
            method=self.method.value,
            process=self.nlin_process.value,
            material=self.material.pypret_material,
        )
        self.trace = preprocess(
            self.trace_raw,
            signal_range=(self.scan.wavelengths[0], self.scan.wavelengths[-1]),
            dark_signal_range=(0, 10),
        )

        if self.trace.units is not None and self.trace.units[1] == "m":
            # scaled in wavelength -> has to be corrected
            wavelength = cast(np.ndarray, self.trace.axes[1])
            frequency = pypret.frequencies.convert(wavelength, "wl", "om")
            self.trace.scale(wavelength * wavelength)
            self.trace.normalize()
            self.trace.axes[1] = frequency
            self.trace.units[1] = "Hz"

        self.trace.interpolate(axis2=pnps.process_w)

        return pypret.Retriever(
            pnps,
            RetrieverSolver(self.solver).value,
            callback=callback,
            verbose=True,
            maxiter=self.max_iter,
        )

    @property
    def _plot_param(self) -> float:
        """Plot parameter for RetrievalResultPlot."""
        if self.plot_position is None:
            return self.retrieval.parameter[self.optimum_fwhm_idx]
        return self.retrieval.parameter[
            (np.nanargmin(np.abs(self.scan.positions - self.plot_position * 1e-3)))
        ]

    @property
    def _final_plot_position(self):
        """Final plot position for RetrievalResultPlot"""
        if self.plot_position is None:
            return self.scan.positions[self.optimum_fwhm_idx]
        return self.plot_position

    def _get_retrieval_plot(self) -> RetrievalResultPlot:
        """
        Get the "full" retrieval plot, with all pertinent information in one
        nice overview.

        Returns
        -------
        RetrievalResultPlot
        """
        return RetrievalResultPlot(
            retrieval_result=self.retrieval,
            retrieval_parameter=self._plot_param,
            fund_range=self.spec_fund_range,
            scan_range=self.spec_scan_range,
            final_position=self._final_plot_position,
            scan_positions=self.scan.positions,
            fundamental=self.get_fund_intensities_bkg_sub(
                use_pulse_spectral_intensity=True
            ),
            fundamental_wavelength=self.fund.wavelengths,
            fourier_transform_limit=self.fourier_transform_limit,
        )

    @classmethod
    def from_data(
        cls,
        fund: SpectrumData,
        scan: ScanData,
        material: Material,
        method: PulseAnalysisMethod,
        nlin_process: NonlinearProcess,
        wedge_angle: float = 8.0,
        blur_sigma: int = 0,
        num_grid_points: int = 3000,
        freq_bandwidth_wl: int = 950,
        max_iter: int = 30,
        plot_position: Optional[float] = None,
        spec_fund_range: Tuple[float, float] = (400, 600),
        spec_scan_range: Tuple[float, float] = (200, 300),
    ) -> PypretResult:
        """
        Generate a PypretResult based on a copy of the input parameters.

        Parameters
        ----------
        fund : SpectrumData
            The fundamental spectrum data.
        scan : ScanData
            The data acquired when performing the d-scan.
        material : Material
            The material used in the d-scan.
        method : PulseAnalysisMethod
            The pulse analysis method to use
        nlin_process : NonlinearProcess
            The non-linear process to use
        verbose : bool, optional
            Verbose mode.  Plot everything and show the results at the end.
        wedge_angle : float, optional
            The wedge angle (for traditional d-scan, not grating)
        blur_sigma : int, optional
            Strength of Gaussian blur applied to raw data (standard deviations).
        num_grid_points : int, optional
            Number of grid points in frequency and time axes.
        freq_bandwidth_wl : int, optional
            Bandwidth around center wavelength for frequency and time axes (nm).
        max_iter : int, optional
            Maximum number of iterations
        plot_position : float or None, optional
            Grating stage position for pypret plot OR glass insertion stage
            position (mm, use None for shortest duration)
        spec_fund_range : Tuple[float, float], optional
            Fundamental spectrum window
        spec_scan_range : Tuple[float, float], optional
            Scan spectrum window

        Returns
        -------
        PypretResult
        """
        return PypretResult(
            fund=copy.deepcopy(fund),
            scan=copy.deepcopy(scan),
            material=material,
            method=method,
            nlin_process=nlin_process,
            wedge_angle=wedge_angle,
            blur_sigma=blur_sigma,
            num_grid_points=num_grid_points,
            freq_bandwidth_wl=freq_bandwidth_wl,
            max_iter=max_iter,
            plot_position=plot_position,
            spec_fund_range=spec_fund_range,
            spec_scan_range=spec_scan_range,
        )

    def run(self, callback: Optional[Callback] = None, verbose: bool = False):
        """
        Pre-process the data and run the retrieval process as described below.

        This method does a number of things:

        1. Truncates the fundamental spectrum data based on spec_fund_range
        2. Configures the pypret.FourierTransform instance that specifies a
           temporal and spectral grid. based on freq_bandwidth_wl,
           fund.raw_center, and num_grid_points
        3. Configures the pypret.Pulse based on the fourier transform from (2),
           the fundamental spectrum raw center, background-subtracted
           fundamental spectra.  Uses ``get_pulse_spectrum`` to set
           ``pulse.spectrum``.  The fourier transform limit is also calculated
           in this step.
        4. Applies a gaussian filter with sigma ``blur_sigma`` on the acquired
           scan intensities.  The scan spectra are then truncated based on
           wavelength limits ``spec_scan_range``.
        5. Subtracts the background for each spectra in the scan.
        6. Generates a pypret.Retriever for the pulse.  This utilizes the
           specified method, non-linear process, material, and solver.
           It also preprocesses the trace data (via ``preprocess``) and
           utilizes previously-generated data above.
           The signal range is configured to be the first/last wavelength
           of the scan data.
           The trace is interpolated based on PNPS ``process_w``.
        8. A random gaussian is applied to the resulting pulse with FWHM
           of the fourier transform limit calculated above.
        9. The retrieval process is run with the ``trace`` and the pulse
           spectrum.  This is stored in ``retrieval``.
        10. RMS error, per-parameter profiles and fwhm are then calculated.

        The resulting PypretResult object may be used to inspect the process
        or generate additional customized plots.

        Parameters
        ----------
        callback : Callback or None, optional
            A callback to run at each iteration.  Of the signature
            ``callback(PypretResult, List[np.ndarray])``
        verbose : bool, optional
            Plot all "extra" / debug plots.
        """

        self.fund.wavelengths, self.fund.intensities = self.fund.truncate_wavelength(
            range_low=self.spec_fund_range[0] * 1e-9,
            range_high=self.spec_fund_range[1] * 1e-9,
        )
        logger.info(
            f"Fundamental center wavelength: {self.fund.raw_center * 1e9:.1f} nm"
        )

        ft = self._get_fourier_transform()
        logger.info(f"Time step = {ft.dt * 1e15:.2f} fs")

        self.pulse, self.fourier_transform_limit = self.fund._get_pulse(ft)
        self.scan.intensities = gaussian_filter(
            self.scan.intensities, sigma=self.blur_sigma
        )

        # Clean scan by truncating wavelength
        (self.scan.wavelengths, self.scan.intensities,) = self.scan.truncate_wavelength(
            range_low=self.spec_scan_range[0] * 1e-9,
            range_high=self.spec_scan_range[1] * 1e-9,
        )

        self.scan.subtract_background_for_all_positions()

        self.retrieval = self._retrieve(callback)

        self.rms_error = self._get_rms_error()
        logger.info(f"RMS spectrum error: {self.rms_error}")

        self.fwhm, self.result_profile = self._calculate_fwhm_and_profile()
        self.plot = self._get_retrieval_plot()

        if verbose:
            self.plot_all_debug()

    def plot_all_debug(
        self,
        limit: bool = True,
        oversampling: int = 8,
        phase_blanking: bool = True,
        phase_blanking_threshold: float = 1e-3,
        show: bool = True,
    ) -> None:
        """
        Plot all "verbose" or debug plots with the provided settings.

        Parameters
        ----------
        limit : bool, optional
            Determine and apply a limit using pypret.
        oversampling : int, optional
            Oversampling count
        phase_blanking : bool, optional
            Enable phase blanking with pypret masking.
        phase_blanking_threshold : float, optional
            Phase blanking threshold.
        show : bool, optional
            Show the plots.
        """

        self.fund.plot(self.pulse)
        self.plot_mesh_data()
        self.plot_fwhm_vs_grating_position()
        self.plot_temporal_profile_vs_grating_position()
        if self.plot is not None:
            self.plot.plot(
                oversampling=oversampling,
                phase_blanking=phase_blanking,
                phase_blanking_threshold=phase_blanking_threshold,
                limit=limit,
            )
        if show:
            plt.show()
