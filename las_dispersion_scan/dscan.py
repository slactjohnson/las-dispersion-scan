from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import os
from types import SimpleNamespace
from typing import Any, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pypret
import pypret.frequencies
import scipy.interpolate
from scipy.ndimage import gaussian_filter

from .plotting import RetrievalResultPlot
from .utils import get_pulse_spectrum, preprocess

logger = logging.getLogger(__name__)


oo = None  # TODO remove ocean optics references


class PulseAnalysisMethod(str, enum.Enum):
    frog = "frog"
    tdp = "tdp"
    dscan = "dscan"
    miips = "miips"
    ifrog = "ifrog"


class NonlinearProcess(str, enum.Enum):
    shg = "shg"
    xpw = "xpw"
    thg = "thg"
    sd = "sd"
    pg = "pg"


class Material(str, enum.Enum):
    fs = "FS"
    bk7 = "BK7"
    gratinga = "gratinga"
    gratingc = "gratingc"

    def get_coefficient(self, wedge_angle: float) -> float:
        if self in {Material.gratinga, Material.gratingc}:
            return 4.0

        if self in {Material.bk7, Material.fs}:
            return np.tan(wedge_angle * np.pi / 180) * np.cos(wedge_angle * np.pi / 360)

        raise ValueError("Unsupported material type")

    @property
    def pypret_material(self) -> pypret.material.BaseMaterial:
        """The pypret material."""
        return {
            Material.fs: pypret.material.FS,
            Material.bk7: pypret.material.BK7,
            Material.gratinga: pypret.material.gratinga,
            Material.gratingc: pypret.material.gratingc,
        }[self]


class RetrieverSolver(str, enum.Enum):
    copra = "copra"
    gpa = "gpa"
    gp_dscan = "gp-dscan"
    pcgpa = "pcgpa"
    pie = "pie"
    lm = "lm"
    bfgs = "bfgs"
    de = "de"
    nelder_mead = "nelder-mead"


class RetrievalResultStandin(SimpleNamespace):
    """
    The retriever returns a SimpleNamespace that contains (likely) all of
    these attributes.

    This class exists for type hinting purposes only.
    """

    parameter: Any
    options: Any
    logging: Any
    measurement: Any
    pnps: Optional[pypret.pnps.BasePNPS]
    # the pulse spectra
    # 1 - the retrieved pulse
    pulse_retrieved: Any
    # 2 - the original test pulse, optional
    pulse_original: Any
    # 3 - the initial guess
    pulse_initial: Any

    # the measurement traces
    # 1 - the original data used for retrieval
    trace_input: Any
    # 2 - the trace error and the trace calculated from the retrieved pulse
    trace_error: Any
    trace_retrieved: Any
    response_function: Any
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


def get_fundamental_spectrum(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    range_low: float,
    range_high: float,
):
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
        return self.wavelengths[idx], self.intensities[idx]

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
    def from_path(cls, path: str) -> SpectrumData:
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
    def from_file(cls, path: str) -> SpectrumData:
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
        return self.wavelengths[idx], self.intensities[:, idx]

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
    def from_path(cls, path: str) -> ScanData:
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


def _default_ndarray():
    """Helper for optional ndarray values in dataclass fields."""
    return np.zeros(0)


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

    def plot_processed_scan(self, *, scan_padding_nm: int = 75) -> pypret.MeshDataPlot:
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

    def _calculate_fwhm_and_profile(self):
        """
        Sets per-retrieval-parameter 'result_profile' and 'fwhm'.

        Side-effects:
        1. Sets ``self.result_profile``
        2. Sets ``self.fwhm``
        """
        pulse = self.pulse
        assert pulse is not None
        assert self.retrieval.pnps is not None
        result_parameter_mid_idx = np.floor(len(pulse.field) / 2) + 1
        self.result_profile = np.zeros(
            (len(pulse.field), len(self.retrieval.parameter))
        )
        self.fwhm = np.zeros((len(self.retrieval.parameter), 1))
        for idx, param in enumerate(self.retrieval.parameter):
            # Updating the spectrum property -> field gets updated
            pulse.spectrum = self.retrieval.pulse_retrieved * self.retrieval.pnps.mask(
                param
            )
            profile = np.power(np.abs(pulse.field), 2)[:]
            profile_max_idx = np.argmax(profile)
            self.result_profile[:, idx] = np.roll(
                profile, -round(profile_max_idx - result_parameter_mid_idx)
            )
            try:
                self.fwhm[idx] = pulse.fwhm(dt=pulse.dt / 100)
            except Exception:
                self.fwhm[idx] = np.nan

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
        fig = plt.plot(self.scan.positions * 1e3, self.fwhm * 1e15)
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

    @property
    def fund_intensities_bkg_sub(self) -> np.ndarray:
        """
        Fundamental intensities with background subtracted.
        """
        assert self.pulse is not None
        # TODO should this be applied to the end result?
        result_spec = scipy.interpolate.interp1d(
            self.pulse.wl,
            self.pulse.spectral_intensity,
            bounds_error=False,
            fill_value=0.0,
        )(self.fund.wavelengths)

        _, intensities = self.fund.subtract_background()
        intensities *= pypret.lib.best_scale(intensities, result_spec)
        return intensities

    def get_rms_error(self) -> float:
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
        return pypret.lib.nrms(self.fund_intensities_bkg_sub, result_spec)

    def _get_retriever(
        self, pulse: pypret.Pulse
    ) -> pypret.retrieval.retriever.BaseRetriever:
        """
        Get the pypret Retriever instance for the pulse.

        Side-effects:

        1. Sets self.trace_raw
        2. Sets self.trace

        Parameters
        ----------
        pulse : pypret.Pulse
            The pulse from the fundamental spectra.

        Returns
        -------
        pypret.Retriever
        """
        self.trace_raw = self._get_mesh_data()
        pnps = pypret.PNPS(
            pulse,
            method=self.method,
            process=self.nlin_process,
            material=self.material.pypret_material,
        )
        self.trace = preprocess(
            self.trace_raw,
            signal_range=(self.scan.wavelengths[0], self.scan.wavelengths[-1]),
            dark_signal_range=(0, 10),
        )

        if self.trace.units is not None and self.trace.units[1] == "m":
            # scaled in wavelength -> has to be corrected
            wavelength = cast(float, self.trace.axes[1])
            frequency = pypret.frequencies.convert(wavelength, "wl", "om")
            self.trace.scale(wavelength * wavelength)
            self.trace.normalize()
            self.trace.axes[1] = frequency
            self.trace.units[1] = "Hz"

        self.trace.interpolate(axis2=pnps.process_w)
        return pypret.Retriever(
            pnps,
            RetrieverSolver(self.solver).value,
            verbose=True,
            maxiter=self.max_iter,
        )

    def _get_retrieval_plot(
        self, plot_position: Optional[float] = None
    ) -> RetrievalResultPlot:
        """
        Get the "full" retrieval plot, with all pertinent information in one
        nice overview.

        Parameters
        ----------
        plot_position : float or None, optional
            Grating stage position for pypret plot OR glass insertion stage
            position (mm, use None for shortest duration)

        Returns
        -------
        RetrievalResultPlot
        """
        if plot_position is None:
            plot_param = self.retrieval.parameter[self.optimum_fwhm_idx]
            final_position = self.scan.positions[self.optimum_fwhm_idx]
        else:
            plot_param = self.retrieval.parameter[
                (np.nanargmin(np.abs(self.scan.positions - plot_position * 1e-3)))
            ]
            final_position = plot_position

        return RetrievalResultPlot(
            retrieval_result=self.retrieval,
            retrieval_parameter=plot_param,
            fund_range=self.spec_fund_range,
            scan_range=self.spec_scan_range,
            final_position=final_position,
            scan_positions=self.scan.positions,
            fundamental=self.fund_intensities_bkg_sub,
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
        verbose: bool = True,
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

        This method does a number of things:

        1. Truncates the fundamental spectrum data based on spec_fund_range
        2. Configures the pypret.FourierTransform based on freq_bandwidth_wl,
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
        fund = copy.deepcopy(fund)
        scan = copy.deepcopy(scan)
        result = PypretResult(
            fund=fund,
            scan=scan,
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
        fund.wavelengths, fund.intensities = fund.truncate_wavelength(
            range_low=result.spec_fund_range[0] * 1e-9,
            range_high=result.spec_fund_range[1] * 1e-9,
        )
        logger.info(f"Fundamental center wavelength: {fund.raw_center * 1e9:.1f} nm")

        ft = result._get_fourier_transform()
        logger.info(f"Time step = {ft.dt * 1e15:.2f} fs")

        result.pulse, result.fourier_transform_limit = fund._get_pulse(ft)
        result.scan.intensities = gaussian_filter(
            result.scan.intensities, sigma=blur_sigma
        )

        # Clean scan by truncating wavelength
        (
            result.scan.wavelengths,
            result.scan.intensities,
        ) = result.scan.truncate_wavelength(
            range_low=result.spec_scan_range[0] * 1e-9,
            range_high=result.spec_scan_range[1] * 1e-9,
        )

        result.scan.subtract_background_for_all_positions()

        retriever = result._get_retriever(result.pulse)
        pypret.random_gaussian(
            result.pulse, result.fourier_transform_limit, phase_max=0.1
        )

        # Experimental bimodal gaussian
        # pypret.random_bigaussian(result.pulse, result.fourier_transform_limit, phase_max=0.1, sep)
        retriever.retrieve(result.trace, result.pulse.spectrum, weights=None)
        result.retrieval = cast(RetrievalResultStandin, retriever.result())

        # Calculate the RMSE between retrieved and measured fundamental spectrum
        rms_error = result.get_rms_error()
        logger.info(f"RMS spectrum error: {rms_error}")

        # Find position of smallest FWHM
        result._calculate_fwhm_and_profile()
        result.plot = result._get_retrieval_plot(result.plot_position)

        if verbose:
            result.fund.plot(result.pulse)
            result.plot_mesh_data()
            result.plot_fwhm_vs_grating_position()
            result.plot_temporal_profile_vs_grating_position()
            result.plot.plot(
                oversampling=8,
                phase_blanking=True,
                phase_blanking_threshold=0.01,
                limit=True,
            )
            plt.show()

        return result
