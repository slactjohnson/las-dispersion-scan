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
import scipy.interpolate
from scipy.ndimage import gaussian_filter

from .plotting import RetrievalResultPlot
from .utils import preprocess, preprocess2, pulse_from_spectrum

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


class RetrievalResultStandin(SimpleNamespace):
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

    def get_pulse_from_spectrum(self, ft: pypret.FourierTransform) -> pypret.Pulse:
        _, _, fund_intensities_bkg_sub = self.subtract_background()
        return pulse_from_spectrum(
            self.wavelengths,
            fund_intensities_bkg_sub,
            pulse=pypret.Pulse(ft, self.raw_center),
        )

    def get_fourier_transform_limit(self, ft: pypret.FourierTransform) -> float:
        pulse = self.get_pulse_from_spectrum(ft)
        return pulse.fwhm(dt=pulse.dt / 100)

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
        wavelength_bkg = np.hstack(
            (self.wavelengths[:count], self.wavelengths[-count:])
        )
        intensities_bkg = np.hstack(
            (self.intensities[:count], self.intensities[-count:])
        )
        return wavelength_bkg, intensities_bkg

    def subtract_background(self, *, count: int = 15):
        wavelength_bkg, intensities_bkg = self.get_background(count=count)
        fit = np.polyfit(wavelength_bkg, intensities_bkg, 1)
        intensities_bkg_fit = self.wavelengths * fit[0] + fit[1]
        intensities_bkg_sub = self.intensities - intensities_bkg_fit
        intensities_bkg_sub /= np.max(intensities_bkg_sub)
        intensities_bkg_sub[intensities_bkg_sub < 0.0025] = 0
        return (
            fit,
            intensities_bkg_fit,
            intensities_bkg_sub,
        )

    def plot(self, pulse: Optional[pypret.Pulse] = None):
        fig, ax = plt.subplots()
        wavelength_bkg, intensities_bkg = self.get_background(count=15)
        _, intensities_bkg_fit, _ = self.subtract_background()

        plt.plot(self.wavelengths * 1e9, self.intensities, "k", label="All data")
        plt.plot(
            wavelength_bkg * 1e9,
            intensities_bkg,
            "xb",
            label="Points for fit",
        )
        plt.plot(
            self.wavelengths * 1e9,
            intensities_bkg_fit,
            "r",
            label="Background fit",
        )
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Counts (arb.)")

        if pulse is not None:
            ftl = pulse.fwhm(dt=pulse.dt / 100)
            plt.title(f"Fundamental Spectrum (FTL) = {ftl * 1e15:.1f} fs")

        return fig, ax


@dataclasses.dataclass
class ScanData:
    #: Scan positions
    positions: np.ndarray
    #: Wavelengths
    wavelengths: np.ndarray
    #: Normalized intensities
    intensities: np.ndarray

    def subtract_background_for_all_positions(self) -> None:
        """
        Clean scan by subtracting linear background for each stage position.

        Works in-place.
        """
        scan_wavelength_bkg = np.hstack((self.wavelengths[:15], self.wavelengths[-15:]))
        for i in range(len(self.positions)):
            scan_intensities_bkg = np.hstack(
                (self.intensities[i, :15], self.intensities[i, -15:])
            )
            p = np.polyfit(scan_wavelength_bkg, scan_intensities_bkg, 1)
            self.intensities[i, :] -= self.wavelengths * p[0] + p[1]

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
    fund: SpectrumData
    scan: ScanData
    material: Material
    method: PulseAnalysisMethod
    nlin_process: NonlinearProcess
    wedge_angle: float = 8.0
    blur_sigma: int = 0
    grid_points: int = 3000
    freq_bandwidth_wl: int = 950
    max_iter: int = 30
    plot_position: Optional[float] = None
    spec_fund_range: Tuple[float, float] = (400, 600)
    spec_scan_range: Tuple[float, float] = (200, 300)
    plot: Optional[RetrievalResultPlot] = None
    retrieval: RetrievalResultStandin = dataclasses.field(
        default_factory=RetrievalResultStandin
    )
    pulse: Optional[pypret.Pulse] = None
    trace_raw: Optional[pypret.MeshData] = None
    trace: Optional[pypret.MeshData] = None
    fwhm: np.ndarray = dataclasses.field(default_factory=_default_ndarray)
    result_profile: np.ndarray = dataclasses.field(default_factory=_default_ndarray)
    result_parameter: np.ndarray = dataclasses.field(default_factory=_default_ndarray)
    fourier_transform_limit: float = 0.0

    def get_fourier_transform(self) -> pypret.FourierTransform:
        # Create frequency-time grid
        freq_bandwidth = (
            self.freq_bandwidth_wl
            * 1e-9
            * 2
            * np.pi
            * 2.998e8
            / self.fund.raw_center**2
        )
        fund_frequency_step = np.round(freq_bandwidth / (self.grid_points - 1), 0)
        return pypret.FourierTransform(
            self.grid_points, dw=fund_frequency_step, w0=-freq_bandwidth / 2
        )

    def get_mesh_data(self) -> pypret.MeshData:
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
        if data is None:
            data = self.get_mesh_data()

        md = pypret.MeshDataPlot(data, show=False)
        ax = cast(plt.Axes, md.ax)
        ax.set_title("Cropped scan")
        ax.set_xlim(
            [
                (self.spec_scan_range[0] + scan_padding_nm) * 1e-9,
                (self.spec_scan_range[1] - scan_padding_nm) * 1e-9,
            ]
        )
        return md

    def plot_processed_scan(self, *, scan_padding_nm: int = 75):
        md = pypret.MeshDataPlot(self.trace, show=False)
        ax = cast(plt.Axes, md.ax)
        ax.set_title("Processed scan")
        ax.set_xlabel("Frequency")
        ax.set_xlim(
            [
                2
                * np.pi
                * 2.99792
                * 1e17
                / (self.spec_scan_range[1] - scan_padding_nm),
                2
                * np.pi
                * 2.99792
                * 1e17
                / (self.spec_scan_range[0] + scan_padding_nm),
            ]
        )
        return md

    def _calculate_fwhm_and_profile(self):
        pulse = self.pulse
        assert pulse is not None
        assert self.retrieval.pnps is not None
        result_parameter = self.retrieval.parameter
        result_parameter_mid_idx = np.floor(len(pulse.field) / 2) + 1
        result_profile = np.zeros((len(pulse.field), len(result_parameter)))
        fwhm = np.zeros((len(result_parameter), 1))
        for idx, param in enumerate(result_parameter):
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

        self.fwhm = fwhm
        self.result_profile = result_profile
        self.result_parameter = result_parameter

    @property
    def optimum_fwhm_idx(self) -> np.ndarray:
        return np.nanargmin(self.fwhm)

    @property
    def optimum_fwhm(self) -> float:
        return self.fwhm[self.optimum_fwhm_idx][0]

    def plot_fwhm_vs_grating_position(self):
        fig = plt.figure()
        ax = cast(plt.Axes, fig.add_subplot(111))
        fig = plt.plot(self.scan.positions * 1e3, self.fwhm * 1e15)
        ax.tick_params(labelsize=12)
        plt.xlabel("Position (mm)")
        plt.ylabel("FWHM (fs)")

        result_optimum_fwhm = self.optimum_fwhm
        fwhm0 = result_optimum_fwhm * 1e15
        pos = self.scan.positions[self.optimum_fwhm_idx] * 1e3
        plt.title(f"Shortest: {fwhm0:.1f} fs @ {pos:.3f} mm")
        plt.ylim(
            0,
            min([np.nanmax(self.fwhm * 1e15), 4 * result_optimum_fwhm * 1e15]),
        )
        return fig

    def plot_temporal_profile_vs_grating_position(self):
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
        plt.xlabel("Time (fs)")
        plt.ylabel("Position (mm)")
        plt.title("Dscan Temporal Profile")
        plt.xlim(-8 * self.optimum_fwhm * 1e15, 8 * self.optimum_fwhm * 1e15)
        return fig

    @property
    def fund_intensities_bkg_sub(self) -> np.ndarray:
        assert self.pulse is not None
        result_spec = scipy.interpolate.interp1d(
            self.pulse.wl,
            self.pulse.spectral_intensity,
            bounds_error=False,
            fill_value=0.0,
        )(self.fund.wavelengths)

        _, _, intensities = self.fund.subtract_background()
        intensities *= pypret.lib.best_scale(intensities, result_spec)
        return intensities

    def get_rms_error(self) -> float:
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
        assert self.trace_raw is not None
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
        preprocess2(self.trace, pnps)

        # Pypret retrieval
        return pypret.Retriever(pnps, "copra", verbose=True, maxiter=self.max_iter)

    def _get_retrieval_plot(
        self, plot_position: Optional[float] = None
    ) -> RetrievalResultPlot:
        # Plot results (Pypret style)
        if plot_position is None:
            plot_param = self.result_parameter[self.optimum_fwhm_idx]
            final_position = self.scan.positions[self.optimum_fwhm_idx]
        else:
            plot_param = self.result_parameter[
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
        grid_points: int = 3000,
        freq_bandwidth_wl: int = 950,
        max_iter: int = 30,
        plot_position: Optional[float] = None,
        spec_fund_range: Tuple[float, float] = (400, 600),
        spec_scan_range: Tuple[float, float] = (200, 300),
    ) -> PypretResult:
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
            grid_points=grid_points,
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

        ft = result.get_fourier_transform()
        logger.info(f"Time step = {ft.dt * 1e15:.2f} fs")

        result.pulse = fund.get_pulse_from_spectrum(ft)
        result.fourier_transform_limit = result.pulse.fwhm(dt=result.pulse.dt / 100)

        logger.info(
            f"Fourier Transform Limit (FTL): {result.fourier_transform_limit * 1e15:.1f} fs"
        )

        scan.intensities = gaussian_filter(scan.intensities, sigma=blur_sigma)

        # Clean scan by truncating wavelength
        scan.wavelengths, scan.intensities = scan.truncate_wavelength(
            range_low=result.spec_scan_range[0] * 1e-9,
            range_high=result.spec_scan_range[1] * 1e-9,
        )

        scan.subtract_background_for_all_positions()

        result.trace_raw = result.get_mesh_data()
        trace_raw = result.get_mesh_data()

        if verbose:
            result.plot_mesh_data(trace_raw)

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

        # Plot FWHM vs grating position
        if verbose:
            result.plot_fwhm_vs_grating_position()

        # Plot temporal profile vs grating position
        if verbose:
            result.plot_temporal_profile_vs_grating_position()

        result.plot = result._get_retrieval_plot(result.plot_position)

        if verbose:
            result.plot.plot(
                oversampling=8,
                phase_blanking=True,
                phase_blanking_threshold=0.01,
                limit=True,
            )
            plt.show()

        return result
