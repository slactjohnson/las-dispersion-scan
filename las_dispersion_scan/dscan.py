import enum
import os
import sys
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pypret
import scipy
from scipy.ndimage import gaussian_filter

from .utils import pulse_from_spectrum, preprocess, preprocess2
from .plotting import RetrievalResultPlot
from typing import Optional, Tuple


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
            return 4.

        if self in {Material.bk7, Material.fs}:
            return np.tan(wedge_angle * np.pi / 180) * np.cos(
                wedge_angle * np.pi / 360
            )

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
    

def run_general_scan():
    ### OPTIONS AND PARAMETERS ###

    ## Method Parameters (see line 455 for documentation in pnps)
    # Set wavelength (optional)
    wavelength_fund = 490  # 800 490
    # Select method of analyzing pulse
    method_pick = PulseAnalysisMethod.dscan
    # Select nonlinear process
    nlin_process = NonlinearProcess.shg
    # Select the material (for the d-scan)
    material_pick = Material.bk7

    # Collect fundamental?
    take_fund = False
    # Collect d-scan?
    take_scan = False
    # Run pypret?
    run_pypret = True
    # Use Thorlabs stage to automatically collect fundamental?
    auto_fund = True
    # Preview spectra?
    preview = True

    ## Filepaths
    # Data directory
    path_data = r"/Users/aaronghrist/Research/Code/Python Code/general_dscan_v0.0.6/Data/XCS/2022_08_15/Dscan_7"
    # path_data = r'/Users/aaronghrist/Research/Code/Python Code/general_dscan_v0.0.6/Data/CXI/2022_05_10/amplifier_dscan_CXI_20220510_2'
    # path_data = r'/Users/aaronghrist/Research/Code/Python Code/general_dscan_v0.0.6/Data/XCS/2022_06_29/Dscan_run'
    # path_data = r'/Users/aaronghrist/Research/Code/Python Code/general_dscan_v0.0.6/Data/Varian/2022_07_23/GSCAN2'
    # Fundamental subpath
    path_sub_fund = r"fund"
    # d-scan subpath
    path_sub_scan = r"scan"
    # pickle subpath
    path_sub_pickle = r"out1"

    ## Conversion from previously collected data
    # Convert previously collected .txt files to .dat?
    update_txt = False
    # Old format fundamental subpath
    path_sub_fund_old = r"fund_conv"
    # Old format d-scan subpath
    path_sub_scan_old = r"dscan_conv"

    ## Spectrometer settings
    # Fundamental integration time (ms)
    spec_time_fund = 20000
    # Fundamental averages
    spec_avgs_fund = 30
    # Scan integration time (ms)
    spec_time_scan = 250000
    # Scan averages
    spec_avgs_scan = 4

    ## Spectrum window edge parameters
    # Fundamental spectrum window
    spec_fund_range = (400, 600)  # [650, 950] [400, 600]
    # Scan spectrum window
    spec_scan_range = (200, 300)  # [300, 500] [200, 300]

    ## Newport stage parameters
    # Center stage position of d-scan
    scan_pos_center = 0.1
    # Amplitude of scan
    scan_amplitude_negative = 0.50
    scan_amplitude_positive = 0.50
    # Number of points in d-scan
    scan_points = 100
    # Wedge angle (for traditional d-scan, not grating)
    wedge_angle = 8  # (degrees)
    # COM port
    ESP_COM_port = "COM8"
    # Axis of stage on controller
    ESP_axis_stage_grating = 1

    ## Thorlabs stage parameters (for automatic fundamental collection)
    # Home actuator initially?
    auto_fund_home = False
    # Standby position
    auto_fund_pos_standby = 0
    # Fundamental measurement position
    auto_fund_pos_fund = 13.3
    # d-scan measurement position
    auto_fund_pos_scan = 0.5
    # Controller ID number
    auto_fund_ID = 27250600

    ## Pypret parameters
    # Include all pypret plots?
    verbose_pypret = True
    # Strength of Gaussian blur applied to raw data (standard deviations)
    pypret_blur_sigma = 0
    # Number of grid points in frequency and time axes
    pypret_grid_points = 3000  # 3000
    # Bandwidth around center wavelength for frequency and time axes (nm)
    pypret_grid_bandwidth_wl = 950  # 1500 950
    # Maximum number of iterations
    pypret_max_iter = 30  # 32
    # Grating stage position for pypret plot OR glass insertion stage position (mm, use None for shortest duration)
    pypret_plot_position = None  # 1.2 (mm)
    # Move dscan stage to position indicated above?
    final_stage_move = False

    ###########

    path_full_fund = os.path.join(path_data, path_sub_fund + ".dat")
    path_full_scan = os.path.join(path_data, path_sub_scan + ".dat")
    path_full_pickle = os.path.join(path_data, path_sub_pickle + ".pickle")
    old_path_fund = os.path.join(path_data, path_sub_fund_old + ".txt")
    old_path_scan = os.path.join(path_data, path_sub_scan_old + ".txt")

    scan = PrototypeScan(
        wavelength_fund=wavelength_fund,
        method_pick=method_pick,
        nlin_process=nlin_process,
        material_pick=material_pick,
        take_fund=take_fund,
        auto_fund=auto_fund,
        take_scan=take_scan,
        run_pypret=run_pypret,
        path_full_fund=path_full_fund,
        path_full_scan=path_full_scan,
        path_full_pickle=path_full_pickle,
        old_path_fund=old_path_fund,
        old_path_scan=old_path_scan,
        update_txt=update_txt,
        final_stage_move=final_stage_move,
        wedge_angle=wedge_angle,
        preview=preview,
        verbose_pypret=verbose_pypret,
        spec_time_fund=spec_time_fund,
        spec_avgs_fund=spec_avgs_fund,
        spec_time_scan=spec_time_scan,
        spec_avgs_scan=spec_avgs_scan,
        spec_fund_range=spec_fund_range,
        spec_scan_range=spec_scan_range,
        scan_pos_center=scan_pos_center,
        scan_points=scan_points,
        scan_amplitude_negative=scan_amplitude_negative,
        scan_amplitude_positive=scan_amplitude_positive,
        ESP_COM_port=ESP_COM_port,
        ESP_axis_stage_grating=ESP_axis_stage_grating,
        auto_fund_home=auto_fund_home,
        auto_fund_ID=auto_fund_ID,
        auto_fund_pos_standby=auto_fund_pos_standby,
        auto_fund_pos_fund=auto_fund_pos_fund,
        auto_fund_pos_scan=auto_fund_pos_scan,
        pypret_blur_sigma=pypret_blur_sigma,
        pypret_grid_points=pypret_grid_points,
        pypret_grid_bandwidth_wl=pypret_grid_bandwidth_wl,
        pypret_max_iter=pypret_max_iter,
        pypret_plot_position=pypret_plot_position,
    )
    scan.run()

    if scan.take_scan or scan.take_fund:
        import pickle

        scan.ESP_stage_device = None
        scan.spec_fund = None
        scan.spec_scan = None
        if not os.path.exists(os.path.dirname(scan.path_full_pickle)):
            os.makedirs(os.path.dirname(scan.path_full_pickle))
        with open(scan.path_full_pickle, "wb") as handle:
            pickle.dump(scan, handle)


class PrototypeScan:
    def __init__(
        self,
        wavelength_fund: int,
        method_pick: PulseAnalysisMethod,
        nlin_process: NonlinearProcess,
        material_pick: Material,
        take_fund: bool,
        auto_fund: bool,
        take_scan: bool,
        run_pypret: bool,
        path_full_fund: str,
        path_full_scan: str,
        path_full_pickle: str,
        old_path_fund: str,
        old_path_scan: str,
        update_txt: bool,
        final_stage_move: bool,
        wedge_angle: int,
        preview: bool = True,
        verbose_pypret: bool = True,
        spec_time_fund: int = 20_000,
        spec_avgs_fund: int = 30,
        spec_time_scan: int = 250_000,
        spec_avgs_scan: int = 4,
        spec_fund_range: Optional[Tuple[int, int]] = (400, 600),
        spec_scan_range: Optional[Tuple[int, int]] = (200, 300),
        scan_pos_center: float = 0.1,
        scan_points: int = 100,
        scan_amplitude_negative: float = 0.5,
        scan_amplitude_positive: float = 0.5,
        ESP_COM_port: str = "COM1",
        ESP_axis_stage_grating: int = 1,
        auto_fund_home: bool = False,
        auto_fund_ID: int = 27_250_600,
        auto_fund_pos_standby: int = 0,
        auto_fund_pos_fund: float = 13.3,
        auto_fund_pos_scan: float = 0.5,
        pypret_blur_sigma: int = 0,
        pypret_grid_points: int = 3000,
        pypret_grid_bandwidth_wl: int = 950,
        pypret_max_iter: int = 30,
        pypret_plot_position: Optional[float] = None,
    ):
        self.material_pick = material_pick
        self.take_fund = take_fund
        self.auto_fund = auto_fund
        self.take_scan = take_scan
        self.run_pypret = run_pypret
        self.path_full_fund = path_full_fund
        self.path_full_scan = path_full_scan
        self.update_txt = update_txt
        self.wedge_angle = wedge_angle
        self.path_full_pickle = path_full_pickle
        self.exists_fund = os.path.exists(path_full_fund)
        self.exists_scan = os.path.exists(path_full_scan)

        if self.take_fund:
            if None in [preview, spec_time_fund, spec_avgs_fund, spec_fund_range]:
                sys.exit("Exit: Undefined fundamental parameter(s)")
            self.preview = preview
            self.spec_time_fund = spec_time_fund
            self.spec_avgs_fund = spec_avgs_fund
            self.spec_fund_range = spec_fund_range

        if self.auto_fund:
            if None in [
                auto_fund_home,
                auto_fund_ID,
                auto_fund_pos_standby,
                auto_fund_pos_fund,
                auto_fund_pos_scan,
            ]:
                sys.exit("Exit: Undefined auto_fund parameter(s)")
            self.auto_fund_home = auto_fund_home
            self.auto_fund_ID = auto_fund_ID
            self.auto_fund_pos_standby = auto_fund_pos_standby
            self.auto_fund_pos_fund = auto_fund_pos_fund
            self.auto_fund_pos_scan = auto_fund_pos_scan

        if self.take_scan:
            if None in [
                preview,
                spec_time_scan,
                spec_avgs_scan,
                spec_scan_range,
                scan_pos_center,
                scan_points,
                scan_amplitude_negative,
                scan_amplitude_positive,
                ESP_COM_port,
                ESP_axis_stage_grating,
            ]:
                sys.exit("Exit: Undefined scan parameter(s)")
            self.preview = preview
            self.spec_time_scan = spec_time_scan
            self.spec_avgs_scan = spec_avgs_scan
            self.spec_fund_range = spec_fund_range
            self.scan_pos_center = scan_pos_center
            self.scan_points = scan_points
            self.scan_amplitude_negative = scan_amplitude_negative
            self.scan_amplitude_positive = scan_amplitude_positive
            self.ESP_COM_port = ESP_COM_port
            self.ESP_axis_stage_grating = ESP_axis_stage_grating

        if self.run_pypret:
            if None in [
                pypret_blur_sigma,
                pypret_grid_points,
                pypret_grid_bandwidth_wl,
                pypret_max_iter,
                verbose_pypret,
            ]:
                sys.exit("Exit: Undefined pypret parameter(s)")
            self.wavelength_fund = wavelength_fund
            self.method_pick = method_pick
            self.nlin_process = nlin_process
            self.pypret_blur_sigma = pypret_blur_sigma
            self.pypret_grid_points = pypret_grid_points
            self.pypret_freq_bandwidth_wl = pypret_grid_bandwidth_wl
            self.pypret_max_iter = pypret_max_iter
            self.pypret_plot_position = pypret_plot_position
            self.spec_fund_range = spec_fund_range
            self.spec_scan_range = spec_scan_range
            self.update_txt = update_txt
            self.verbose_pypret = verbose_pypret
            self.old_path_fund = old_path_fund
            self.old_path_scan = old_path_scan
            self.final_stage_move = final_stage_move

    def run(self):

        if self.take_fund or self.take_scan:

            ## Initialize actuator
            if self.auto_fund:
                import thorlabs_apt as apt

                print("Available Thorlabs actuators:")
                print(apt.list_available_devices())
                motor = apt.Motor(self.auto_fund_ID)
                motor.set_stage_axis_info(0, 15, 1, 0.5)
                if self.auto_fund_home:
                    print("Homing Thorlabs actuator")
                    motor.move_home(True)

            ## Initialize spectrometer and ESP stage
            if self.take_fund:
                if self.exists_fund:
                    sys.exit("Exit: Fundamental spectrum file already exists")
            if self.take_scan:
                if self.exists_scan:
                    sys.exit("Exit: d-scan file already exists")

                self.ESP_stage_device = esp.ESP_stage(self.ESP_COM_port)
                if self.scan_pos_center is None:
                    self.scan_pos_center = self.ESP_stage_device.position(
                        self.ESP_axis_stage_grating, True
                    )
                self.scan_points_list = np.linspace(
                    self.scan_pos_center - self.scan_amplitude_negative,
                    self.scan_pos_center + self.scan_amplitude_positive,
                    num=self.scan_points,
                )
            specs = oo.spectrometers()

        ## Collect fundamental
        if self.take_fund:
            if self.auto_fund:
                try:
                    motor.move_to(self.auto_fund_pos_fund, True)
                except:
                    motor.move_to(self.auto_fund_pos_fund, False)
                    sleep(15)
            self.spec_fund = oo.spectrum(
                specs.devices,
                integration_time=self.spec_time_fund,
                averages=self.spec_avgs_fund,
            )
            if self.preview:
                print("Check fundamental spectrum. Close window to continue.")
                self.spec_fund.plot_live(wavelength_limits=self.spec_fund_range)
                print("Continuing...")
            self.spec_fund.collect_intensities()
            wavelength = self.spec_fund.wavelength
            intensities = self.spec_fund.intensities
            self.wavelength_fund = wavelength[
                (wavelength > self.spec_fund_range[0])
                & (wavelength < self.spec_fund_range[1])
            ]
            self.intensities_fund = intensities[
                (wavelength > self.spec_fund_range[0])
                & (wavelength < self.spec_fund_range[1])
            ]
            fundamental_spectrum = np.column_stack(
                (self.wavelength_fund, self.intensities_fund)
            )
            if not os.path.exists(os.path.dirname(self.path_full_fund)):
                os.makedirs(os.path.dirname(self.path_full_fund))
            np.savetxt(self.path_full_fund, fundamental_spectrum)

        ## Collect d-scan
        if self.take_scan:
            if self.auto_fund:
                try:
                    motor.move_to(self.auto_fund_pos_scan, True)
                except:
                    motor.move_to(self.auto_fund_pos_scan, False)
                    sleep(15)
            self.ESP_stage_device.move_to(
                self.ESP_axis_stage_grating, self.scan_pos_center, True
            )
            self.spec_scan = oo.spectrum(
                specs.devices,
                integration_time=self.spec_time_scan,
                averages=self.spec_avgs_scan,
            )
            if self.preview:
                print("Check SHG spectrum. Close window to continue.")
                self.spec_scan.plot_live(wavelength_limits=self.spec_scan_range)
                print("Continuing...")
            self.scan_position_list = []
            for counter, pos in enumerate(self.scan_points_list):
                self.ESP_stage_device.move_to(self.ESP_axis_stage_grating, pos, True)
                self.scan_position_list.append(
                    self.ESP_stage_device.position(self.ESP_axis_stage_grating)
                )
                self.spec_scan.collect_intensities()
                if counter == 0:
                    wavelength = self.spec_scan.wavelength
                    self.wavelength_scan = wavelength[
                        (wavelength > self.spec_scan_range[0])
                        & (wavelength < self.spec_scan_range[1])
                    ]
                    wavelength_cut = self.wavelength_scan
                    scan_data = np.insert(wavelength_cut, 0, np.nan)
                intensities = self.spec_scan.intensities
                intensities_cut = intensities[
                    (wavelength > self.spec_scan_range[0])
                    & (wavelength < self.spec_scan_range[1])
                ]
                intensities_cut_pos = np.insert(
                    intensities_cut, 0, self.scan_position_list[counter]
                )
                scan_data = np.column_stack((scan_data, intensities_cut_pos))
            if not os.path.exists(os.path.dirname(self.path_full_scan)):
                os.makedirs(os.path.dirname(self.path_full_scan))
            np.savetxt(self.path_full_scan, scan_data)
            self.ESP_stage_device.move_to(
                self.ESP_axis_stage_grating, self.scan_pos_center, True
            )
            scan_intensities = scan_data[1:, 1:]
            scan_positions = scan_data[0, 1:]
            scan_wavelength = scan_data[1:, 0]
            plt.close("all")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig = plt.contourf(scan_positions, scan_wavelength, scan_intensities.T, 100)
            ax.set_ylabel("Grating position (mm)", size=12)
            ax.set_xlabel("Wavelength (nm)", size=12)
            ax.tick_params(labelsize=12)
            plt.show()

        ## Return actuator to standy position
        if self.take_scan or self.take_fund:
            if self.auto_fund:
                try:
                    motor.move_to(self.auto_fund_pos_standby, True)
                except:
                    motor.move_to(self.auto_fund_pos_standby, False)
                    sleep(15)

        # Convert from .txt (old format) to .dat (new format)
        if self.update_txt:
            self.exists_old_fund_post = os.path.exists(self.old_path_fund)
            self.exists_old_scan_post = os.path.exists(self.old_path_scan)
            if not self.exists_old_fund_post:
                sys.exit("Exit: Fundamental spectrum missing (.txt conversion)")
            if not self.exists_old_fund_post:
                sys.exit("Exit: Scan spectra missing (.txt conversion)")

            old_fund = np.loadtxt(self.old_path_fund)
            np.savetxt(self.path_full_fund, old_fund)
            old_scan = np.loadtxt(self.old_path_scan)
            M = np.empty([len(old_scan[0, 1:]), len(old_scan[:, 0])])
            M[0, 1:] = old_scan[1:, 0].transpose()
            M[1:, 0] = old_scan[0, 2:].transpose()
            M[1:, 1:] = old_scan[1:, 2:].transpose()
            np.savetxt(self.path_full_scan, M)

        ## Run pypret
        if self.run_pypret:
            # Initialization

            self.exists_fund_post = os.path.exists(self.path_full_fund)
            self.exists_scan_post = os.path.exists(self.path_full_scan)
            if not self.exists_fund_post:
                sys.exit("Exit: Fundamental spectrum missing (Pypret)")
            if not self.exists_fund_post:
                sys.exit("Exit: Scan spectra missing (Pypret)")

            # Load fundamental
            fund_data = np.loadtxt(self.path_full_fund)
            fund_wavelength = fund_data[:, 0] * 1e-9
            fund_intensities = fund_data[:, 1]
            fund_intensities -= np.average(fund_intensities[:7])
            # fund_intensities *= fund_wavelength * fund_wavelength

            # Clean fundamental by truncating wavelength
            fund_wavelength_idx = (fund_wavelength > self.spec_fund_range[0] * 1e-9) & (
                fund_wavelength < self.spec_fund_range[1] * 1e-9
            )
            fund_wavelength = fund_wavelength[fund_wavelength_idx]
            fund_intensities = fund_intensities[fund_wavelength_idx]
            wavelength_raw_center = sum(
                np.multiply(fund_wavelength, fund_intensities)
            ) / sum(fund_intensities)
            # wavelength_raw_center = self.wavelength_fund * 1E-9
            print(f"Fundamental center wavelength: {wavelength_raw_center * 1e9:.1f} nm")

            # Create frequency-time grid
            freq_bandwidth = (
                self.pypret_freq_bandwidth_wl
                * 1e-9
                * 2
                * np.pi
                * 2.998e8
                / wavelength_raw_center ** 2
            )
            fund_frequency_step = np.round(
                freq_bandwidth / (self.pypret_grid_points - 1), 0
            )
            ft = pypret.FourierTransform(
                self.pypret_grid_points, dw=fund_frequency_step, w0=-freq_bandwidth / 2
            )
            print(f"Time step = {ft.dt * 1e15:.2f} fs")
            pulse = pypret.Pulse(ft, wavelength_raw_center)

            # Subtract background
            fund_wavelength_bkg = np.hstack(
                (fund_wavelength[:15], fund_wavelength[-15:])
            )
            fund_intensities_bkg = np.hstack(
                (fund_intensities[:15], fund_intensities[-15:])
            )
            p = np.polyfit(fund_wavelength_bkg, fund_intensities_bkg, 1)
            fund_intensities_bkg_fit = fund_wavelength * p[0] + p[1]
            fund_intensities_bkg_sub = fund_intensities - fund_intensities_bkg_fit
            fund_intensities_bkg_sub /= np.max(fund_intensities_bkg_sub)
            fund_intensities_bkg_sub[fund_intensities_bkg_sub < 0.0025] = 0

            # Fourier limit
            pulse = pulse_from_spectrum(
                fund_wavelength, fund_intensities_bkg_sub, pulse=pulse
            )
            FTL = pulse.fwhm(dt=pulse.dt / 100)
            print(f"Fourier Transform Limit (FTL): {FTL * 1e15:.1f} fs")

            # Plot fundamental
            if self.verbose_pypret:
                fig, ax = plt.subplots()
                plt.plot(fund_wavelength * 1e9, fund_intensities, "k", label="All data")
                plt.plot(
                    fund_wavelength_bkg * 1e9,
                    fund_intensities_bkg,
                    "xb",
                    label="Points for fit",
                )
                plt.plot(
                    fund_wavelength * 1e9,
                    fund_intensities_bkg_fit,
                    "r",
                    label="Background fit",
                )
                plt.legend()
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Counts (arb.)")
                plt.title(f"Fundamental Spectrum (FTL) = {FTL * 1e15:.1f} fs")
                plt.show()

            # Load scan
            scan_data = np.loadtxt(self.path_full_scan)
            scan_positions = scan_data[0, 1:] * 1e-3
            scan_wavelength = scan_data[1:, 0] * 1e-9
            scan_intensities = scan_data[1:, 1:].transpose()
            scan_intensities /= np.amax(scan_intensities)
            scan_intensities = gaussian_filter(
                scan_intensities, sigma=self.pypret_blur_sigma
            )

            # Clean scan by truncating wavelength
            scan_wavelength_idx = (scan_wavelength > self.spec_scan_range[0] * 1e-9) & (
                scan_wavelength < self.spec_scan_range[1] * 1e-9
            )
            scan_wavelength = scan_wavelength[scan_wavelength_idx]
            scan_intensities = scan_intensities[:, scan_wavelength_idx]

            # Clean scan by subtracting linear background for each stage position
            scan_wavelength_bkg = np.hstack(
                (scan_wavelength[:15], scan_wavelength[-15:])
            )
            for i in range(len(scan_positions)):
                scan_intensities_bkg = np.hstack(
                    (scan_intensities[i, :15], scan_intensities[i, -15:])
                )
                p = np.polyfit(scan_wavelength_bkg, scan_intensities_bkg, 1)
                scan_intensities[i, :] -= scan_wavelength * p[0] + p[1]

            # Defines the proper conversion from stage position
            method_coef = self.material_pick.get_coefficient()

            trace_raw = pypret.MeshData(
                scan_intensities,
                method_coef * (scan_positions - min(scan_positions)),
                scan_wavelength,
                labels=["Insertion", "Wavelength"],
                units=["m", "m"],
            )
            scan_padding = 75  # (nm)
            if self.verbose_pypret:
                md = pypret.MeshDataPlot(trace_raw, show=False)
                md.ax.set_title("Cropped scan")
                md.ax.set_xlim(
                    [
                        (self.spec_scan_range[0] + scan_padding) * 1e-9,
                        (self.spec_scan_range[1] - scan_padding) * 1e-9,
                    ]
                )
                md.show()

            # Pypret initialization and processing
            method_set = {"frog", "tdp", "dscan", "miips", "ifrog"}
            nlin_set = {"shg", "xpw", "thg", "sd", "pg"}
            material_set = {"FS", "BK7", "gratinga", "gratingc"}
            if (
                self.method_pick not in method_set
                or self.nlin_process not in nlin_set
                or self.material_pick not in material_set
            ):
                sys.exit("Undefined or incorrect pypret retrieval attributes")

            pnps = pypret.PNPS(
                pulse,
                method=self.method_pick,
                process=self.nlin_process,
                material=Material(self.material_pick).pypret,
            )
            trace = preprocess(
                trace_raw,
                signal_range=(scan_wavelength[0], scan_wavelength[-1]),
                dark_signal_range=(0, 10),
            )
            preprocess2(trace, pnps)
            if self.verbose_pypret:
                md = pypret.MeshDataPlot(trace, show=False)
                md.ax.set_title("Processed scan")
                md.ax.set_xlabel("Frequency")
                md.ax.set_xlim(
                    [
                        2
                        * np.pi
                        * 2.99792
                        * 1e17
                        / (self.spec_scan_range[1] - scan_padding),
                        2
                        * np.pi
                        * 2.99792
                        * 1e17
                        / (self.spec_scan_range[0] + scan_padding),
                    ]
                )
                md.show()

            # Pypret retrieval
            ret = pypret.Retriever(
                pnps, "copra", verbose=True, maxiter=self.pypret_max_iter
            )
            pypret.random_gaussian(pulse, FTL, phase_max=0.1)
            # pypret.random_bigaussian(pulse, FTL, phase_max=0.1, sep)  # Experimental bimodal gaussian
            ret.retrieve(trace, pulse.spectrum, weights=None)
            result = ret.result()
            # Calculate the RMSE between retrieved and measured fundamental spectrum
            result_spec = pulse.spectral_intensity
            result_spec = scipy.interpolate.interp1d(
                pulse.wl, result_spec, bounds_error=False, fill_value=0.0
            )(fund_wavelength)
            fund_intensities_bkg_sub *= pypret.lib.best_scale(
                fund_intensities_bkg_sub, result_spec
            )
            rms_error = pypret.lib.nrms(fund_intensities_bkg_sub, result_spec)
            print(f"RMS spectrum error: {rms_error}")

            # Find position of smallest FWHM
            result_parameter = result.parameter
            result_parameter_mid_idx = np.floor(len(pulse.field) / 2) + 1
            result_profile = np.zeros((len(pulse.field), len(result_parameter)))
            result_FWHM = np.zeros((len(result_parameter), 1))
            for i, p in enumerate(result_parameter):
                pulse.spectrum = result.pulse_retrieved * result.pnps.mask(p)
                profile = np.power(np.abs(pulse.field), 2)[:]
                profile_max_idx = np.argmax(profile)
                result_profile[:, i] = np.roll(
                    profile, -round(profile_max_idx - result_parameter_mid_idx)
                )
                try:
                    result_FWHM[i] = pulse.fwhm(dt=pulse.dt / 100)
                except:
                    result_FWHM[i] = np.nan
            result_optimum_idx = np.nanargmin(result_FWHM)
            result_optimum_fwhm = result_FWHM[result_optimum_idx]

            # Plot FWHM vs grating position
            if self.verbose_pypret:
                plt.close("all")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                fig = plt.plot(scan_positions * 1e3, result_FWHM * 1e15)
                ax.tick_params(labelsize=12)
                plt.xlabel("Position (mm)")
                plt.ylabel("FWHM (fs)")
                plt.title(
                    "Shortest: "
                    + str(np.round(result_optimum_fwhm[0] * 1e15, 1))
                    + " fs @ "
                    + str(np.round(scan_positions[result_optimum_idx] * 1e3, 3))
                    + "mm"
                )
                plt.ylim(
                    0,
                    min(
                        [np.nanmax(result_FWHM * 1e15), 4 * result_optimum_fwhm * 1e15]
                    ),
                )
                plt.show()

            # Plot temporal profile vs grating position
            if self.verbose_pypret:
                plt.close("all")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                fig = plt.contourf(
                    pulse.t * 1e15,
                    scan_positions * 1e3,
                    result_profile.transpose(),
                    200,
                    cmap="nipy_spectral",
                )
                ax.tick_params(labelsize=12)
                plt.xlabel("Time (fs)")
                plt.ylabel("Position (mm)")
                plt.title("Dscan Temporal Profile")
                plt.xlim(
                    -8 * result_optimum_fwhm * 1e15, 8 * result_optimum_fwhm * 1e15
                )
                plt.show()

            # Plot results (Pypret style)
            if self.pypret_plot_position is None:
                self.pypret_plot_parameter = result_parameter[result_optimum_idx]
                final_position = scan_positions[result_optimum_idx]
            else:
                self.pypret_plot_parameter = result_parameter[
                    (
                        np.nanargmin(
                            np.abs(scan_positions - self.pypret_plot_position * 1e-3)
                        )
                    )
                ]
                final_position = self.pypret_plot_position

            plot = RetrievalResultPlot(
                result,
                self.pypret_plot_parameter,
                self.spec_fund_range,
                self.spec_scan_range,
                final_position,
                scan_positions,
            )
            plot.plot(
                fundamental=fund_intensities_bkg_sub,
                fundamental_wavelength=fund_wavelength,
                oversampling=8,
                phase_blanking=True,
                phase_blanking_threshold=0.01,
                limit=True,
                FTL=FTL,
            )

            # Moves the stage to the position indicated in the pypret retrieval
            if self.final_stage_move:
                self.ESP_stage_device.move_to(
                    self.ESP_axis_stage_grating, final_position, True
                )
