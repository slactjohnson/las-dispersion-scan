import datetime
import enum
import logging
import pathlib
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Type, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pydm.widgets
import typhos.related_display
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from qtpy import QtCore, QtWidgets
from qtpy.uic import loadUiType
from typhos.utils import raise_to_operator

from . import dscan
from . import loader as device_loader
from . import options, plotting, utils

logger = logging.getLogger(__name__)


class _UiForm(Protocol):
    @staticmethod  # <-- this is not entirely true, but PyQt5 uses it as such
    def setupUi(*args):
        ...

    @staticmethod
    def retranslateUi(*args, **kwargs):
        ...


class DesignerDisplay:
    """Helper class for loading designer .ui files and adding logic."""

    filename: ClassVar[str]
    ui_form: ClassVar[Type[_UiForm]]

    @classmethod
    def _load_ui_if_needed(cls):
        """Load the UI file on first load."""
        if not hasattr(cls, "ui_form"):
            cls.ui_form, _ = loadUiType(str(utils.SOURCE_PATH / "ui" / cls.filename))

    def __init__(self, *args, **kwargs):
        """Apply the file to this widget when the instance is created"""
        self._load_ui_if_needed()
        super().__init__(*args, **kwargs)
        self.ui_form.setupUi(self, self)

    def retranslateUi(self, *args, **kwargs):
        """Required function for setupUi to work in __init__"""
        self.ui_form.retranslateUi(self, *args, **kwargs)

    def show_type_hints(self):
        """Show type hints of widgets included in the display for development help."""
        cls_attrs = set()
        obj_attrs = set(dir(self))
        annotated = set(self.__annotations__)
        for cls in type(self).mro():
            cls_attrs |= set(dir(cls))
        likely_from_ui = obj_attrs - cls_attrs - annotated
        for attr in sorted(likely_from_ui):
            try:
                obj = getattr(self, attr, None)
            except Exception:
                ...
            else:
                if obj is not None:
                    cls = f"{obj.__class__.__module__}.{obj.__class__.__name__} "
                    cls = cls.removeprefix("PyQt5.")
                    cls = cls.removeprefix("las_dispersion_scan.widgets.")
                    print(f"{attr}: {cls}")


class EnumComboBox(QtWidgets.QComboBox):
    enum_default: ClassVar[enum.Enum]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        for option in type(self.enum_default):
            self.addItem(option.value)

        self._enum_cls = type(self.enum_default)
        self.setCurrentIndex(list(self._enum_cls).index(self.enum_default))

    @property
    def current_enum_value(self) -> enum.Enum:
        """The currently-selected enum value."""
        index = self.currentIndex()
        return list(self._enum_cls)[index]


class MaterialComboBox(EnumComboBox):
    enum_default = options.Material.gratinga


class NonlinearComboBox(EnumComboBox):
    enum_default = options.NonlinearProcess.shg


class PulseAnalysisComboBox(EnumComboBox):
    enum_default = options.PulseAnalysisMethod.dscan


class SolverComboBox(EnumComboBox):
    enum_default = options.RetrieverSolver.copra


class PlotWidget(FigureCanvasQTAgg):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        width: int = 4,
        height: int = 4,
        dpi: int = 100,
    ):
        fig = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        # self.axes = fig.add_subplot(111)
        super().__init__(fig)

        if parent is not None:
            self.setParent(parent)


class DscanMain(DesignerDisplay, QtWidgets.QWidget):
    """
    Main display.
    """

    filename: ClassVar[str] = "main.ui"

    # [mu, parameter, process_w, new_spectrum]
    new_scan_point: ClassVar[QtCore.Signal] = QtCore.Signal(dscan.ScanPointData)
    scan_finished: ClassVar[QtCore.Signal] = QtCore.Signal(dscan.ScanData)
    retrieval_update: ClassVar[QtCore.Signal] = QtCore.Signal(
        dscan.PypretResult, int, list
    )
    retrieval_finished: ClassVar[QtCore.Signal] = QtCore.Signal(dscan.PypretResult)
    replotted: ClassVar[QtCore.Signal] = QtCore.Signal()

    # pypret / scan-related
    data: dscan.Acquisition
    result: Optional[dscan.PypretResult]
    loader: device_loader.Loader
    devices: device_loader.Devices
    scan: Optional[dscan.AcquisitionScan]
    _scan_thread: Optional[utils.ThreadWorker]
    _retrieval_thread: Optional[utils.ThreadWorker]
    auto_save_path: Optional[pathlib.Path]
    _scan_saved: bool

    # UI-derived widgets:
    acquired_or_retrieved_frame: QtWidgets.QFrame
    acquired_radio: QtWidgets.QRadioButton
    apply_limit_checkbox: QtWidgets.QCheckBox
    apply_limit_label: QtWidgets.QLabel
    blur_sigma_label: QtWidgets.QLabel
    blur_sigma_spinbox: QtWidgets.QSpinBox
    calculated_pulse_length_label: QtWidgets.QLabel
    center_banwidth_label: QtWidgets.QLabel
    debug_mode_checkbox: QtWidgets.QCheckBox
    debug_mode_label: QtWidgets.QLabel
    difference_radio: QtWidgets.QRadioButton
    dscan_acquired_plot: PlotWidget
    dscan_difference_plot: PlotWidget
    dscan_retrieved_plot: PlotWidget
    dwell_time_label: QtWidgets.QLabel
    dwell_time_spinbox: QtWidgets.QDoubleSpinBox
    export_button: QtWidgets.QPushButton
    freq_bandwidth_spinbox: QtWidgets.QDoubleSpinBox
    frequency_radio: QtWidgets.QRadioButton
    fundamental_high_spinbox: QtWidgets.QDoubleSpinBox
    fundamental_low_spinbox: QtWidgets.QDoubleSpinBox
    fundamental_range_label: QtWidgets.QLabel
    grid_points_label: QtWidgets.QLabel
    import_button: QtWidgets.QPushButton
    iterations_label: QtWidgets.QLabel
    iterations_spinbox: QtWidgets.QSpinBox
    left_frame: QtWidgets.QFrame
    material_combo: MaterialComboBox
    material_label: QtWidgets.QLabel
    nonlinear_combo: NonlinearComboBox
    nonlinear_process_label: QtWidgets.QLabel
    num_grid_points_spinbox: QtWidgets.QSpinBox
    oversampling_label: QtWidgets.QLabel
    oversampling_spinbox: QtWidgets.QSpinBox
    params_label: QtWidgets.QLabel
    spectra_per_step_spinbox: QtWidgets.QSpinBox
    phase_blanking_checkbox: QtWidgets.QCheckBox
    phase_blanking_label: QtWidgets.QLabel
    phase_blanking_threshold_label: QtWidgets.QLabel
    phase_blanking_threshold_spinbox: QtWidgets.QDoubleSpinBox
    plot_label: QtWidgets.QLabel
    plot_settings_label: QtWidgets.QLabel
    pulse_analysis_combo: PulseAnalysisComboBox
    pulse_analysis_label: QtWidgets.QLabel
    pulse_layout: QtWidgets.QHBoxLayout
    pulse_length_lineedit: QtWidgets.QLineEdit
    reconstruct_label: QtWidgets.QLabel
    reconstructed_frequency_plot: PlotWidget
    reconstructed_time_plot: PlotWidget
    replot_button: QtWidgets.QPushButton
    retrieved_radio: QtWidgets.QRadioButton
    retrieval_progress: QtWidgets.QProgressBar
    retriever_settings_label: QtWidgets.QLabel
    right_frame: QtWidgets.QFrame
    save_automatically_checkbox: QtWidgets.QCheckBox
    scan_button: QtWidgets.QPushButton
    scan_end_spinbox: QtWidgets.QDoubleSpinBox
    scan_progress: QtWidgets.QProgressBar
    scan_start_spinbox: QtWidgets.QDoubleSpinBox
    scan_status_label: QtWidgets.QLabel
    scan_steps_label: QtWidgets.QLabel
    scan_steps_spinbox: QtWidgets.QSpinBox
    scan_wavelength_high_spinbox: QtWidgets.QDoubleSpinBox
    scan_wavelength_label: QtWidgets.QLabel
    scan_wavelength_low_spinbox: QtWidgets.QDoubleSpinBox
    solver_combo: SolverComboBox
    solver_label: QtWidgets.QLabel
    spectrometer_label: QtWidgets.QLabel
    spectrometer_status_label: pydm.widgets.label.PyDMLabel
    spectrometer_suite_button: typhos.related_display.TyphosRelatedSuiteButton
    splitter: QtWidgets.QSplitter
    stage_label: QtWidgets.QLabel
    stage_status_label: pydm.widgets.label.PyDMLabel
    stage_suite_button: typhos.related_display.TyphosRelatedSuiteButton
    take_fundamental_button: QtWidgets.QPushButton
    start_pos_label: QtWidgets.QLabel
    time_radio: QtWidgets.QRadioButton
    update_button: QtWidgets.QPushButton
    wedge_angle_label: QtWidgets.QLabel
    wedge_angle_spin: QtWidgets.QDoubleSpinBox

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        loader: Optional[device_loader.Loader] = None,
        debug: bool = False,
        auto_save_path: Optional[pathlib.Path] = None,
    ):
        super().__init__(parent=parent)

        self.loader = loader
        if loader is not None:
            self.devices = loader.load()
        else:
            self.devices = None

        self.show_type_hints()
        self.result = None
        self.acquisition_scan = None
        self.saved_filename = None
        self._debug = debug
        self._retrieval_thread = None
        self._scan_saved = False
        self._scan_thread = None
        self._export_menu = None
        self.auto_save_path = auto_save_path
        self.data = dscan.Acquisition()
        self.update_button.clicked.connect(self._start_retrieval)
        self.import_button.clicked.connect(self._start_import)
        self.replot_button.clicked.connect(self._update_plots)
        self.retrieval_update.connect(self._on_retrieval_partial_update)
        self.retrieval_finished.connect(self._on_retrieval_finished)
        self.take_fundamental_button.clicked.connect(self.take_fundamental)
        self.save_automatically_checkbox.toggled.connect(
            self._save_automatically_checked
        )
        self.scan_finished.connect(self._on_scan_finished)

        for radio in [
            self.time_radio,
            self.frequency_radio,
            self.acquired_radio,
            self.difference_radio,
            self.retrieved_radio,
        ]:
            radio.clicked.connect(self._switch_plot)

        self.scan_button.clicked.connect(self._start_scan)
        self.new_scan_point.connect(self._on_new_scan_point)
        self.retrieval_progress.setVisible(False)
        self.scan_progress.setVisible(False)
        self._switch_plot()
        self._create_menus()
        self._connect_devices()
        self._update_title()

    def _connect_devices(self) -> None:
        if self.devices is None:
            return

        stage = self.devices.stage
        spectrometer = self.devices.spectrometer
        self.stage_suite_button.setText(stage.name)
        self.stage_suite_button.add_device(stage)

        if stage.connected:
            # TODO: this may be on a connection callback
            self.scan_start_spinbox.setSuffix(f" {stage.egu}")
            self.scan_end_spinbox.setSuffix(f" {stage.egu}")

        self.spectrometer_suite_button.setText(spectrometer.name)
        self.spectrometer_suite_button.add_device(spectrometer)

        readback = getattr(stage, "user_readback", None)
        if readback is not None:
            self.stage_status_label.channel = utils.channel_from_signal(readback)

        spec_status = getattr(spectrometer, "status", None)
        if spec_status is not None:
            self.spectrometer_status_label.channel = utils.channel_from_signal(
                spec_status
            )

    def _update_title(self) -> None:
        title = "D-scan Diagnostic"

        if self.devices is not None and self.devices.status.prefix:
            title = f"{title} ({self.devices.status.prefix})"

        if self.saved_filename is not None:
            title = f"{title}: {self.saved_filename}"

        self.setWindowTitle(title)

    @QtCore.Slot(bool)
    def _save_automatically_checked(self, checked: bool) -> None:
        if checked:
            if self.auto_save_path is None:
                path = QtWidgets.QFileDialog.getExistingDirectory(
                    self, "Directory to save files (dscan_*.npz; dscan_*.png)"
                )
                if path:
                    self.auto_save_path = pathlib.Path(path)
            if not self.auto_save_path:
                self.save_automatically_checkbox.setChecked(False)
                return
        else:
            self.auto_save_path = None

    def _switch_plot(self):
        self.reconstructed_time_plot.setVisible(self.time_radio.isChecked())
        self.reconstructed_frequency_plot.setVisible(self.frequency_radio.isChecked())
        self.dscan_acquired_plot.setVisible(self.acquired_radio.isChecked())
        self.dscan_difference_plot.setVisible(self.difference_radio.isChecked())
        self.dscan_retrieved_plot.setVisible(self.retrieved_radio.isChecked())

    def _create_menus(self):
        self._export_menu = QtWidgets.QMenu()
        npz_export = self._export_menu.addAction("Save as one .npz file")
        npz_export.triggered.connect(self._save_as_npz)
        dat_export = self._export_menu.addAction("Save as separate .dat files")
        dat_export.triggered.connect(self._save_as_dat)
        self.export_button.setMenu(self._export_menu)

    @property
    def retrieval_is_running(self) -> bool:
        """Is the retrieval thread running?"""
        return self._retrieval_thread is not None and self._retrieval_thread.isRunning()

    @property
    def scan_is_running(self) -> bool:
        """Is the scan thread running?"""
        return self._scan_thread is not None and self._scan_thread.isRunning()

    def _save_as_dat(
        self, *, directory: Optional[Union[pathlib.Path, str]] = None
    ) -> Optional[pathlib.Path]:
        if directory is None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Directory to save files (fund.dat; scan.dat)"
            )
            if not directory:
                return

        directory = pathlib.Path(directory)
        self.data.save(directory, format="dat")
        self.saved_filename = f"{directory}/*.dat"
        self._update_title()
        return directory

    def _save_as_npz(
        self, *, filename: Optional[Union[pathlib.Path, str]] = None
    ) -> Optional[pathlib.Path]:
        if filename is None:
            filename, filter_ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save all data", ".", "Numpy zip file (*.npz);;All files (*.*)"
            )
            if not filename:
                return

        filename = pathlib.Path(filename)
        self.data.settings = dict(**self.retrieval_parameters, **self.plot_parameters)
        self.data.save(filename, format="npz")
        self.saved_filename = filename
        self._update_title()
        return filename

    @QtCore.Slot(object)
    def _on_scan_finished(self, data: dscan.ScanData):
        self.scan_progress.setVisible(False)
        if self.retrieval_is_running:
            return
        if self.scan is not None and self.scan.stopped:
            return

        if not len(self.data.fundamental.wavelengths):
            self.data.fundamental.wavelengths = data.wavelengths
            self.data.fundamental.intensities = data.intensities[0, :]

        self._start_retrieval()

    @QtCore.Slot(object, int, list)
    def _on_retrieval_partial_update(
        self, result: dscan.PypretResult, iteration: int, data: List[np.ndarray]
    ):
        self.retrieval_progress.setValue(iteration)
        # TODO: some sort of live view here?
        # mu, parameter, process_w, new_spectrum = data  # noqa

        # self.pulse_length_lineedit.setText(
        #     f"(Working)"
        # )
        # fig = self.reconstructed_time_plot.figure
        # fig.clf()

        # ax1 = cast(plt.Axes, fig.subplots(nrows=1, ncols=1))
        # ax12 = cast(plt.Axes, ax1.twinx())

        # pypret.graphics.plot_complex(
        #     ...,
        #     ...,
        #     new_spectrum,
        #     ax1,
        #     ax12,
        #     yaxis="intensity",
        #     phase_blanking=False,
        #     limit=True,
        #     phase_blanking_threshold=0.01,
        # )
        # self.reconstructed_time_plot.draw()

    def get_autosave_date(self) -> str:
        """Get an autosave filename suffix that includes the date/time."""
        # Filename of the form:
        # *2022-10-17T11_20_00_39000.npz
        suffix = datetime.datetime.now().isoformat()
        suffix = suffix.replace(":", "_")
        suffix = suffix.replace(".", "_")
        return suffix

    @property
    def should_auto_save(self) -> bool:
        """Should the auto-save process happen after retrieval?"""
        return self.save_automatically_checkbox.isChecked() and not self._scan_saved

    @QtCore.Slot(object)
    def _on_retrieval_finished(self, result: dscan.PypretResult):
        """Slot when the retrieval process finishes."""
        self.retrieval_progress.setVisible(False)
        self._update_plots()
        self.auto_save()

    def auto_save(self) -> Optional[pathlib.Path]:
        """Save the scan data and reconstruction plots."""
        if not self.should_auto_save or self.auto_save_path is None:
            return None

        autosave_suffix = self.get_autosave_date()
        path = self.auto_save_path / f"dscan_{autosave_suffix}.npz"
        self._save_as_npz(filename=path)
        self.reconstructed_time_plot.figure.savefig(
            self.auto_save_path / f"dscan_time_{autosave_suffix}.png"
        )
        self.dscan_retrieved_plot.figure.savefig(
            self.auto_save_path / f"dscan_retrieved_{autosave_suffix}.png"
        )
        self._scan_saved = True
        return path

    def take_fundamental(self) -> None:
        if self.devices is None:
            return

        self.data.fundamental = dscan.SpectrumData.from_device(
            self.devices.spectrometer
        )
        _, ax = self.data.fundamental.plot()
        stage = self.devices.stage
        stage_pos = stage.user_readback.get()
        ax.set_title(f"Fundamental spectrum at {stage_pos:.6f} {stage.egu}")
        plt.show()

    def load_path(self, path: Union[str, pathlib.Path]) -> None:
        """
        Load data from the provided path.

        Filenames matching the supported data formats are searched in order of
        "new" (.dat) to "old" (.txt).
        """
        path = pathlib.Path(path).resolve()
        self.data = dscan.Acquisition.from_path(str(path))
        self.saved_filename = path
        self._update_title()

    def _start_import(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import data",
            ".",
            ";;".join(
                (
                    "Numpy zip file (*.npz)",
                    "Newer .dat file format (*.dat)",
                    "Original text file format (*.txt)",
                    "All files (*.*)",
                )
            ),
        )
        if not filename:
            return

        path = pathlib.Path(filename)
        if path.suffix.lower() in (".dat", ".txt"):
            logger.warning(
                "Old Importing old file format: ignoring filename %s "
                "The GUI will load either .dat or .txt from %s",
                path.name,
                path.parent,
            )
            path = path.parent

        self.load_path(path)

    def _on_new_scan_point(self, data: dscan.ScanPointData) -> None:
        self.scan_progress.setValue(data.index + 1)
        self.scan_status_label.setText(
            f"Acquired [{data.index + 1}] at {data.readback * 1e-3:.3g} mm"
        )

    def _start_scan(self) -> None:
        """Start a scan/acquisition to get spectra for the configured points."""
        if self.scan_is_running:
            if self.scan is not None:
                self.scan.stop()
                return

            raise RuntimeError("Another scan is currently running")

        if self.devices is None:
            raise RuntimeError("Device loader not configured; scans not possible")

        def scan() -> dscan.ScanData:
            assert self.scan is not None

            for point in self.scan.run(
                positions=list(positions),
                dwell_time=dwell_time,
                per_step_spectra=self.spectra_per_step_spinbox.value(),
            ):
                self.new_scan_point.emit(point)

            assert self.scan.data is not None
            return self.scan.data

        def scan_finished(
            return_value: Optional[dscan.ScanData], ex: Optional[Exception]
        ) -> None:
            self.scan_button.setText("&Start")
            self.scan_status_label.setText("Scan finished.")

            if return_value is None or ex is not None:
                raise_to_operator(ex)
                return

            self.data.scan = return_value
            self.scan_finished.emit(return_value)

        positions = np.linspace(
            start=self.scan_start_spinbox.value(),
            stop=self.scan_end_spinbox.value(),
            num=self.scan_steps_spinbox.value(),
        )
        dwell_time = self.dwell_time_spinbox.value()
        if len(positions) <= 1:
            raise ValueError("Invalid scan parameters specified.")

        self.scan = dscan.AcquisitionScan(
            stage=self.devices.stage, spectrometer=self.devices.spectrometer
        )
        self.scan_status_label.setText("Scan initialized...")
        self._scan_saved = False
        self._scan_thread = utils.ThreadWorker(func=scan)
        self._scan_thread.returned.connect(scan_finished)
        self.scan_progress.setMinimum(0)
        self.scan_progress.setValue(0)
        self.scan_progress.setMaximum(self.scan_steps_spinbox.value())
        self.scan_progress.setVisible(True)

        self.scan_button.setText("&Stop")
        self._scan_thread.start()

    def _start_retrieval(self) -> None:
        """Run the pulse retrieval process and update the plots."""
        if self.data is None:
            raise ValueError("Data not acquired or loaded; cannot run retrieval")

        if self.retrieval_is_running:
            raise RuntimeError("Another retrieval is currently running")

        def per_step_callback_in_thread(data: List[np.ndarray]) -> None:
            """
            Per-iteration step for pypret, called in a thread.

            Do not do GUI operations here; subscribe to the signal instead.
            """
            nonlocal iteration
            iteration += 1
            self.retrieval_update.emit(self.pypret_result, iteration, data)

        def retrieval() -> dscan.PypretResult:
            if self._debug:
                # In debug mode, we keep the numpy random seed consistent
                # between retrieval runs
                np.random.seed(0)
            try:
                result = dscan.PypretResult.from_data(**self.retrieval_parameters)
                self.pypret_result = result
                result.run(callback=per_step_callback_in_thread)
            except Exception as ex:
                if self._debug:
                    logger.warning(
                        "Caught exception: %s. Starting debug mode IPython console.",
                        ex,
                        exc_info=True,
                    )
                    from IPython import embed

                    embed()
                raise

            return result

        def retrieval_finished(
            return_value: Optional[dscan.PypretResult], ex: Optional[Exception]
        ) -> None:
            self.update_button.setEnabled(True)
            self.replot_button.setEnabled(True)

            if return_value is None or ex is not None:
                raise_to_operator(ex)
                return

            self.result = return_value
            self.pulse_length_lineedit.setText(f"{self.result.pulse_width_fs:.2f}")
            self.retrieval_finished.emit(self.pypret_result)

        iteration = 0
        self._retrieval_thread = utils.ThreadWorker(func=retrieval)
        self._retrieval_thread.returned.connect(retrieval_finished)
        self.update_button.setEnabled(False)
        self.replot_button.setEnabled(False)
        self.retrieval_progress.setMinimum(0)
        self.retrieval_progress.setMaximum(self.iterations_spinbox.value())
        self.retrieval_progress.setValue(0)
        self.retrieval_progress.setVisible(True)
        self._retrieval_thread.start()

    def _update_plots(self) -> None:
        """Update all of the plots based on the results from pypret."""
        if self.result is None:
            return

        widgets = [
            self.reconstructed_time_plot,
            self.reconstructed_frequency_plot,
            self.dscan_acquired_plot,
            self.dscan_difference_plot,
            self.dscan_retrieved_plot,
        ]

        for widget in widgets:
            widget.figure.clear()

        self.result.plot_trace(
            fig=self.dscan_acquired_plot.figure,
            option=plotting.PlotTrace.measured,
        )

        self.result.plot_trace(
            fig=self.dscan_retrieved_plot.figure,
            option=plotting.PlotTrace.retrieved,
        )

        self.result.plot_trace(
            fig=self.dscan_difference_plot.figure,
            option=plotting.PlotTrace.difference,
        )

        self.result.plot_frequency_domain_retrieval(
            fig=self.reconstructed_frequency_plot.figure, **self.plot_parameters
        )

        self.result.plot_time_domain_retrieval(
            fig=self.reconstructed_time_plot.figure, **self.plot_parameters
        )

        if self.debug_mode_checkbox.isChecked():
            self.result.plot_all_debug(**self.plot_parameters)

        for widget in widgets:
            widget.draw()

        self.replotted.emit()

    @property
    def plot_parameters(self) -> Dict[str, Any]:
        """Parameters to be passed to ``PypretResult.plot*`` methods."""
        return dict(
            limit=self.apply_limit_checkbox.isChecked(),
            oversampling=self.oversampling_spinbox.value(),
            phase_blanking=self.phase_blanking_checkbox.isChecked(),
            phase_blanking_threshold=self.phase_blanking_threshold_spinbox.value(),
        )

    @property
    def retrieval_parameters(self) -> Dict[str, Any]:
        """Parameters to be passed to ``PypretResult.from_data``."""
        if (
            self.fundamental_low_spinbox.value()
            >= self.fundamental_high_spinbox.value()
        ):
            self.fundamental_low_spinbox.setFocus()
            raise ValueError(
                "The low fundamental wavelength must be less than the high fundamental "
                "value."
            )

        if (
            self.scan_wavelength_low_spinbox.value()
            >= self.scan_wavelength_high_spinbox.value()
        ):
            self.scan_wavelength_low_spinbox.setFocus()
            raise ValueError(
                "The low scan wavelength must be less than the high scan wavelength "
                "value."
            )

        return dict(
            fund=self.data.fundamental,
            scan=self.data.scan,
            material=self.material_combo.current_enum_value,
            method=self.pulse_analysis_combo.current_enum_value,
            nlin_process=self.nonlinear_combo.current_enum_value,
            wedge_angle=self.wedge_angle_spin.value(),
            blur_sigma=self.blur_sigma_spinbox.value(),
            num_grid_points=self.num_grid_points_spinbox.value(),
            freq_bandwidth_wl=self.freq_bandwidth_spinbox.value(),
            max_iter=self.iterations_spinbox.value(),
            spec_fund_range=(
                self.fundamental_low_spinbox.value(),
                self.fundamental_high_spinbox.value(),
            ),
            spec_scan_range=(
                self.scan_wavelength_low_spinbox.value(),
                self.scan_wavelength_high_spinbox.value(),
            ),
            plot_position=None,  # TODO
        )
