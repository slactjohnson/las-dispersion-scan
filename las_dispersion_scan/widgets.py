import enum
import pathlib
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Type, Union

import matplotlib.figure
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
    enum_default = options.Material.bk7


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
    retrieval_update: ClassVar[QtCore.Signal] = QtCore.Signal(dscan.PypretResult, list)

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
    retriever_settings_label: QtWidgets.QLabel
    right_frame: QtWidgets.QFrame
    scan_button: QtWidgets.QPushButton
    scan_end_spinbox: QtWidgets.QDoubleSpinBox
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
    start_pos_label: QtWidgets.QLabel
    time_radio: QtWidgets.QRadioButton
    update_button: QtWidgets.QPushButton
    wedge_angle_label: QtWidgets.QLabel
    wedge_angle_spin: QtWidgets.QDoubleSpinBox

    data: Optional[dscan.Acquisition]
    result: Optional[dscan.PypretResult]
    loader: device_loader.Loader
    devices: device_loader.Devices

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        loader: Optional[device_loader.Loader] = None,
        debug: bool = False,
    ):
        super().__init__(parent=parent)

        self.loader = loader
        if loader is not None:
            self.devices = loader.load()
        else:
            self.devices = None

        self.show_type_hints()
        self.result = None
        self.data = None
        self._debug = debug
        self._retrieval_thread = None
        self.load_path("/Users/klauer/Repos/general_dscan/Data/XCS/2022_08_15/Dscan_7")
        self.update_button.clicked.connect(self._start_retrieval)
        self.replot_button.clicked.connect(self._update_plots)
        self.retrieval_update.connect(self._retrieval_partial_update)

        for radio in [
            self.time_radio,
            self.frequency_radio,
            self.acquired_radio,
            self.difference_radio,
            self.retrieved_radio,
        ]:
            radio.clicked.connect(self._switch_plot)

        self._switch_plot()
        self._connect_devices()

    def _connect_devices(self) -> None:
        if self.devices is None:
            return

        stage = self.devices.stage
        spectrometer = self.devices.spectrometer
        status = self.devices.status
        self.stage_suite_button.setText(stage.name)
        self.stage_suite_button.add_device(stage)

        self.spectrometer_suite_button.setText(spectrometer.name)
        self.spectrometer_suite_button.add_device(spectrometer)
        self.setWindowTitle(f"D-scan Diagnostic ({status.prefix})")

        readback = getattr(stage, "user_readback", None)
        if readback is not None:
            self.stage_status_label.channel = utils.channel_from_signal(readback)

        spec_status = getattr(spectrometer, "status", None)
        if spec_status is not None:
            self.spectrometer_status_label.channel = utils.channel_from_signal(
                spec_status
            )

    def _switch_plot(self):
        self.reconstructed_time_plot.setVisible(self.time_radio.isChecked())
        self.reconstructed_frequency_plot.setVisible(self.frequency_radio.isChecked())
        self.dscan_acquired_plot.setVisible(self.acquired_radio.isChecked())
        self.dscan_difference_plot.setVisible(self.difference_radio.isChecked())
        self.dscan_retrieved_plot.setVisible(self.retrieved_radio.isChecked())

    @QtCore.Slot(object, list)
    def _retrieval_partial_update(
        self, result: dscan.PypretResult, data: List[np.ndarray]
    ):
        ...
        # TODO: some sort of live view here
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

    def load_path(self, path: Union[str, pathlib.Path]):
        """
        Load data from the provided path.

        Filenames matching the supported data formats are searched in order of
        "new" (.dat) to "old" (.txt).
        """
        path = pathlib.Path(path).resolve()
        self.data = dscan.Acquisition.from_path(str(path))

    def _start_retrieval(self) -> None:
        """Run the pulse retrieval process and update the plots."""
        if self.data is None:
            raise ValueError("Data not acquired or loaded; cannot run retrieval")

        if self._retrieval_thread is not None:
            raise RuntimeError("Another retrieval is currently running")

        def per_step_callback_in_thread(data: List[np.ndarray]) -> None:
            """
            Per-iteration step for pypret, called in a thread.

            Do not do GUI operations here; subscribe to the signal instead.
            """
            self.retrieval_update.emit(self.pypret_result, data)

        def retrieval() -> dscan.PypretResult:
            if self._debug:
                # In debug mode, we keep the numpy random seed consistent
                # between retrieval runs
                np.random.seed(0)
            result = dscan.PypretResult.from_data(**self.retrieval_parameters)
            self.pypret_result = result
            result.run(callback=per_step_callback_in_thread)
            return result

        def retrieval_finished(
            return_value: Optional[dscan.PypretResult], ex: Optional[Exception]
        ) -> None:
            self.update_button.setEnabled(True)
            self.replot_button.setEnabled(True)
            self._retrieval_thread = None

            if return_value is None or ex is not None:
                raise_to_operator(ex)
                return

            self.result = return_value
            self.pulse_length_lineedit.setText(f"{self.result.pulse_width_fs:.2f}")
            self._update_plots()

        self._retrieval_thread = utils.ThreadWorker(func=retrieval)
        self._retrieval_thread.returned.connect(retrieval_finished)
        self.update_button.setEnabled(False)
        self.replot_button.setEnabled(False)
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
        if self.data is not None:
            acquisition = dict(
                fund=self.data.fundamental,
                scan=self.data.scan,
            )
        else:
            acquisition = {}

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
            **acquisition,
        )
