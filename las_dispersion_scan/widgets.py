import enum
import pathlib
from typing import Any, ClassVar, Dict, Optional, Protocol, Type, Union

import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from qtpy import QtWidgets
from qtpy.uic import loadUiType

from . import dscan, options, utils


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
    enum_cls: ClassVar[Type[enum.Enum]]
    enum_default: ClassVar[enum.Enum]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        for option in self.enum_cls:
            self.addItem(option.value)

        self.setCurrentIndex(list(self.enum_cls).index(self.enum_default))

    @property
    def current_enum_value(self) -> enum.Enum:
        """The currently-selected enum value."""
        index = self.currentIndex()
        return list(self.enum_cls)[index]


class MaterialComboBox(EnumComboBox):
    enum_cls = options.Material
    enum_default = options.Material.bk7


class NonlinearComboBox(EnumComboBox):
    enum_cls = options.NonlinearProcess
    enum_default = options.NonlinearProcess.shg


class PulseAnalysisComboBox(EnumComboBox):
    enum_cls = options.PulseAnalysisMethod
    enum_default = options.PulseAnalysisMethod.dscan


class SolverComboBox(EnumComboBox):
    enum_cls = options.RetrieverSolver
    enum_default = options.RetrieverSolver.copra


class PlotWidget(FigureCanvasQTAgg):
    def __init__(
        self, parent: Optional[QtWidgets.QWidget] = None, width=5, height=4, dpi=100
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

    # UI-derived widgets:
    apply_limit_checkbox: QtWidgets.QCheckBox
    apply_limit_label: QtWidgets.QLabel
    blur_sigma_label: QtWidgets.QLabel
    blur_sigma_spinbox: QtWidgets.QSpinBox
    calculated_pulse_length_label: QtWidgets.QLabel
    center_banwidth_label: QtWidgets.QLabel
    dscan_plot: PlotWidget
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
    reconstructed_time_plot: PlotWidget
    reconstructed_frequency_plot: PlotWidget
    replot_button: QtWidgets.QPushButton
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
    splitter: QtWidgets.QSplitter
    start_pos_label: QtWidgets.QLabel
    time_frequency_frame: QtWidgets.QHBoxLayout
    time_or_frequency_frame: QtWidgets.QFrame
    time_radio: QtWidgets.QRadioButton
    update_button: QtWidgets.QPushButton
    wedge_angle_label: QtWidgets.QLabel
    wedge_angle_spin: QtWidgets.QDoubleSpinBox
    data: Optional[dscan.Acquisition]
    result: Optional[dscan.PypretResult]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # self.show_type_hints()
        self.result = None
        self.data = None
        self.load_path("/Users/klauer/Repos/general_dscan/Data/XCS/2022_08_15/Dscan_7")
        self.update_button.clicked.connect(self._run_retrieval)
        self.replot_button.clicked.connect(self._update_plots)
        self.reconstructed_frequency_plot.setVisible(False)
        self.time_radio.clicked.connect(self._switch_plot)
        self.frequency_radio.clicked.connect(self._switch_plot)

    def _switch_plot(self):
        self.reconstructed_time_plot.setVisible(self.time_radio.isChecked())
        self.reconstructed_frequency_plot.setVisible(self.frequency_radio.isChecked())

    def load_path(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path).resolve()
        self.data = dscan.Acquisition.from_path(str(path))

    def _run_retrieval(self):
        if self.data is None:
            raise ValueError("Data not acquired or loaded; cannot run retrieval")

        self.result = dscan.PypretResult.from_data(**self.retrieval_parameters)

        self._update_plots()

    def _update_plots(self):
        if self.result is None:
            return

        for widget in [
            self.reconstructed_time_plot,
            self.reconstructed_frequency_plot,
            self.dscan_plot,
        ]:
            widget.figure.clear()

        self.result.plot_frequency_domain_retrieval(
            fig=self.reconstructed_frequency_plot.figure, **self.plot_parameters
        )
        self.reconstructed_frequency_plot.draw()

        self.result.plot_time_domain_retrieval(
            fig=self.reconstructed_time_plot.figure, **self.plot_parameters
        )
        self.reconstructed_time_plot.draw()

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
                "The low fundamental wavelength must be less than the high fundamental "
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
            verbose=False,  # TODO
            **acquisition,
        )
