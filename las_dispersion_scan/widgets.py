import enum
from typing import ClassVar, Optional, Protocol, Type

import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from qtpy import QtWidgets
from qtpy.uic import loadUiType

from . import options, utils


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
                    print(
                        f"{attr}: {obj.__class__.__module__}.{obj.__class__.__name__}"
                    )


class EnumComboBox(QtWidgets.QComboBox):
    enum_cls: ClassVar[Type[enum.Enum]]
    enum_default: ClassVar[enum.Enum]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        for option in self.enum_cls:
            self.addItem(option.value)

        self.setCurrentIndex(list(self.enum_cls).index(self.enum_default))


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
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

        if parent is not None:
            self.setParent(parent)


class DscanMain(DesignerDisplay, QtWidgets.QWidget):
    """
    Main display.
    """

    filename: ClassVar[str] = "main.ui"
