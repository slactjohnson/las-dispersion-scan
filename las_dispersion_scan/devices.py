import logging
from typing import Type, cast

import ophyd
import ophyd.sim
import pcdsdevices.lasers.qmini as qmini
from ophyd import Component as Cpt
from ophyd import EpicsSignal, Signal
from ophyd.sim import make_fake_device
from pcdsdevices.device import UpdateComponent as UCpt

logger = logging.getLogger(__name__)


class DscanStatus(ophyd.Device):
    scan_start = Cpt(EpicsSignal, "Scan:Start")
    scan_stop = Cpt(EpicsSignal, "Scan:Stop")
    scan_steps = Cpt(EpicsSignal, "Scan:Steps")


class Stage(ophyd.Device, ophyd.sim.SoftPositioner):
    """
    This is a mock implementation that's used to define the interface.

    Users should not subclass from this, but rather use either EpicsMotor
    or PVPositioner classes.
    """

    user_readback = Cpt(Signal, kind="hinted")
    user_setpoint = Cpt(Signal)

    def _set_position(self, value, **kwargs):
        """Set the current internal position, run the readback subscription"""
        self.user_readback.put(value)
        super()._set_position(value, **kwargs)

    @property
    def egu(self) -> str:
        return "mm"


class Spectrometer(ophyd.Device):
    """
    This is a mock implementation that's used to define the interface.

    Users should not subclass from this, but rather implement the signals
    specified here.

    If the signals are not available, ``Cpt(Signal)``.
    """

    start_acquisition = Cpt(Signal)
    status = Cpt(Signal)
    spectrum = Cpt(Signal)
    wavelengths = Cpt(Signal)

    def stop(self):
        ...


class Qmini(qmini.QminiSpectrometer):
    start_acquisition = Cpt(Signal, doc="(No-op signal)")
    # Markers that these components exist in the superclass. We could customize
    # them here:
    status = UCpt()
    spectrum = UCpt()
    wavelengths = UCpt()

    def stop(self):
        ...


FakeDScanStatus = cast(Type[DscanStatus], make_fake_device(DscanStatus))
FakeMotor = cast(Type[Stage], make_fake_device(Stage))
FakeSpectrometer = cast(Type[Spectrometer], make_fake_device(Spectrometer))
