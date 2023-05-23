import logging
import os
from typing import Type, cast

import ophyd
import ophyd.sim
import pcdsdevices.lasers.qmini as qmini
from ophyd import Component as Cpt
from ophyd import EpicsSignal, Signal
from ophyd.sim import make_fake_device
from pcdsdevices.device import UpdateComponent as UCpt
from pcdsdevices.lasers import elliptec

from .motion import move_with_retries

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


class Qmini(qmini.QminiSpectrometer):
    start_acquisition = Cpt(Signal, doc="(No-op signal)")
    # Markers that these components exist in the superclass. We could customize
    # them here:
    status = UCpt()
    spectrum = UCpt()
    wavelengths = UCpt()

    wavelength_units = "nm"


class EllLinear(elliptec.EllLinear):
    def __init__(
        self,
        *args,
        motor_units: str = "",
        use_retries: bool = True,
        **kwargs,
    ):
        self._motor_units = motor_units or os.environ.get("MOTOR_UNITS", "mm")
        self._use_retries = use_retries
        super().__init__(*args, **kwargs)

    @property
    def egu(self) -> str:
        return self._motor_units

    def set(
        self,
        position: float,
        *,
        wait: bool = True,
        retry_timeout: float = 1.0,
        retry_deadband: float = 0.01,
        max_retries: int = 10,
        timeout: float = 10.0,
        **kwargs,
    ):
        if not self._use_retries:
            return super().set(position, wait=wait)

        return move_with_retries(
            self,
            position=position,
            retry_timeout=retry_timeout,
            retry_deadband=retry_deadband,
            max_retries=max_retries,
            timeout=timeout,
        )


FakeDScanStatus = cast(Type[DscanStatus], make_fake_device(DscanStatus))
FakeMotor = cast(Type[Stage], make_fake_device(Stage))
FakeSpectrometer = cast(Type[Spectrometer], make_fake_device(Spectrometer))
