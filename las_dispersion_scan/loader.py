import importlib
import logging
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Type, cast

import happi
import ophyd
from ophyd import Component as Cpt
from ophyd import EpicsSignal
from ophyd.sim import make_fake_device

from .utils import get_device_from_happi

logger = logging.getLogger(__name__)


class DscanStatus(ophyd.Device, ophyd.PositionerBase):
    scan_start = Cpt(EpicsSignal, "Scan:Start")
    scan_stop = Cpt(EpicsSignal, "Scan:Stop")
    scan_steps = Cpt(EpicsSignal, "Scan:Steps")


class Motor(ophyd.Device, ophyd.PositionerBase):
    ...


class Spectrometer(ophyd.Device):
    ...


FakeDScanStatus = cast(Type[DscanStatus], make_fake_device(DscanStatus))
FakeMotor = cast(Type[Motor], make_fake_device(Motor))
FakeSpectrometer = cast(Type[Spectrometer], make_fake_device(Spectrometer))


@dataclass
class Devices:
    status: DscanStatus
    motor: Motor
    spectrometer: Spectrometer


def import_module(module_name: str) -> ModuleType:
    """
    Import a module given its name.

    The module must be on the PYTHONPATH.

    Parameters
    ----------
    module_name : str
        The module name.

    Returns
    -------
    mod : ModuleType
        The module.
    """
    # Import the module if not already present
    # Otherwise use the stashed version in sys.modules
    try:
        return sys.modules[module_name]
    except KeyError:
        logger.debug("Importing %s", module_name)
        return importlib.import_module(module_name)


@dataclass
class Loader:
    """
    Helper to load the user-provided Devices.

    Devices may come from happi or an external script.
    The status object needs only to have its prefix configured.
    """

    #: Prefix of the status device.
    prefix: Optional[str] = None
    #: An external script to run to define 'motor' and 'spectrometer'
    script: Optional[str] = None
    #: An optional happi item name for the motor.
    motor: Optional[str] = None
    #: An optional happi item name for the spectrometer.
    spectrometer: Optional[str] = None

    def load(self, client: Optional[happi.Client] = None) -> Devices:
        cls = DscanStatus if self.prefix else FakeDScanStatus
        status = cls(self.prefix or "", name="DscanStatus")

        motor = FakeMotor("", name="no-motor")
        spectrometer = FakeSpectrometer("", name="no-spectrometer")

        if self.script is not None:
            module = import_module(self.script)
            motor = getattr(module, "motor", motor)
            spectrometer = getattr(module, "spectrometer", spectrometer)

        if self.motor is not None:
            motor = cast(Motor, get_device_from_happi(self.motor, client=client))
        if self.spectrometer is not None:
            spectrometer = cast(
                Spectrometer, get_device_from_happi(self.spectrometer, client=client)
            )

        return Devices(
            status=status,
            motor=motor,
            spectrometer=spectrometer,
        )
