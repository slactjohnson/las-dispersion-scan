import importlib
import logging
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import List, Optional, Type, cast

import happi
import ophyd
import ophyd.sim

from . import devices
from .utils import get_device_from_happi

logger = logging.getLogger(__name__)


@dataclass
class Devices:
    status: devices.DscanStatus
    stage: devices.Stage
    spectrometer: devices.Spectrometer


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


def check_device_against_interface(
    device: ophyd.Device,
    interface: Type[ophyd.Device],
    attrs: List[str],
):
    """
    Check a device instance against a device class that defines an interface.

    Requires that all components of the interface be included in the device.
    Requires that all ``attrs`` provided exist on the device.
    Does not recurse to sub-devices.

    Parameters
    ----------
    device : ophyd.Device
        The device instance.
    interface : Type[ophyd.Device]
        The interface.
    attrs : List[str]
        Additional attributes required to exist.
    """
    cls = type(device)
    missing_components = set(interface.component_names) - set(device.component_names)
    description = f"Device {device.name} ({cls.__module__}{cls.__name__})"

    if missing_components:
        raise ValueError(
            f"{description} is missing "
            f"these required components: {missing_components}"
        )

    missing_attrs = [attr for attr in attrs if not hasattr(device, attr)]
    if missing_attrs:
        raise ValueError(
            f"{description} is missing "
            f"these required attributes or methods: {missing_attrs}"
        )


@dataclass
class Loader:
    """
    Helper to load the user-provided Devices.

    Devices may come from happi or an external script.
    The status object needs only to have its prefix configured.
    """

    #: Prefix of the status device.
    prefix: Optional[str] = None
    #: An external script to run to define 'stage' and 'spectrometer'
    script: Optional[str] = None
    #: An optional happi item name for the stage.
    stage: Optional[str] = None
    #: An optional happi item name for the spectrometer.
    spectrometer: Optional[str] = None

    def load(self, client: Optional[happi.Client] = None) -> Devices:
        cls = devices.DscanStatus if self.prefix else devices.FakeDScanStatus
        status = cls(self.prefix or "", name="DscanStatus")

        stage = devices.FakeMotor("", name="no-motor")
        spectrometer = devices.FakeSpectrometer("", name="no-spectrometer")

        if self.script is not None:
            module = import_module(self.script)
            stage = getattr(module, "stage", stage)
            spectrometer = getattr(module, "spectrometer", spectrometer)

        if self.stage is not None:
            stage = cast(
                devices.Stage, get_device_from_happi(self.stage, client=client)
            )
        if self.spectrometer is not None:
            spectrometer = cast(
                devices.Spectrometer,
                get_device_from_happi(self.spectrometer, client=client),
            )

        result = Devices(
            status=status,
            stage=stage,
            spectrometer=spectrometer,
        )
        self._check(result)
        return result

    def _check(self, dev: Devices):
        check_device_against_interface(dev.stage, devices.Stage, attrs=["egu"])
        check_device_against_interface(dev.spectrometer, devices.Spectrometer, attrs=[])
        check_device_against_interface(dev.status, devices.DscanStatus, attrs=[])
