import os

from ophyd import Component as Cpt
from ophyd import DerivedSignal, Signal
from ophyd.pseudopos import real_position_argument
from ophyd.signal import AttributeSignal
from pcdsdevices.lasers import elliptec
from pcdsdevices.pseudopos import SyncAxis

from ..devices import Qmini
from ..motion import move_with_retries

qmini_prefix = os.environ.get("QMINI_PREFIX", "LAS:LLN:QMINI:01")
stage_prefix = os.environ.get("MOTOR_PREFIX", "IOC:TST:DScan:")
qmini_name = os.environ.get("QMINI_NAME", "las_lln_spec_01")
stage_name = os.environ.get("MOTOR_NAME", "las_lln_ell_01")
motor_units = os.environ.get("MOTOR_UNITS", "mm")

spectrometer = Qmini(qmini_prefix, name=qmini_name)


class EllLinear(elliptec.EllLinear):
    _stop_requested: bool

    def __init__(self, *args, **kwargs):
        self._stop_requested = False
        super().__init__(*args, **kwargs)

    def stop(self, *args, **kwargs):
        self._stop_requested = True
        super().stop(*args, **kwargs)

    @property
    def egu(self) -> str:
        return motor_units

    def set(
        self,
        position: float,
        *,
        wait: bool = True,
        retry_timeout: float = 1.0,
        retry_deadband: float = 0.01,
        max_retries: int = 10,
        timeout: float = 10.0,
        **kwargs
    ):
        self._stop_requested = False
        return move_with_retries(
            self,
            position=position,
            retry_timeout=retry_timeout,
            retry_deadband=retry_deadband,
            max_retries=max_retries,
            timeout=timeout,
        )


class DualElliptec(SyncAxis):
    upstream = Cpt(EllLinear, "", port=0, channel=1, atol=0.05)
    downstream = Cpt(EllLinear, "", port=0, channel=2, atol=0.05)

    user_setpoint = Cpt(AttributeSignal, attr="_user_setpoint_change")
    user_readback = Cpt(DerivedSignal, derived_from="sync.readback")

    scale_downstream = Cpt(Signal, value=1.0)
    scale_upstream = Cpt(Signal, value=1.0)
    offset_downstream = Cpt(Signal, value=0.0)
    offset_upstream = Cpt(Signal, value=0.0)

    @property
    def scales(self) -> dict[str, float]:
        try:
            return {
                "downstream": self.scale_downstream.get(),
                "upstream": self.scale_upstream.get(),
            }
        except AttributeError:
            # TODO: instantiation order means this has to be available before the components
            return {"downstream": 1.0, "upstream": 1.0}

    @scales.setter
    def scales(self, values):
        if not hasattr(self, "scale_downstream"):
            # TODO: instantiation order means this has to be available before the components
            return
        self.scale_downstream.put(values["downstream"])
        self.scale_upstream.put(values["upstream"])

    @property
    def offsets(self) -> dict[str, float]:
        try:
            return {
                "downstream": self.offset_downstream.get(),
                "upstream": self.offset_upstream.get(),
            }
        except AttributeError:
            # TODO: instantiation order means this has to be available before the components
            return {"downstream": 0.0, "upstream": 0.0}

    @offsets.setter
    def offsets(self, values):
        if not hasattr(self, "scale_downstream"):
            # TODO: instantiation order means this has to be available before the components
            return
        self.offset_downstream.put(values["downstream"])
        self.offset_upstream.put(values["upstream"])

    @property
    def _user_setpoint_change(self) -> float:
        try:
            return self.sync.target
        except Exception:
            return 0.0

    @_user_setpoint_change.setter
    def _user_setpoint_change(self, value: float):
        self.sync.move(value, wait=False)

    @property
    def egu(self) -> str:
        return "mm"

    @property
    def precision(self) -> int:
        return 3

    @real_position_argument
    def inverse(self, real_pos):
        """
        Calculate where the sync motor is based on the first real motor.
        The calculation is:
        1. Subtract the offset
        2. Divide by the scale
        """
        self._setup_offsets()
        attr1 = real_pos._fields[0]
        pos1 = real_pos[0]
        calc1 = self.inverse_single(attr1, pos1)
        attr2 = real_pos._fields[1]
        pos2 = real_pos[1]
        calc2 = self.inverse_single(attr2, pos2)
        calc = (calc1 + calc2) / 2
        return self.PseudoPosition(sync=calc)


stage = DualElliptec(stage_prefix, name=stage_name, concurrent=True)

stage.warn_deadband = 0.01
