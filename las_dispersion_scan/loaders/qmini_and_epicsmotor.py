import os

import ophyd

# from ..devices import Qmini
from las_dispersion_scan.devices import Qmini

qmini_prefix = os.environ.get("QMINI_PREFIX", "LAS:XCS:QMINI:01")
stage_prefix = os.environ.get("MOTOR_PREFIX", "IOC:TST:motor")
qmini_name = os.environ.get("QMINI_NAME", "xcs_las_mmn_02")
stage_name = os.environ.get("MOTOR_NAME", "XCS:LAS:MMN:02")
motor_units = os.environ.get("MOTOR_UNITS", "")

spectrometer = Qmini(qmini_prefix, name=qmini_name)

if motor_units:

    class EpicsMotor(ophyd.EpicsMotor):
        @property
        def egu(self) -> str:
            return motor_units

else:
    EpicsMotor = ophyd.EpicsMotor

stage = EpicsMotor(stage_prefix, name=stage_name)
