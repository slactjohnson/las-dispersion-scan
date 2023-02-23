import os

import ophyd

# from ..devices import Qmini
from las_dispersion_scan.devices import Qmini
from pcdsdevices.lasers import elliptec

qmini_prefix = os.environ.get("QMINI_PREFIX", "LAS:LLN:QMINI:01")
stage_prefix = os.environ.get("MOTOR_PREFIX", "LAS:ELL:LLN:TEST")
qmini_name = os.environ.get("QMINI_NAME", "las_lln_spec_01")
stage_name = os.environ.get("MOTOR_NAME", "las_lln_ell_01")
motor_units = os.environ.get("MOTOR_UNITS", "mm")

spectrometer = Qmini(qmini_prefix, name=qmini_name)

if motor_units:

    class EllLinear(elliptec.EllLinear):
        @property
        def egu(self) -> str:
            return motor_units

else:
    EllLinear = elliptec.EllLinear

stage = EllLinear(stage_prefix, name=stage_name, atol=0.010)
