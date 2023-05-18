import os

from ..devices import EllLinear, Qmini

qmini_prefix = os.environ.get("QMINI_PREFIX", "LAS:LLN:QMINI:01")
stage_prefix = os.environ.get("MOTOR_PREFIX", "LAS:ELL:LLN:TEST")
qmini_name = os.environ.get("QMINI_NAME", "las_lln_spec_01")
stage_name = os.environ.get("MOTOR_NAME", "las_lln_ell_01")
motor_units = os.environ.get("MOTOR_UNITS", "mm")

spectrometer = Qmini(qmini_prefix, name=qmini_name)
stage = EllLinear(stage_prefix, name=stage_name, motor_units=motor_units, atol=0.010)
