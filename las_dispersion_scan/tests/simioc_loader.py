import ophyd

from ..devices import Qmini

spectrometer = Qmini("IOC:TST:DScan:Qmini", name="sim-qmini")
stage = ophyd.EpicsMotor("IOC:TST:DScan:m1", name="sim-m1")
