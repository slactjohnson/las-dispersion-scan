import enum

import numpy as np
import pypret


class PulseAnalysisMethod(str, enum.Enum):
    """ "
    pypret.Retriever pulse analysis method.
    """

    frog = "frog"
    tdp = "tdp"
    dscan = "dscan"
    miips = "miips"
    ifrog = "ifrog"


class NonlinearProcess(str, enum.Enum):
    """ "
    pypret.Retriever non-linear process method.
    """

    shg = "shg"
    xpw = "xpw"
    thg = "thg"
    sd = "sd"
    pg = "pg"


class Material(str, enum.Enum):
    """ "
    pypret.Retriever material.
    """

    fs = "FS"
    bk7 = "BK7"
    gratinga = "gratinga"
    gratingc = "gratingc"

    def get_coefficient(self, wedge_angle: float) -> float:
        """
        Calculate the material coefficient for pypret MeshData, provided the
        wedge angle.

        Parameters
        ----------
        wedge_angle : float
            The wedge angle.

        Returns
        -------
        float
            The material coefficient.
        """
        if self in {Material.gratinga, Material.gratingc}:
            return 4.0

        if self in {Material.bk7, Material.fs}:
            return np.tan(wedge_angle * np.pi / 180) * np.cos(wedge_angle * np.pi / 360)

        raise ValueError("Unsupported material type")

    @property
    def pypret_material(self) -> pypret.material.BaseMaterial:
        """The pypret material."""
        return {
            Material.fs: pypret.material.FS,
            Material.bk7: pypret.material.BK7,
            Material.gratinga: pypret.material.gratinga,
            Material.gratingc: pypret.material.gratingc,
        }[self]


class RetrieverSolver(str, enum.Enum):
    copra = "copra"
    gpa = "gpa"
    gp_dscan = "gp-dscan"
    pcgpa = "pcgpa"
    pie = "pie"
    lm = "lm"
    bfgs = "bfgs"
    de = "de"
    nelder_mead = "nelder-mead"
