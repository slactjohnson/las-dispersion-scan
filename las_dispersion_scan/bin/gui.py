"""The main laser dispersion scan GUI."""

import argparse
import logging
import signal
import sys
from typing import List, Optional

import typhos
from pydm.exception import install as install_exception_handler
from qtpy import QtWidgets

from .. import utils
from ..widgets import DscanMain

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def _sigint_handler(signal, frame):
    logger.info("Caught Ctrl-C (SIGINT); exiting.")
    sys.exit(1)


def _configure_stylesheet(paths: Optional[List[str]] = None) -> str:
    """
    Configure stylesheets for the d-scan GUI.

    Parameters
    ----------
    paths : List[str], optional
        A list of paths to stylesheets to load.
        Defaults to those packaged in las-dispersion-scan.

    Returns
    -------
    str
        The full stylesheet.
    """
    app = QtWidgets.QApplication.instance()
    typhos.use_stylesheet()

    if paths is None:
        paths = [
            str(utils.SOURCE_PATH / "ui" / "stylesheet.qss"),
            str(utils.SOURCE_PATH / "ui" / "pydm.qss"),
        ]

    stylesheets = [app.styleSheet()]

    for path in paths:
        with open(path, "rt") as fp:
            stylesheets.append(fp.read())

    full_stylesheet = "\n\n".join(stylesheets)

    app.setStyleSheet(full_stylesheet)
    return full_stylesheet


def configure_ophyd():
    """Configure ophyd defaults."""
    from ophyd.signal import EpicsSignalBase

    EpicsSignalBase.set_defaults(
        timeout=10.0,
        connection_timeout=10.0,
        auto_monitor=True,
    )


def build_arg_parser(argparser=None):
    if argparser is None:
        argparser = argparse.ArgumentParser()

    argparser.description = DESCRIPTION
    argparser.formatter_class = argparse.RawTextHelpFormatter

    # argparser.add_argument(
    #     "argument_name",
    #     type=str,
    #     help="Get help on this.",
    # )

    return argparser


def main(screen: str = "main", stylesheet: Optional[str] = None) -> None:
    """Launch the d-scan GUI"""
    signal.signal(signal.SIGINT, _sigint_handler)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    install_exception_handler()
    configure_ophyd()

    try:
        _configure_stylesheet(paths=[stylesheet] if stylesheet else None)
    except Exception:
        logger.exception("Failed to load stylesheet; things may look odd...")

    try:
        if screen == "main":
            widget = DscanMain()
        else:
            raise ValueError(f"Unexpected screen type: {screen}")
        widget.show()
    except Exception:
        logger.exception("Failed to load user interface")
        raise
    app.exec_()


if __name__ == "__main__":
    main()
