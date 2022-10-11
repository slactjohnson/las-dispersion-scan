"""
`las-dispersion-scan` is the top-level command for accessing various subcommands.

Try:

"""

import argparse
import importlib
import logging

import las_dispersion_scan

DESCRIPTION = str(__doc__)


MODULES = ("gui",)


def _try_import(module):
    relative_module = f".{module}"
    return importlib.import_module(relative_module, "las_dispersion_scan.bin")


def _build_commands():
    global DESCRIPTION
    result = {}
    unavailable = []

    for module in sorted(MODULES):
        try:
            mod = _try_import(module)
        except Exception as ex:
            unavailable.append((module, ex))
        else:
            result[module] = (mod.build_arg_parser, mod.main)
            DESCRIPTION += f"\n    $ las-dispersion-scan {module} --help"

    if unavailable:
        DESCRIPTION += "\n\n"

        for module, ex in unavailable:
            DESCRIPTION += (
                f'\nWARNING: "las-dispersion-scan {module}" is unavailable due to:'
                f"\n\t{ex.__class__.__name__}: {ex}"
            )

    return result


COMMANDS = _build_commands()


def main():
    top_parser = argparse.ArgumentParser(
        prog="las-dispersion-scan",
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    top_parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=las_dispersion_scan.__version__,
        help="Show the las-dispersion-scan version number and exit.",
    )

    top_parser.add_argument(
        "--log",
        "-l",
        dest="log_level",
        default="INFO",
        type=str,
        help="Python logging level (e.g. DEBUG, INFO, WARNING)",
    )

    subparsers = top_parser.add_subparsers(help="Possible subcommands")
    for command_name, (build_func, main) in COMMANDS.items():
        sub = subparsers.add_parser(command_name)
        build_func(sub)
        sub.set_defaults(func=main)

    args = top_parser.parse_args()
    kwargs = vars(args)
    log_level = kwargs.pop("log_level")

    logger = logging.getLogger("las_dispersion_scan")
    logger.setLevel(log_level)
    logging.basicConfig()

    if hasattr(args, "func"):
        func = kwargs.pop("func")
        logger.debug("%s(**%r)", func.__name__, kwargs)
        func(**kwargs)
    else:
        top_parser.print_help()


if __name__ == "__main__":
    main()
