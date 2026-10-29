"""Download JPL Horizons planet ephemerides at the SO site, used for planet solid angles."""

import argparse as ap
from pathlib import Path

import astropy.units as u
from astropy.table import Table, vstack
from astroquery.jplhorizons import Horizons

from latcom.utils.abscal_utils import EPH_PATH
from latcom.utils.common import add_time_args

# Horizons ids of the planets to query
PLANET_IDS = {
    "jupiter": "599",
    "saturn": "699",
    "mars": "499",
    "uranus": "799",
    "neptune": "899",
}


def _debabyl(deg: float, arcmin: float, arcsec: float) -> float:
    """
    Convert sexagesimal degrees to decimal degrees.

    Parameters
    ----------
    deg : float
        Degrees.
    arcmin : float
        Arcminutes.
    arcsec : float
        Arcseconds.

    Returns
    -------
    deg : float
        Decimal degrees.
    """
    return deg + arcmin / 60 + arcsec / 3600


SO_SITE = {
    "lon": -_debabyl(67, 47, 15.68) * u.deg,
    "lat": -_debabyl(22, 57, 39.47) * u.deg,
    "elevation": 5188.0 / 1000.0 * u.km,
}


def _make_parser() -> ap.ArgumentParser:
    """
    Make the argument parser.

    Returns
    -------
    parser : ap.ArgumentParser
        The parser.
    """
    parser = ap.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outpath",
        "-o",
        default=str(EPH_PATH),
        help="Path to save the ephemerides fits table to",
    )
    parser.add_argument(
        "--step", default="20m", help="Horizons sampling step of the ephemerides"
    )
    add_time_args(parser, start="2025-01-01", end="2027-01-01")
    return parser


def main() -> None:
    """Query the ephemerides of each planet and save them as one fits table."""
    args = _make_parser().parse_args()

    epochs = {
        "start": args.start.strftime("%Y-%m-%d"),
        "stop": args.end.strftime("%Y-%m-%d"),
        "step": args.step,
    }
    quantities = ",".join(str(i) for i in range(1, 44))  # all observer columns

    tabs = []
    for planet, pid in PLANET_IDS.items():
        eph = Horizons(id=pid, location=SO_SITE, epochs=epochs).ephemerides(
            quantities=quantities
        )
        tab = Table(eph)
        tab["planet"] = planet
        print(planet, len(tab), set(tab["targetname"]))
        tabs.append(tab)

    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    vstack(tabs).write(args.outpath, overwrite=True)
    print(f"Saved {args.outpath}")


if __name__ == "__main__":
    main()
