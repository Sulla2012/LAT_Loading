"""Download APEX PWV data from the ESO archive, used by pwv_interp."""

import argparse as ap
import datetime as dt
from io import StringIO

import numpy as np
import requests

from latcom.utils.common import APEX_PWV_PATH, add_time_args

APEX_DATA_URL = "http://archive.eso.org/wdb/wdb/eso/meteo_apex/query"


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
        default=str(APEX_PWV_PATH),
        help="Path to save the PWV npz to",
    )
    add_time_args(
        parser,
        start="2025-05-01",
        end=(dt.date.today() + dt.timedelta(days=1)).isoformat(),
    )
    return parser


def main() -> None:
    """Query the APEX PWV for the time range and save it as an npz."""
    args = _make_parser().parse_args()

    fmt = "%Y-%m-%dT%H:%M:%S"
    request = requests.post(
        APEX_DATA_URL,
        data={
            "wdbo": "csv/download",
            "max_rows_returned": 9000000,
            "start_date": f"{args.start.strftime(fmt)}..{args.end.strftime(fmt)}",
            "tab_pwv": "on",
            "shutter": "SHUTTER_OPEN",
        },
        timeout=600,
    )
    request.raise_for_status()

    data = np.genfromtxt(
        StringIO(request.text),
        delimiter=",",
        skip_header=2,
        converters={0: dt.datetime.fromisoformat},
        dtype=[("dates", dt.datetime), ("pwv", float)],
    )
    np.savez(
        args.outpath,
        dates=data["dates"],
        timestamp=[d.timestamp() for d in data["dates"]],
        pwv=data["pwv"],
    )
    print(f"Saved {args.outpath}")


if __name__ == "__main__":
    main()
