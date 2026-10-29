"""Print the number of valid abscal measurements per UFM and band in an epoch."""

import argparse as ap
import datetime as dt

import dill as pk
import numpy as np

from latcom.utils.common import add_time_args, bands_for_ufm, ufm_dict


def _make_parser() -> ap.ArgumentParser:
    """
    Make the argument parser.

    Returns
    -------
    parser : ap.ArgumentParser
        The parser.
    """
    parser = ap.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Path to a results_<MMDDYYYY>.pk")
    add_time_args(
        parser,
        start="2026-01-01",
        end=(dt.date.today() + dt.timedelta(days=1)).isoformat(),
    )
    return parser


def count_valid(
    result_dict: dict, start: dt.datetime, end: dt.datetime
) -> dict[str, dict[str, int]]:
    """
    Count the valid abscal measurements of every UFM and band in [start, end).
    A measurement is valid if its abscal and relcal are finite, i.e. it can be
    used by make_db.

    Parameters
    ----------
    result_dict : dict
        Results by [ufm][band][field], see abscal_utils.make_results_dict.
    start : dt.datetime
        Start of the epoch.
    end : dt.datetime
        End of the epoch.

    Returns
    -------
    counts : dict[str, dict[str, int]]
        Number of valid measurements by [ufm][band], for every UFM in ufm_dict.
    """
    counts = {}
    for ufms in ufm_dict.values():
        for ufm in ufms:
            counts[ufm] = {}
            for band in bands_for_ufm(ufm):
                entry = result_dict.get(ufm, {}).get(band, {})
                time = np.asarray(entry.get("time", []), dtype=float)
                cal = np.asarray(entry.get("cal", []), dtype=float)
                relcal = np.asarray(entry.get("relcal", np.ones_like(cal)), dtype=float)
                valid = (
                    (time >= start.timestamp())
                    & (time < end.timestamp())
                    & np.isfinite(cal)
                    & np.isfinite(relcal)
                )
                counts[ufm][band] = int(np.count_nonzero(valid))
    return counts


def main() -> None:
    """Load the results and print the counts, grouped by tube."""
    args = _make_parser().parse_args()
    with open(args.results, "rb") as f:
        result_dict = pk.load(f)

    counts = count_valid(result_dict, args.start, args.end)
    print(f"Valid abscals in [{args.start:%Y-%m-%d}, {args.end:%Y-%m-%d}) UTC")
    for ot, ufms in ufm_dict.items():
        for ufm in ufms:
            bands = "  ".join(f"{band}: {n:4d}" for band, n in counts[ufm].items())
            print(f"{ot:3s} {ufm:7s} {bands}")


if __name__ == "__main__":
    main()
