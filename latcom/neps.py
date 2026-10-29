"""Compute detector NEPs for CMB observations. Unlike nets.py, no abscals are needed."""

import argparse as ap
import multiprocessing
import os
from functools import partial

from sotodlib import core

import latcom.utils.net_utils as nu
from latcom.utils.common import (
    CTX_DIR,
    REPO_DIR,
    add_time_args,
    date_str,
    pwv_interp,
    save_pickle,
)


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
        "--ctx",
        "-c",
        default=str(CTX_DIR / "smurf_det_preproc.yaml"),
        help="Path to context file used to query observations",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=str(REPO_DIR / "nets"),
        help="Directory to save results to",
    )
    add_time_args(parser, start="2025-04-01", end="2027-01-01")
    return parser


def main() -> None:
    """Compute NEPs for all CMB observations in the time range and save them."""
    args = _make_parser().parse_args()

    obs_ctx_list = nu.get_obs_ctx_list(core.Context(args.ctx), args.start, args.end)
    driver_func = partial(nu.get_neps, pwv=pwv_interp())
    with multiprocessing.Pool() as pool:
        results = pool.map(driver_func, obs_ctx_list)

    save_pickle(
        nu.parse_results(results, nu.NEP_FIELDS),
        os.path.join(args.outdir, f"neps_{date_str()}.pk"),
    )


if __name__ == "__main__":
    main()
