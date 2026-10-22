"""Collect IV p_sats, PWVs, elevations and FPA temps per UFM to measure optical loading."""

import argparse as ap
import multiprocessing
import os
from functools import partial

from sotodlib import core

from latcom.utils.common import CTX_DIR, REPO_DIR, add_time_args, save_pickle
from latcom.utils.optical_loading_utils import UXM_dict, get_all_iv_data


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
        default=str(CTX_DIR / "smurf_detsets_local.yaml"),
        help="Path to context file",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=str(REPO_DIR / "ivs"),
        help="Directory to save results to",
    )
    add_time_args(parser, start="2026-06-01", end="2026-07-01")
    return parser


def main() -> None:
    """Collect IV data for every UFM with UXM measurements and save one file per UFM."""
    args = _make_parser().parse_args()

    ctx = core.Context(args.ctx)
    ufms = [ufm for ufm, value in UXM_dict["low"].items() if value is not None]
    obs_lists = [
        ctx.obsdb.query(
            f"{args.end.timestamp()} > timestamp and timestamp > {args.start.timestamp()}"
            f" and type=='oper' and subtype=='iv' and stream_ids_list == 'ufm_{ufm}'"
        )
        for ufm in ufms
    ]

    with multiprocessing.Pool() as pool:
        results = pool.map(partial(get_all_iv_data, ctx_path=args.ctx), obs_lists)

    for ufm, iv_dict in zip(ufms, results):
        if iv_dict is not None:
            save_pickle(iv_dict, os.path.join(args.outdir, f"ivs_{ufm}.pk"))


if __name__ == "__main__":
    main()
