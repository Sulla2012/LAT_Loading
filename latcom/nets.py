import argparse as ap
import datetime as dt
import glob
import multiprocessing
from functools import partial
from zoneinfo import ZoneInfo

import dill as pk
from sotodlib import core

import latcom.utils.net_utils as nu
from latcom.utils.optical_loading import pwv_interp


def _make_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(
        description="Compute abscal factors for Saturn/Mars observations"
    )
    parser.add_argument(
        "--ctx_path",
        "-c",
        default="../smurf_det_preproc.yaml",
        help="Path to context file. ",
    )

    parser.add_argument(
        "--start",
        "-s",
        type=lambda d: dt.datetime.strptime(d, "%Y-%m-%d").astimezone(dt.timezone.utc),
        default="2025-04-01",
        help="Start time for obs",
    )

    parser.add_argument(
        "--end",
        "-e",
        type=lambda d: dt.datetime.strptime(d, "%Y-%m-%d").astimezone(dt.timezone.utc),
        default="2027-01-01",
        help="Start time for obs",
    )
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()

    result_path = sorted(glob.glob("../results_*.pk"))[-1]

    with open(result_path, "rb") as f:
        result_dict = pk.load(f)

    ctx = core.Context(args.ctx_path)

    obs_list = ctx.obsdb.query(
        f"{args.end.timestamp()} > timestamp and timestamp > {args.start.timestamp()} and type=='obs' and subtype=='cmb'"
    )

    pwv = pwv_interp()

    obs_ids = []
    for i, obs in enumerate(obs_list):
        obs_ids.append(str(obs["obs_id"]))

    with multiprocessing.Pool() as pool:
        driver_func = partial(
            nu.get_nets,
            abscal_list=list(result_dict.keys()),
            pwv=pwv,
            ctx_path=args.ctx_path,
        )
        results = pool.map(driver_func, list(obs_ids))

    net_dict = nu.parse_net_results(results)

    today = dt.datetime.now(tz=ZoneInfo("America/New_York")).date()
    date_str = str(today.month).zfill(2) + str(today.day).zfill(2) + str(today.year)
    nets = f"../nets_{date_str}.pk"

    with open(nets, "wb") as f:
        pk.dump(net_dict, f)
