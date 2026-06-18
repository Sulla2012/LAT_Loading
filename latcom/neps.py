import argparse as ap
import datetime as dt
import multiprocessing
from functools import partial
from zoneinfo import ZoneInfo

import dill as pk
from sotodlib import core

import latcom.utils.net_utils as nu
from latcom.utils.optical_loading import (
    aso_tubes,
    lf_tubes,
    pwv_interp,
    so_nominal_tubes,
)


def _make_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(
        description="Compute abscal factors for Saturn/Mars observations"
    )
    parser.add_argument(
        "--ctx_path",
        "-c",
        default="../ctxs/",
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

    ctx = core.Context(args.ctx_path + "smurf_det_preproc.yaml")

    obs_list = ctx.obsdb.query(
        f"{args.end.timestamp()} > timestamp and timestamp > {args.start.timestamp()} and type=='obs' and subtype=='cmb'"
    )

    pwv = pwv_interp()

    obs_ctx_list = []
    for i, obs in enumerate(obs_list):
        ot = str(obs["obs_id"]).split("_")[2][3:]
        obs_id = str(obs["obs_id"])
        if ot in so_nominal_tubes:
            obs_ctx_list.append((obs_id, args.ctx_path + "/preprocess_nominal.yaml"))
        elif ot in aso_tubes:
            obs_ctx_list.append((obs_id, args.ctx_path + "/preprocess.yaml"))
        elif ot in lf_tubes:
            obs_ctx_list.append((obs_id, args.ctx_path + "/preprocess_lf.yaml"))

    with multiprocessing.Pool() as pool:
        driver_func = partial(
            nu.get_neps,
            pwv=pwv,
        )
        results = pool.map(driver_func, obs_ctx_list)
    nep_dict = nu.parse_nep_results(results)

    today = dt.datetime.now(tz=ZoneInfo("America/New_York")).date()
    date_str = str(today.month).zfill(2) + str(today.day).zfill(2) + str(today.year)
    neps = f"../neps_{date_str}.pk"

    with open(neps, "wb") as f:
        pk.dump(nep_dict, f)
