import argparse as ap
import datetime as dt
import multiprocessing

from sotodlib import core

from latcom.utils.optical_loading import (
    UXM_dict,
    get_all_iv_data,
)


def _make_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(description="Compute optical loading for observations")

    parser.add_argument(
        "--start",
        "-s",
        type=lambda d: dt.datetime.strptime(d, "%Y-%m-%d").astimezone(dt.timezone.utc),
        default="2026-06-01",
        help="Start time for obs",
    )

    parser.add_argument(
        "--end",
        "-e",
        type=lambda d: dt.datetime.strptime(d, "%Y-%m-%d").astimezone(dt.timezone.utc),
        default="2026-07-01",
        help="End time for obs",
    )
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()

    ctx = core.Context(
        "/global/u2/j/jorlo/dev/LAT_Loading/ctxs/smurf_detsets_local.yaml"
    )
    ufms = [ufm for ufm, value in UXM_dict["low"].items() if value is not None]

    obs_lists = []
    for ufm in ufms:
        obs_list = ctx.obsdb.query(
            f"{args.end.timestamp()} > timestamp and timestamp > {args.start.timestamp()} and type=='oper' and subtype=='iv'"
            " and stream_ids_list == 'ufm_{}'".format(ufm)
        )
        obs_lists.append(obs_list)

    with multiprocessing.Pool() as pool:
        results = pool.map(get_all_iv_data, obs_lists)
