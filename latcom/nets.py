import datetime as dt
import glob
import multiprocessing
from functools import partial
from zoneinfo import ZoneInfo

import dill as pk
from sotodlib import core

import latcom.utils.net_utils as nu
from latcom.utils.optical_loading import pwv_interp

result_path = sorted(glob.glob("../results_*.pk"))[-1]
abscal_path = sorted(glob.glob("../abscals_*.pk"))[-1]

with open(result_path, "rb") as f:
    result_dict = pk.load(f)

ctx = core.Context("../smurf_det_preproc.yaml")

start = dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)
end = dt.datetime(2026, 12, 21, tzinfo=dt.timezone.utc)
obs_list = ctx.obsdb.query(
    f"{end.timestamp()} > timestamp and timestamp > {start.timestamp()} and type=='obs' and subtype=='cmb'"
)

pwv = pwv_interp()

with multiprocessing.Pool() as pool:
    driver_func = partial(nu.get_nets, abscal_list=result_dict.keys(), pwv=pwv)
    results = pool.map(nu.get_nets, obs_list)

net_dict = nu.parse_net_results(results, abscal_path)

today = dt.datetime.now(tz=ZoneInfo("America/New_York")).date()
date_str = str(today.month).zfill(2) + str(today.day).zfill(2) + str(today.year)
nets = f"../nets_{date_str}.pk"

with open(nets, "wb") as f:
    pk.dump(net_dict, f)
