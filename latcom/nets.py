import datetime as dt
import glob
from zoneinfo import ZoneInfo

import dill as pk
import numpy as np
from sotodlib import core
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.tod_ops.flags import get_det_bias_flags

import latcom.utils.net_utils as nu
from latcom.utils.optical_loading import pwv_interp

results = sorted(glob.glob("../results_*.pk"))[-1]
abscals = sorted(glob.glob("../abscals_*.pk"))[-1]

with open(results, "rb") as f:
    result_dict = pk.load(f)

with open(abscals, "rb") as f:
    abscal_dict = pk.load(f)

try:
    nets = sorted(glob.glob("../nets_*.pk"))[-1]
    with open(nets, "rb") as f:
        net_dict = pk.load(f)

except (OSError, FileNotFoundError, IndexError):
    today = dt.datetime.now(tz=ZoneInfo("America/New_York")).date()
    date_str = str(today.month).zfill(2) + str(today.day).zfill(2) + str(today.year)
    nets = f"../nets_{date_str}.pk"
    net_dict = nu.gen_empty_net_dict(abscal_dict)

# ctx = core.Context("../smurf_det_preproc.yaml")

ctx = core.Context("../smurf_det_preproc.yaml")

start = dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)
end = dt.datetime(2026, 12, 21, tzinfo=dt.timezone.utc)
obs_list = ctx.obsdb.query(
    f"{end.timestamp()} > timestamp and timestamp > {start.timestamp()} and type=='obs' and subtype=='cmb'"
)

pwv = pwv_interp()

if "index" in net_dict:
    start_index = net_dict["index"]
else:
    start_index = 0

no_preproc = []

for i in range(start_index, len(obs_list)):
    cur_obs = obs_list[i]

    if i % 100 == 0:
        net_dict["index"] = i
        print("Index: ", net_dict["index"])
        with open(nets, "wb") as f:
            pk.dump(net_dict, f)

    try:  # Much faster than ctx.get_meta
        det_info = ctx.get_det_info(cur_obs["obs_id"])
    except LoaderError:
        print("No meta data for obs {}".format(cur_obs["obs_id"]))
        continue
    wafers = np.unique(det_info["stream_id"])
    bands = np.unique(det_info["wafer.bandpass"])
    bands = np.array([b[1:] for b in bands if len(b) > 1 and b[0] == "f"])

    for j in range(len(wafers)):
        cur_wafer = wafers[j].split("_")[-1]
        if "mv" in cur_wafer:
            if (
                cur_obs["obs_id"] in net_dict[cur_wafer]["090"]["obs"]
                and cur_obs["obs_id"] in net_dict[cur_wafer]["150"]["obs"]
            ):
                print("Already done")
                continue
        elif "uv" in cur_wafer and (
            cur_obs["obs_id"] in net_dict[cur_wafer]["220"]["obs"]
            and cur_obs["obs_id"] in net_dict[cur_wafer]["280"]["obs"]
        ):
            print("Already done")
            continue

        if cur_wafer not in result_dict:
            print(f"No abscal for ufm {cur_wafer}")
            continue

        for band in bands:
            if "mv" in cur_wafer:
                if band == "090":
                    ufm_band = 1
                elif band == "150":
                    ufm_band = 2
            if "uv" in cur_wafer:
                if band == "220":
                    ufm_band = 1
                elif band == "280":
                    ufm_band = 2

            try:
                meta = ctx.get_meta(
                    cur_obs["obs_id"],
                    dets={
                        "dets:stream_id": "ufm_" + str(cur_wafer),
                        "dets:wafer.bandpass": "f" + str(band),
                    },
                )
            except LoaderError:
                print("No meta data for obs {}".format(cur_obs["obs_id"]))
                no_preproc.append(cur_obs["obs_id"])
                continue
            flags = get_det_bias_flags(meta).det_bias_flags
            meta.restrict("dets", ~core.flagman.has_any_cuts(flags))
            wafer_flag = np.array([cur_wafer in ufm for ufm in meta.det_info.stream_id])

            if len(wafer_flag) == 0:
                print("No det_info for obs {}".format(cur_obs["obs_id"]))
                continue

            bp = (meta.det_cal.bg % 4) // 2

            if ufm_band == 1:
                net_flag = wafer_flag * (bp == 0)
            elif ufm_band == 2:
                net_flag = wafer_flag * (bp == 1)

            raw_cal = np.nanmedian(meta.abscal.raw_abscal_rj[net_flag])
            if "noise" in meta.preprocess:
                wnoise = meta.preprocess.noise.white_noise[net_flag]
            elif "noiseT" in meta.preprocess:
                wnoise = meta.preprocess.noiseT.white_noise[net_flag]
            else:
                print(f"Error: no valid noise ken in {meta.preprocess.keys()}")
                continue
            ndets = len(np.where(wnoise != 0)[0])

            net_mes = 1 / np.sqrt(2) * wnoise * raw_cal
            clean_nets = []
            for net in net_mes:
                if net * 1e6 > 100:
                    clean_nets.append(net)
            clean_nets = np.array(clean_nets)
            array_net = np.nansum((clean_nets * 1e6) ** (-2)) ** (-1 / 2)

            net_dict[cur_wafer][band]["raw_cal"].append(raw_cal)
            net_dict[cur_wafer][band]["obs"].append(cur_obs["obs_id"])
            net_dict[cur_wafer][band]["ndets"].append(ndets)
            net_dict[cur_wafer][band]["nets"].append(array_net)
            net_dict[cur_wafer][band]["pwv"].append(pwv(cur_obs["timestamp"]))
            net_dict[cur_wafer][band]["el"].append(meta.obs_info.el_center)
            net_dict[cur_wafer][band]["neps"].append(wnoise)
            net_dict[cur_wafer][band]["phiconv"].append(
                meta.det_cal.phase_to_pW[net_flag]
            )


print("Final Dump")

with open(nets, "wb") as f:
    pk.dump(net_dict, f)

with open("no_preproc.txt", "w") as f:
    f.write("\n".join(no_preproc))
