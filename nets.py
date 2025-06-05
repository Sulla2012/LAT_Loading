from sotodlib import core

import datetime as dt

import numpy as np
import dill as pk
from numba import prange

from optical_loading import pwv_interp

with open("results_05272025.pk", "rb") as f:
    result_dict = pk.load(f)
    
with open("abscals.pk", "rb") as f:
    abscal_dict = pk.load(f)

try:
    with open("nets.pk", "rb") as f:
        net_dict = pk.load(f)
    for key in abscal_dict.keys():
        ufm = key.split("_")[4]
        freq = key.split("_")[5]
        if ufm in abscal_dict.keys():
            continue
        if "090" in freq or "150" in freq:
            net_dict[ufm] = {"090":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}, "150":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}}
        else:
            net_dict[ufm] = {"220":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}, "280":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}}

except:
    net_dict = {}

    for key in abscal_dict.keys():
        ufm = key.split("_")[4]
        freq = key.split("_")[5]
        if ufm in abscal_dict.keys():
            continue
        if "090" in freq or "150" in freq:
            net_dict[ufm] = {"090":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}, "150":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}}
        else:
            net_dict[ufm] = {"220":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}, "280":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[], "el":[], "pwv":[], "neps":[], "phiconv":[]}}
    
ctx = core.Context('./smurf_det_preproc.yaml')

start = dt.datetime(2025,4,17, tzinfo=dt.timezone.utc)
end = dt.datetime(2025,5,27, tzinfo=dt.timezone.utc)
obs_list = ctx.obsdb.query(
    f"{end.timestamp()} > timestamp and timestamp > {start.timestamp()} and type=='obs' and subtype=='cmb'"
)

pwv = pwv_interp()

for i in prange(len(obs_list)):
    cur_obs = obs_list[i]
    wafers = cur_obs["stream_ids_list"].split(",")
    for i in range(len(wafers)):
        cur_wafer = wafers[i].split("_")[-1]

        if cur_wafer not in result_dict.keys():
            print("No abscal for ufm {}".format(cur_wafer))
            continue
        if cur_wafer in net_dict.keys():
            if "mv" in cur_wafer: 
                if cur_obs["obs_id"] in net_dict[cur_wafer]["090"]["obs"] and cur_obs["obs_id"] in net_dict[cur_wafer]["150"]["obs"]:
                    continue
            elif "uv" in cur_wafer:
                if cur_obs["obs_id"] in net_dict[cur_wafer]["220"]["obs"] and cur_obs["obs_id"] in net_dict[cur_wafer]["280"]["obs"]:
                    continue
        try:
            meta = ctx.get_meta(cur_obs["obs_id"])
        except:
            print("No meta data for obs {}".format(cur_obs["obs_id"]))
            continue
        for ufm_band in [1,2]:
            if "mv" in cur_wafer:
                if ufm_band == 1:
                    band = "090"
                elif ufm_band == 2:
                    band = "150"
            if "uv" in cur_wafer:
                if ufm_band == 1:
                    band = "220"
                elif ufm_band == 2:
                    band = "280"
            wafer_flag = np.array([cur_wafer in ufm for ufm in meta.det_info.stream_id])

            bp = (meta.det_cal.bg % 4) // 2

            if ufm_band == 1:
                net_flag = wafer_flag * (bp==0)
            elif ufm_band == 2:
                net_flag = wafer_flag * (bp==1)

            try: 
                times = np.array([float(label.split("_")[0]) for label in result_dict[cur_wafer][band]["obs"]])
            except KeyError:
                continue

            closest_idx = np.argmin(np.abs(times-cur_obs["timestamp"]))
            closest_obs = times[closest_idx]
            closest_chi = result_dict[cur_wafer][band]["chi"][closest_idx]
            if np.abs(times[closest_idx]-cur_obs["timestamp"])/3600 < 24 and 100< closest_chi and closest_chi<500: #If most recent obs within a day
                raw_cal = result_dict[cur_wafer][band]["raw_cal"][closest_idx]
                chisq = closest_chi
            else:
                raw_cal = np.mean(result_dict[cur_wafer][band]["raw_cal"])
                chisq = 999999
            ndets = len(np.where((meta.preprocess.noise.white_noise[net_flag] != 0))[0])

            net_mes = 1/np.sqrt(2) * meta.preprocess.noise.white_noise[net_flag] * raw_cal * meta.det_cal.phase_to_pW[net_flag]
            clean_nets = []
            for net in net_mes:
                if net*1e6 >= 125:
                    clean_nets.append(net)
            clean_nets = np.array(clean_nets)
            array_net = np.nansum((clean_nets*1e6)**(-2))**(-1/2)

            net_dict[cur_wafer][band]["raw_cal"].append(raw_cal)
            net_dict[cur_wafer][band]["chi"].append(chisq)
            net_dict[cur_wafer][band]["obs"].append(cur_obs["obs_id"])
            net_dict[cur_wafer][band]["ndets"].append(ndets)
            net_dict[cur_wafer][band]["nets"].append(array_net)
            net_dict[cur_wafer][band]["pwv"].append(pwv(cur_obs["timestamp"]))
            net_dict[cur_wafer][band]["el"].append(meta.obs_info.el_center)
            net_dict[cur_wafer][band]["neps"].append(meta.preprocess.noise.white_noise[net_flag] * meta.det_cal.phase_to_pW[net_flag])
            net_dict[cur_wafer][band]["phiconv"].append(meta.det_cal.phase_to_pW[net_flag]) 
with open("nets.pk", "wb") as f:
    pk.dump(net_dict, f)
