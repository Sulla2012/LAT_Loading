from sotodlib import core

import datetime as dt

import numpy as np
import dill as pk

with open("results_05012025.pk", "rb") as f:
    result_dict = pk.load(f)
    
with open("abscals.pk", "rb") as f:
    abscal_dict = pk.load(f)

net_dict = {}

for key in abscal_dict.keys():
    ufm = key.split("_")[4]
    freq = key.split("_")[5]
    if ufm in abscal_dict.keys():
        continue
    if "090" in freq or "150" in freq:
        net_dict[ufm] = {"090":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[]}, "150":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[]}}
    else:
        net_dict[ufm] = {"220":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[]}, "280":{"chi":[], "obs":[], "ndets":[], "nets":[], "raw_cal":[]}}
    
ctx = core.Context('./smurf_det_preproc.yaml')

start = dt.datetime(2025,3,1, tzinfo=dt.timezone.utc)
end = dt.datetime(2025,4,28, tzinfo=dt.timezone.utc)
obs_list = ctx.obsdb.query(
    f"{end.timestamp()} > timestamp and timestamp > {start.timestamp()} and type=='obs' and subtype=='cmb'"
)

for i in range(len(obs_list)):
    cur_obs = obs_list[i]
    wafers = cur_obs["stream_ids_list"].split(",")
    cur_wafer = wafers[0].split("_")[-1]
    
    if cur_wafer not in result_dict.keys():
        print("No abscal for ufm {}".format(cur_wafer))
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
        if np.abs(times[5]-cur_obs["timestamp"])/3600 < 24 and 25 <= closest_chi and closest_chi < 1000 : #If most recent obs within a day
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

with open("nets.pk", "wb") as f:
    pk.dump(net_dict, f)