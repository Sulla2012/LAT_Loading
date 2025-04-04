import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from sotodlib import core
import sotodlib.io.load_book as lb
import so3g
import sotodlib.tod_ops as tod_ops

from so3g.hk import HKTree

def get_ivs(tube: str, date: dt.datetime = dt.datetime(2025,2,20)):
    
    if tube not in ["c1", "i1", "i2", "i3", "i4", "i5", "i6", "o1", "o2", "o3", "o4", "o5", "o6"]:
        raise ValueError("Invalid tube {tube}")

    ctx = core.Context('/so/metadata/lat/contexts/smurf_detsets.yaml')
    obs_list = ctx.obsdb.query(f"timestamp > {date.timestamp()} and "
                           "type=='oper' and subtype=='iv' and tube_slot == '{}' and wafer_slots_list LIKE '%2'".format(tube)) #IDK why but you have to use format

    keys = []
    for obs in obs_list:
        obs_id = obs['obs_id']
        meta = ctx.get_meta(obs_id, )
        for det_id in  meta.det_info.readout_id:
            keys.append('_'.join([det_id.split("_")[2], *det_id.split("_")[-2:]]))
    keys = set(keys)

    #TODO: inefficient to be loading all these meta data twice, but I need the keys first
    det_dict = {key: {"p_sat":[], "bgmap":[]} for key in keys}
    for j, obs in enumerate(obs_list):
        try:
            iv_data = lb.load_smurf_npy_data(ctx, obs['obs_id'], 'iv_analysis')
        except FileNotFoundError: continue
        obs_id = obs['obs_id']
        meta = ctx.get_meta(obs_id, )
        for i, det_id in enumerate(meta.det_info.readout_id):
            cur_det_id = '_'.join([det_id.split("_")[2], *det_id.split("_")[-2:]])
            if iv_data["p_sat"][i] is not np.nan:
                det_dict[cur_det_id]["p_sat"].append(iv_data["p_sat"][i]*1e12)
            else:
                det_dict[cur_det_id]["p_sat"].append(np.nan)
            if j == 0:
                det_dict[cur_det_id]["bgmap"] = iv_data["bgmap"][i] 
    return det_dict
    
def get_dark_dets_thresh(det_dict: dict, threshold: float=0.001):
    psat_vars = []
    dark_dets = {}
    for det in det_dict:
        psat_var = np.nanvar(det_dict[det]["p_sat"])
        psat_vars.append(psat_var)
        if psat_var < threshold:
            dark_dets[det]={"var":psat_var, "bgmap":det_dict[det]["bgmap"]}

    return dark_dets, psat_vars

def get_dark_dets_num(det_dict: dict, num: int=36):
    psat_vars = []
    dets = []
    dark_dets = {}

    for det in det_dict:
        psat_var = np.nanvar(det_dict[det]["p_sat"])
        psat_vars.append(psat_var)
        dets.append(det)

    psat_vars = np.array(psat_vars)
    dets = np.array(dets) 

    p = psat_vars.argsort()
    dets = dets[p]
    temp_psat_vars = psat_vars[p]
    i = 0
    j = 0
    while i < num:
        det = dets[j]
        j += 1
        if det_dict[det]["bgmap"] != -1:
            dark_dets[det]={"var":temp_psat_vars[j], "bgmap":det_dict[det]["bgmap"]}
            i += 1

    return dark_dets, psat_vars

