from os import listdir
from os.path import isfile, isdir, join
import h5py
import numpy as np
import pandas as pd
import datetime as dt
import math
from scipy import interpolate

from sqlite3 import OperationalError

from sotodlib import core
from sotodlib.io import load_book, hkdb
import sotodlib.io.g3tsmurf_utils as utils
from sotodlib.io.load_smurf import G3tSmurf, Observations
from sotodlib.io.g3thk_db import G3tHk, HKFields, HKAgents, HKFiles
import sotodlib.coords.det_match as dm
from so3g.hk import load_range

from sotodlib.core.metadata.loader import IncompleteMetadataError

import sys

ufm_dict = {"c1":["uv38", "uv39", "uv46"],
            "i1":["mv21", "mv24", "mv28"],
            "i3":["mv13", "mv20", "mv34"],
            "i4":["mv14", "mv32", "mv49"],
            "i5":["uv31", "uv42", "uv47"],
            "i6":["mv11", "mv25", "mv26"],
           }

UXM_dict = {"low":{"uv42":{"psat_dark": 28.2, "kappa":None, "G":669}, #Kappa at n=3.0
                   "uv47":{"psat_dark": 31.4, "kappa":None, "G":780},
                   "uv31":{"psat_dark": 31.3, "kappa":None, "G":817},
                   "uv39":{"psat_dark": 22.3, "kappa":None, "G":708},
                   "uv38":{"psat_dark": 26.9, "kappa":None, "G":668},
                   "uv46":{"psat_dark": 33.9, "kappa":None, "G":808},
                   "mv32":{"psat_dark": 3.1, "kappa":978, "G":77},
                   "mv49":None,
                   "mv14":{"psat_dark": 2.7, "kappa":657, "G":59},
                   "mv20":{"psat_dark": 3.2, "kappa":849, "G":71},
                   "mv13":{"psat_dark": 2.9, "kappa":752, "G":66},
                   "mv34":{"psat_dark": 2.8, "kappa":887, "G":69},
                   "mv11":{"psat_dark": 3, "kappa":1004, "G":80},
                   "mv25":{"psat_dark": 3.5, "kappa":944, "G":78},
                   "mv26":{"psat_dark": 3.8, "kappa":1004, "G":80},
                   "mv21":{"psat_dark": 3.0, "kappa":1042, "G":80},
                   "mv24":{"psat_dark": 3.7, "kappa":980, "G":84},
                   "mv28":{"psat_dark": 3.7, "kappa":1004, "G":86}
                },
            "high":{"uv42":{"psat_dark": 30.3, "kappa":None, "G":669}, #Kappa at n=3.0
                   "uv47":{"psat_dark": 33.6, "kappa":None, "G":780},
                   "uv31":{"psat_dark": 33.3, "kappa":None, "G":817},
                   "uv39":{"psat_dark": 23.9, "kappa":None, "G":708},
                   "uv38":{"psat_dark": 28.8, "kappa":None, "G":668},
                   "uv46":{"psat_dark": 36.4, "kappa":None, "G":808},
                   "mv32":{"psat_dark": 8.8, "kappa":978, "G":77},
                   "mv49":None,
                   "mv14":{"psat_dark": 6.6, "kappa":657, "G":59},
                   "mv20":{"psat_dark": 8.7, "kappa":849, "G":71},
                   "mv13":{"psat_dark": 8.6, "kappa":752, "G":66},
                   "mv34":{"psat_dark": 8.6, "kappa":887, "G":69},
                   "mv11":{"psat_dark": 8.1, "kappa":1004, "G":80},
                   "mv25":{"psat_dark": 9.4, "kappa":944, "G":78},
                   "mv26":{"psat_dark": 10.1, "kappa":1004, "G":80},
                   "mv21":{"psat_dark": 7.5, "kappa":1042, "G":80},
                   "mv24":{"psat_dark": 9.5, "kappa":980, "G":84},
                   "mv28":{"psat_dark": 9.7, "kappa":1004, "G":86}
                }
           }
            

_therm_dict = {"c1":"lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_03_T",
              "i1":"lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_15_T",
              "i3":"lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_09_T",
              "i4":"lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_11_T",
              "i5":"lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_01_T",
              "i6":"lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_14_T",
             }

therm_dict = {"c1":"cryo-ls372-lsa22vr.temperatures.Channel_03_T",
              "i1":"cryo-ls372-lsa22vr.temperatures.Channel_15_T",
              "i3":"cryo-ls372-lsa22vr.temperatures.Channel_09_T",
              "i4":"cryo-ls372-lsa22vr.temperatures.Channel_11_T",
              "i5":"cryo-ls372-lsa22vr.temperatures.Channel_01_T",
              "i6":"cryo-ls372-lsa22vr.temperatures.Channel_14_T",
             }

def pwv_interp(filepath="/so/home/jorlo/dev/LAT_analysis/apex_pwv_data.npz", time_cut = 17410*1e5):
    data = {}
    with np.load(filepath, allow_pickle=True) as x:
        for k in x.keys():
            data[k] = x[k]

    flags = np.where((data["timestamp"] >= time_cut))[0]

    for key in data.keys():
        data[key] = data[key][flags]
        
    data["pwv"] = 0.03 + 0.84 * data["pwv"] #APEX to CLASS best fit from Max

    pwv = interpolate.interp1d(data["timestamp"], data["pwv"])
    return pwv

def get_fpa_temps(obs_list):
    fpa_temps = np.zeros( (len(obs_list),))
    cfg = hkdb.HkConfig.from_yaml('/so/home/jorlo/dev/LAT_analysis/hkdb-lat.cfg')
    for o, obs in enumerate(obs_list):
        field = therm_dict[obs["tube_slot"]]
        lspec = hkdb.LoadSpec(
            cfg=cfg, start=obs['start_time'], end=obs['stop_time'],
            fields = [field],
        )
        result = hkdb.load_hk(lspec, show_pb=False)
        try:
            fpa_temps[o] = np.mean(result.data[field][1])
        except KeyError:
            fpa_temps[o] = np.nan
    return fpa_temps

def _get_fpa_temps(obs_list):
    fpa_temps = np.zeros( (len(obs_list),))
    for o, obs in enumerate(obs_list):
        field = _therm_dict[obs["tube_slot"]]
        data = load_range(
                obs['start_time'], obs['stop_time'],
                fields = [field],
                alias = ['fpa_temp'],
                data_dir='/so/level2-daq/lat/hk/'
        )
        try:
            fpa_temps[o] = np.mean(data['fpa_temp'][1])
        except KeyError:
            fpa_temps[o] = np.nan

def get_obs_biases(iva: dict) -> dict:
    """
    This function is exactly how we currently choose biases from IV's.
    
    Do not change these parameters or you will not get the correct chosen bias!
    """
    
    rfrac_range = (0.3, 0.6)
    bias_groups = iva['bias_groups']

    biases={}
    Rfrac = (iva["R"].T / iva['R_n']).T
    in_range = (rfrac_range[0] < Rfrac) & (Rfrac < rfrac_range[1])
    Rn_range=(5e-3, 12e-3)
    
    for bg in bias_groups:
        m = (iva['bgmap'] == bg)
        m = m & (Rn_range[0] < iva['R_n']) & (iva['R_n'] < Rn_range[1])

        if not m.any():
            continue

        nchans_in_range = np.sum(in_range[m, :], axis=0)
        target_idx = np.nanargmax(nchans_in_range)
        biases[bg] = iva['v_bias'][target_idx]
        
    return biases

def get_dark_cal(stream_id: str, platform: str) -> np.array:
    """
    THIS FUNCTION CURRENTLY REFERENCES FILES THAT ONLY EXIST FOR THE SAT
    """
    dark_cal_dir = '/so/home/mrandall/Analysis/IVs/Dark_Cal'
    dark_cal_file = f'{platform}_dark_cal.h5'
    with h5py.File(join(dark_cal_dir, dark_cal_file), 'r') as f:
        return np.array(f[stream_id])

def get_dark_rset(stream_id: str, ot: str) -> dm.ResSet:
    data = get_dark_cal(stream_id, ot) 
    
    north_is_highband = dm.get_north_is_highband(data['band'], data['bg'])
    idx = 0
    resonances = []
    for x in data:
        is_north = north_is_highband ^ (x['band'] < 4)
        res = dm.Resonator(
            idx=idx,
            smurf_band=x['band'], smurf_channel=x['channel'],
            bg=x['bg'], res_freq=x['freq'], is_north=is_north,
            is_optical=~x['masked']
        )
        if res.res_freq >=6000:
            res.res_freq -= 2000
        idx +=1 
        resonances.append(res)
    rset = dm.ResSet(resonances)
    rset.name = f'{stream_id} pton'
    return rset

def find_smurf_tune_file(tune_name: str, ot: str, ufm_name: str) -> str:
    #Get the Tunefile for this obs. Yes this is a dumb brute force check for the file.
    try:
        #Try Level3 data first
        tune_file = None
        tune_num = tune_name[:5]
        tune_setup_filepath = f"/so/data/lat/smurf/smurf_{tune_num}_lat/{ufm_name}"
        for x in listdir(tune_setup_filepath):
            if "uxm_setup" in x or "setup_tune" in x:
                if isfile(join(tune_setup_filepath, x, "outputs", tune_name)):
                    tune_file = join(tune_setup_filepath, x, "outputs", tune_name)
                    break
                    
        return tune_file
    
    #TODO: IDK if this except is valid for LAT
    except FileNotFoundError:
        #Try level2 Data if its not in level3
        tune_file = None
        tune_num = tune_name[:5]
        tune_setup_filepath = f"/so/level2-daq/lat/smurf/{tune_num}/{ufm_name}"
        for x in listdir(tune_setup_filepath):
            if "uxm_setup" in x or "setup_tune" in x:
                if isfile(join(tune_setup_filepath, x, "outputs", tune_name)):
                    tune_file = join(tune_setup_filepath, x, "outputs", tune_name)
                    break
                    
        return tune_file
    
def find_smurf_bgmap_file(bgmap_name: str, ot: str, ufm_name: str) -> str:
    #Try searching in level3 data first!
    try:
        bgmap_file = None
        bgmap_num = bgmap_name[:5]
        bg_setup_filepath = f"/so/data/{platform}/smurf/smurf_{bgmap_num}_{platform}/{ufm_name}"
        #bg_setup_filepath = f"/so/level2-daq/{platform}/smurf/{bgmap_num}/{ufm_name}"
        for x in listdir(bg_setup_filepath):
            if "take_bgmap" in x:
                if isfile(join(bg_setup_filepath, x, "outputs", bgmap_name)):
                    bgmap_file = join(bg_setup_filepath, x, "outputs", bgmap_name)
                    break
                    
        return bgmap_file
    
    except FileNotFoundError:
        #Try searching in level2 if its not in level3
        bgmap_file = None
        bgmap_num = bgmap_name[:5]
        bg_setup_filepath = f"/so/level2-daq/{platform}/smurf/{bgmap_num}/{ufm_name}"
        for x in listdir(bg_setup_filepath):
            if "take_bgmap" in x:
                if isfile(join(bg_setup_filepath, x, "outputs", bgmap_name)):
                    bgmap_file = join(bg_setup_filepath, x, "outputs", bgmap_name)
                    break
                    
        return bgmap_file


def get_obs_from_obs_ids(obs_ids: int, ot: str) -> list[core.metadata.resultset.ResultSet]:
    ctx = core.Context(f'/so/metadata/lat/contexts/smurf_detcal.yaml')

    obs_list = []
    for obs_id in obs_ids: #TODO: why is this a loop and not just a querry?
        obs = ctx.obsdb.query(f"(obs_id=='{obs_id}') and (type=='oper') and (subtype=='iv') and tube_slot == '{ot}'")
        obs_list.append(obs[0])

    return obs_list

def get_obs_from_time(ot: str, start: dt.datetime = dt.datetime(2025,2,20), end: dt.datetime = dt.datetime.now()) -> core.metadata.resultset.ResultSet:
    
    ctx = core.Context('/so/metadata/lat/contexts/smurf_detcal.yaml')
    obs_list = ctx.obsdb.query(f"{end.timestamp()} > timestamp and timestamp > {start.timestamp()} and type=='oper' and subtype=='iv'  and tube_slot == '{ot}'") #IDK why but you have to use format
    
    return obs_list

def load_ivs_from_times(ot: str, start: dt.datetime = dt.datetime(2025,2,20), end: dt.datetime = dt.datetime.now(), ufm_names: list[str] | None=None, full_load: bool=False) -> dict:
    if ufm_names is None:
        ufm_names = ufm_dict[ot]
        
    for ufm in ufm_names:
        if ufm not in ufm_dict[ot]:
            raise ValueError("Error: UFM {} not in OT {}".format(ufm, ot)) 
        
    ctx = core.Context(f'/so/metadata/lat/contexts/smurf_detcal.yaml')

    ivs = {}
    for ufm_name in ufm_names:
        ivs[ufm_name] = {}
        
    obs_list = get_obs_from_time(ot=ot, start=start, end=end)
    
    for i, obs in enumerate(obs_list):
        obs_ufm_name = obs["stream_ids_list"].split("_")[-1]
        try:
            iv_data = load_book.load_smurf_npy_data(ctx, obs['obs_id'], 'iv')
        except:
            print(f"File for {obs['obs_id']} not found!")
            for ufm_name in ufm_names:
                ivs[ufm_name][obs['obs_id']] = None
        for ufm_name in ufm_names:
            if ufm_name == obs_ufm_name:
                if full_load:
                    ivs[ufm_name][obs['obs_id']] = iv_data

                else:
                    ivs[ufm_name][obs['obs_id']] = {}
                    ivs[ufm_name][obs['obs_id']]['p_sat'] = iv_data['p_sat']*1e12
                    ivs[ufm_name][obs['obs_id']]["R"] = iv_data["R"]
                    ivs[ufm_name][obs['obs_id']]["R_n"] = iv_data["R_n"]
                    ivs[ufm_name][obs['obs_id']]["v_bias"] = iv_data["v_bias"]
                    ivs[ufm_name][obs['obs_id']]['bands'] = iv_data['bands']
                    ivs[ufm_name][obs['obs_id']]['channels'] = iv_data['channels']
                    ivs[ufm_name][obs['obs_id']]['bias_groups'] = iv_data['bias_groups']
                    ivs[ufm_name][obs['obs_id']]['bgmap'] = iv_data['bgmap']
                    ivs[ufm_name][obs['obs_id']]['meta'] = {}
                    ivs[ufm_name][obs['obs_id']]['meta']['tunefile'] = iv_data['meta']['tunefile']
                    ivs[ufm_name][obs['obs_id']]['meta']['bgmap_file'] = iv_data['meta']['bgmap_file']

                for start_times in iv_data["start_times"]:
                    ivs[ufm_name][obs['obs_id']]["start_time"] = start_times[0]
                    
                if start_times[0] != 0:
                    break
                
                obs_rfracs = []
                for r, r_n in zip(iv_data["R"], iv_data["R_n"]):
                    rfrac = r / r_n
                    obs_rfracs.append(rfrac)

                ivs[ufm_name][obs['obs_id']]["R_frac"] = obs_rfracs

                obs_biases = get_obs_biases(iv_data)
                ivs[ufm_name][obs['obs_id']]['chosen_biases'] = obs_biases
    return ivs

def detmatch_ivs(ivs, ot, opt_shifted=False):
    """
    This function adds the det_ids and loadings to all of the ivs in ivs.
    
    Args:
        ivs: A dictionary of dictionaries that holds the ivs in the form of:
        ivs = {ufm_mv#: ufm_ivs, ufm_mv#: ufm_ivs...}
            where ufm_ivs is a dictionary of the following form:
            ufm_ivs = {obs_id: obs_iv_data, obs_id: obs_iv_data,...}
            
            So to get the data for a specific obs you take: obs_iv_data = ivs[ufm_name][obs_id]
            ivs is naturally the output of load_ivs_from_time
            
    This function loops over all ufm_names found in ivs and then over all obs found in ivs[ufm_name].
    
    Returns nothing, edits all the iv_data in place to add ivs[ufm_name][obs_id]["det_id"]
    and ivs[ufm_name][obs_id]["loading"]
    """
    
    ctx = core.Context(f'/so/metadata/lat/contexts/smurf_detcal.yaml')
    
    ufm_names = ivs.keys()
    
    for ufm_name in ufm_names:

        #Get info for the PTon Dark PSats
        rset_dark = get_dark_rset(ufm_name, platform)
        dark_cal = get_dark_cal(ufm_name, platform)
        
        #Prepare the Dark cal data to add to the IV data.
        dark_psats = dark_cal['p_sat']
        dark_ks = dark_cal['k']
        dark_ns = dark_cal['n']
        dark_Tcs = dark_cal['Tc']

        #We go through each obs/g3tsmurf id
        for obs_id in list(ivs[ufm_name].keys()):
            if ivs[ufm_name][obs_id] is not None:
                #We just need the oper# at the end of the g3tsmurf id
                obs_iv = ivs[ufm_name][obs_id]
                
                #Prepare the necessary data from the IV
                bands = obs_iv['bands']
                channels = obs_iv['channels']
                bgs = obs_iv['bgmap']
                psats = obs_iv['p_sat']
                
                try:
                    #Get the Tunefile for this obs. Yes this is a dumb brute force check for the file.
                    tune_name = obs_iv['meta']['tunefile'].split('/')[-1]
                    tune_file = find_smurf_tune_file(tune_name, platform, ufm_name)
                    
                    #Get the bgmap_file for this obs. ALso a dumb brute force check for the file.
                    bgmap_name = obs_iv['meta']['bgmap_file'].split('/')[-1]
                    bgmap_file = find_smurf_bgmap_file(bgmap_name, platform, ufm_name)

                    #Get the ctx am and rset for this obs
                    am = ctx.get_meta(obs_id, ignore_missing=True)
                    north_is_highband = dm.get_north_is_highband(bands, bgs)
                    rset = dm.ResSet.from_tunefile(tunefile=tune_file, bgmap_file=bgmap_file, north_is_highband=north_is_highband)
                    
                    #Match this obs to the dark Pton Measurements
                    match_pars = dm.MatchParams(
                    freq_width=1, assigned_bg_mismatch_pen= 100,
                    assigned_bg_unmatched_pen=200000
                    )
                    match = dm.Match(rset, rset_dark, match_pars=match_pars)
                
                except:
                    continue
