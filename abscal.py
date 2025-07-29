from pixell import utils, enmap, bunch, reproject, colors, coordinates

import numpy as np

import map_utils as mu
from optical_loading import pwv_interp, keys_from_wafer, bandpass_interp
from mars_temps import T_b

from sotodlib import core

from astropy import units as u
from astropy import constants as consts

import datetime as dt
import dill as pk
import os
import h5py

def data_to_cal_factor(p_meas, beam_solid_angle, band, wafer, mars_diameter, obs_id):
    fiducial_solid_angle = mu.angular_diameter_to_solid_angle(mars_diameter)

    timestamp = str(obs_id)
    planet_temp = T_b[timestamp]
    
    fill_factor = (fiducial_solid_angle / (beam_solid_angle * 1e-6))
    t_eff_planet = planet_temp[band] *fill_factor
    cal_factor = t_eff_planet / p_meas # K -> pW
    
    bandpass = bandpasses[band]
    
    opt_eff = ((1/(cal_factor)*u.pW/u.K)/(consts.k_B * bandpass * u.GHz)).to(1)
    
    return cal_factor, opt_eff


fwhm_cuts = {"090": [1.8, 2.3],
             "150": [1.3, 1.6],
             "220": [0.7, 1.1],
             "280": [0.7, 1.0],
            }

bandpasses = {"090": 28.83, #Im 90 percent sure these come from LAT_MF/UHF_bands.csv but Im not sure
              "150": 29.49,
              "220": 55.54,
              "280": 46.74
             }

if __name__ == '__main__':
    with open("atmosphere_eff.pk", "rb") as f:
        atmosphere_eff = pk.load(f)

    fiducial_elevation = 50
    fiducial_pwv = 1 # mm
    el_key = "50" #hardcoded :(
    pwv = pwv_interp()

    fpath="/so/home/saianeesh/data/beams/lat/source_maps/per_obs/fits/beam_pars.h5"
    f = h5py.File(fpath, mode="r")                                         
    obs_ids = []                                            
    times = []                                              
    stream_ids = []                                         
    bands = []                                              
    for o in f.keys():                                     
        for s in f[o].keys():                                 
            for b in f[o][s].keys():                        
                obs_ids += [o]                              
                times += [float(o.split("_")[1])]            
                stream_ids += [s]                              
                bands += [b]                                 

    limit_bands = ["f090", "f150", "f220", "f280"]            
    msk = np.isin(bands, limit_bands)                         
    obs_ids = np.array(obs_ids)[msk]                         
    times = np.array(times)[msk]                             
    stream_ids = np.array(stream_ids)[msk]                   
    bands = np.array(bands)[msk]                            

    amans = np.array([
        core.AxisManager.load(f[os.path.join(o, s, b)])
        for o, s, b in zip(obs_ids, stream_ids, bands)
    ])

    cal_dict = {}

    for i, aman in enumerate(amans):

        obs_id = obs_ids[i].split("_")[1]
        wafer = stream_ids[i]
        band = bands[i][1:]
        tb_time = 0
        for key in T_b.keys():
            if np.abs(int(obs_id) - int(key)) <= 300: #Within 5 minutes is OK
                tb_time = key
                break
        if not tb_time:
            print("No Mars data for obs {}".format(obs_id))
            continue         
        

        ufm_type, ufm_band = keys_from_wafer(wafer, band)
        
        fitted_fwhm = np.sqrt(aman.fwhm_dec**2 + aman.fwhm_ra**2).to(u.arcmin).value
        data_solid_angle = aman.data_solid_angle_corr
        amp = aman.amp.value
        
        if fwhm_cuts[band][1] < fitted_fwhm or fitted_fwhm < fwhm_cuts[band][0]:
            continue        

        #Get pwv/el adjustment
        start_date = dt.datetime.utcfromtimestamp(int(obs_id)) - dt.timedelta(days=1)
        end_date = dt.datetime.utcfromtimestamp(int(obs_id)) + dt.timedelta(days=1)

        pwv_idx = np.where(np.array([np.abs(pwv - fiducial_pwv) < 0.1 for pwv in atmosphere_eff['pwv']]))[0]

        try:
            pwv_obs = pwv(obs_id)
        except:
            print("obs {} outside of pwv range".format(obs_id))
            continue
        if pwv_obs > 2.5:
            continue
        ctx = core.Context('/so/metadata/lat/contexts/smurf_detsets.yaml')
        meta = ctx.get_meta(paths[0].split("/")[-2])

        el_obs = meta.obs_info.el_center

        if el_obs > 90:
            el_obs = 180 - el_obs

        obs_idx_pwv = np.where(np.isclose(np.abs(atmosphere_eff['pwv'] - pwv_obs), np.min(np.abs(atmosphere_eff['pwv'] - pwv_obs))))[0][0]
        try:
            obs_key_el = [el for el in atmosphere_eff['LF']["LF_1"].keys() if np.abs(int(el) - el_obs) < 2.5][0]
        except:
            print("El {} for obs {} out of range".format(el_obs, paths[0].split("/")[-2]))
            continue
            
        t_atm_obs = atmosphere_eff[ufm_type][ufm_band][obs_key_el][obs_idx_pwv]
        t_atm_fiducial = atmosphere_eff[ufm_type][ufm_band][el_key][pwv_idx]
        pwv_adjust = t_atm_fiducial / t_atm_obs

        mars_diameter = mu.get_planet_diameter(int(obs_id), "Mars") # arcsec, we are using exact temperatures

        adjusted_amplitude = amp * pwv_adjust[0]

        cal_factor, cal_opt_efc = data_to_cal_factor(adjusted_amplitude, data_solid_angle, band, wafer, mars_diameter, tb_time)
        raw_factor, raw_opt_efc = data_to_cal_factor(amp, data_solid_angle, band, wafer, mars_diameter, tb_time)

        cal_dict[str(obs_id)] = {"adj_cal": cal_factor, "raw_cal": raw_factor, "pwv":pwv_obs, "el":el_obs, 
                                                            "omega_data": data_solid_angle, "fwhm":fitted_fwhm, "raw_opt":raw_opt_efc, "cal_opt":cal_opt_efc}
   
    with open("abscals.pk", "wb") as f:
        pk.dump(cal_dict, f)
                     
    result_dict = {}

    for key in cal_dict.keys():
        ufm = key.split("_")[4]
        freq = key.split("_")[5]
        if ufm in result_dict.keys():
            continue
        if "090" in freq or "150" in freq:
            result_dict[ufm] = {"090":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[]},
                                "150":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[]}
                               }
        else:
            result_dict[ufm] = {"220":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[]},
                                "280":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[]},
                               }
    for key in cal_dict.keys():
        ufm = key.split("_")[4]
        freq = key.split("_")[5][1:]
        result_dict[ufm][freq]["cal"].append(cal_dict[key]["adj_cal"])
        result_dict[ufm][freq]["raw_cal"].append(cal_dict[key]["raw_cal"])
        result_dict[ufm][freq]["el"].append(cal_dict[key]["el"])
        result_dict[ufm][freq]["pwv"].append(cal_dict[key]["pwv"])
        result_dict[ufm][freq]["obs"].append(key)
        result_dict[ufm][freq]["fwhm"].append(cal_dict[key]["fwhm"])
        result_dict[ufm][freq]["raw_opt"].append(cal_dict[key]["raw_opt"])
        result_dict[ufm][freq]["cal_opt"].append(cal_dict[key]["cal_opt"])
        result_dict[ufm][freq]["omega_data"].append(cal_dict[key]["omega_data"])
        
    today = dt.date.today()
    date_str = str(today.month).zfill(2)+str(today.day).zfill(2)+str(today.year)
    with open("results_{}.pk".format(date_str), "wb") as f:
        pk.dump(result_dict, f)


    
