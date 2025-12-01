from pixell import utils, enmap, bunch, reproject, colors, coordinates

import numpy as np
import pandas as pd

import map_utils as mu
from optical_loading import pwv_interp, keys_from_wafer, bandpass_interp, get_bandwidth
from mars_temps import T_b

from sotodlib import core
import sotodlib.io.metadata as io_meta

from astropy import units as u
from astropy import constants as consts
from astropy.convolution import convolve, Gaussian2DKernel

import datetime as dt
import dill as pk
import os
import h5py
import yaml
from glob import glob
import copy

def data_to_cal_factor(p_meas, beam_solid_angle, planet_diameter, bandwidth, planet_temp):
    fiducial_solid_angle = mu.angular_diameter_to_solid_angle(planet_diameter)
    
    fill_factor = (fiducial_solid_angle / (beam_solid_angle))
    t_eff_planet = planet_temp * fill_factor
    cal_factor = t_eff_planet / p_meas # K -> pW
        
    opt_eff = ((1/(cal_factor)*u.pW/u.K)/(consts.k_B * bandwidth * u.GHz)).to(1)
    
    return cal_factor, opt_eff


fwhm_cuts = {"090": [1.7, 2.3],
             "150": [1.0, 1.6],
             "220": [0.7, 1.2],
             "280": [0.65, 1.0],
            }

#These are roughly 20% - 500% the expected beam value
beam_volume_cuts = {"090": [8e-8, 2e-6],
                    "150": [4e-8, 1e-6],
                    "220": [1e-8, 2e-7],
                    "280": [1e-8, 2e-7],
                }

if __name__ == '__main__':
    with open("atmosphere_eff.pk", "rb") as f:
        atmosphere_eff = pk.load(f)

    fiducial_elevation = 50
    fiducial_pwv = 1 # mm
    el_key = "50" #hardcoded :(
    pwv = pwv_interp()

    #load matt hasslefield type beams. Unfortunately these are in two locations,
    #this somewhat awkward code block correctly loads all beams from both
    #dirs but doesn't load beams

    #fpath="/so/home/saianeesh/data/beams/lat/source_maps/per_obs/fits/beam_pars.h5"
    fpath = "/so/home/saianeesh/data/beams/lat_old/source_maps/pointing_model/fits/beam_pars.h5"
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

    #make bandwidth dict for speed
    #Note we use the site measured bandwidth for an array if available
    #Else we use the average of all arrays at that freq (also site measured)
    bandwidths = {}
    for stream_id in set(stream_ids):
        ufm = stream_id.split("_")[1]
        if ufm not in bandwidths.keys():
            bandwidths[ufm] = {}
        if "mv" in ufm:
            cur_bands = ["090", "150"]
        elif "uv" in ufm:
            cur_bands = ["220", "280"]
        else:
            raise ValueError(f"Error: ufm {ufm} is not valid")
        for band in cur_bands:
            bandwidths[ufm][band] = get_bandwidth(band=band, ufm=ufm)

    ctx = core.Context('/so/metadata/lat/contexts/smurf_detsets.yaml')


    arrays = ["mv21", "mv24", "mv28",
              "mv13", "mv20", "mv34",
              "mv14", "mv32", "mv49",
              "mv11", "mv25", "mv26",
              "uv31", "uv42", "uv48",
              "uv38", "uv39", "uv46",
             ]



    radii_saturn = {array:{} for array in arrays}
    means_datas_saturn = {array:{} for array in arrays}
    means_fits_saturn = {array:{} for array in arrays}

    for array in arrays:
        if "mv" in array:
            radii_saturn[array] = {"f090":[], "f150":[]}
            means_datas_saturn[array] = {"f090":[], "f150":[]}
            means_fits_saturn[array] = {"f090":[], "f150":[]}
        elif "uv" in array:
            radii_saturn[array] = {"f220":[], "f280":[]}
            means_datas_saturn[array] = {"f220":[], "f280":[]}
            means_fits_saturn[array] = {"f220":[], "f280":[]}

            radii_mars = copy.deepcopy(radii_saturn)

    means_datas_mars = copy.deepcopy(means_datas_saturn)
    means_fits_mars = copy.deepcopy(means_fits_saturn)

    for i, aman in enumerate(amans):
        obs_id = obs_ids[i].split("_")[1]
        ufm = stream_ids[i].split("_")[1]
        band = bands[i][1:]     
        ufm_type, ufm_band = keys_from_wafer(ufm, band)
        
        fitted_fwhm = aman.data_fwhm.to(u.arcmin).value
        data_solid_angle = aman.data_solid_angle_corr.value
        
        if "amp_outer" in aman.keys():
            amp = aman.amp.value + aman.amp_outer.value
        else:
            amp = aman.amp.value
        
        if fwhm_cuts[band][1] < fitted_fwhm or fitted_fwhm < fwhm_cuts[band][0]:
            print(fwhm_cuts[band][0], fitted_fwhm, fwhm_cuts[band][1], band, ufm)
            continue     
            
        if beam_volume_cuts[band][1] < data_solid_angle or data_solid_angle < beam_volume_cuts[band][0]:
            print(beam_volume_cuts[band][0], data_solid_angle, beam_volume_cuts[band][1], band, ufm)
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
        if np.isnan(pwv_obs):
            continue
        if pwv_obs > 2.5:
            continue
            
        
        meta = ctx.get_meta(obs_ids[i])

        el_obs = meta.obs_info.el_center

        if el_obs > 90:
            el_obs = 180 - el_obs

        obs_idx_pwv = np.where(np.isclose(np.abs(atmosphere_eff['pwv'] - pwv_obs), np.min(np.abs(atmosphere_eff['pwv'] - pwv_obs))))[0][0]
        try:
            obs_key_el = [el for el in atmosphere_eff['LF']["LF_1"].keys() if np.abs(int(el) - el_obs) < 2.5][0]
        except:
            print("El {} for obs {} out of range".format(el_obs, obs_ids[i]))
            continue
            
        t_atm_obs = atmosphere_eff[ufm_type][ufm_band][obs_key_el][obs_idx_pwv]
        t_atm_fiducial = atmosphere_eff[ufm_type][ufm_band][el_key][pwv_idx]
        pwv_adjust = t_atm_fiducial / t_atm_obs

        adjusted_amplitude = amp * pwv_adjust[0]

        tags = ctx.obsdb.get(obs_ids[i], tags=True)["tags"]
        
        if "mars" in tags:
            planet="mars"
            tb_time = 0
            for key in T_b.keys():
                if np.abs(int(obs_id) - int(key)) <= 300: #Within 5 minutes is OK
                    tb_time = key
                    break
            if not tb_time:
                print("No Mars data for obs {}".format(obs_id))
                continue    
            planet_temp = T_b[tb_time][band]
            
        elif "saturn" in tags:
            planet="saturn"
            if band == "090":
                planet_temp = 142.9
            elif band == "150":
                planet_temp = 142.6
            elif band == "220":
                planet_temp = 139.7
            elif band == "280":
                planet_temp = 138.7
            else:
                print("Error: invalid freq {}".format(freq))
                continue
        else:
            print("Error: no planet in tags: {}".format(tags[-1]))
            continue

        planet_diameter = mu.get_planet_diameter(int(obs_id), planet.capitalize()) # arcsec, we are using exact temperatures

        cal_factor, cal_opt_efc = data_to_cal_factor(p_meas=adjusted_amplitude, beam_solid_angle=data_solid_angle,
                                                     planet_diameter=planet_diameter, bandwidth=bandwidths[ufm][band], planet_temp=planet_temp)
        raw_factor, raw_opt_efc = data_to_cal_factor(p_meas=amp, beam_solid_angle=data_solid_angle,
                                                     planet_diameter=planet_diameter, bandwidth=bandwidths[ufm][band], planet_temp=planet_temp)
        
        if raw_factor >= 40 and planet == "saturn":
            continue #Some of the saturn observations are accidentally of Neptune, leading to very high abscals (when using Saturn temp)
                     #Matt is working on a real fix but for now since the Neptune amp is >10x lower, a cut on the abscal is safe
        cal_dict[str(ufm)+"_"+str(band)+"_"+str(obs_id)] = {"adj_cal": cal_factor, "raw_cal": raw_factor, "pwv":pwv_obs, "el":el_obs, 
                                                            "omega_data": data_solid_angle, "fwhm":fitted_fwhm, "raw_opt":raw_opt_efc, 
                                                            "cal_opt":cal_opt_efc, "source":planet, "time":obs_id}
        
        #Make profiles
        solved_file = "/so/home/saianeesh/data/beams/lat/source_maps/per_obs/{}/".format(planet)+str(obs_id[:5])+"/"+str(obs_ids[i])+"/"+str(obs_ids[i])+"_"+str(stream_ids[i])+"_"+str(bands[i])+"_solved.fits"
        weight_file = solved_file.replace("solved", "weights")
        binned_file = solved_file.replace("solved", "binned")
        try:
            solved = enmap.read_map(solved_file)[0]
            weights = enmap.read_map(weight_file)[0][0]
            binned = enmap.read_map(binned_file)[0]

        except: continue
            
        kernel = Gaussian2DKernel(5)
        smoothed = convolve(solved, kernel)
        cent = np.argmax(smoothed)
        cent = np.unravel_index(cent, solved.shape)
        pixsize = np.abs(solved.wcs.wcs.cdelt[0]*60)

        r = 20
        solved = solved[cent[0]-r:cent[0]+r,cent[1]-r:cent[1]+r]
        weights = weights[cent[0]-r:cent[0]+r,cent[1]-r:cent[1]+r]
        binned = binned[cent[0]-r:cent[0]+r,cent[1]-r:cent[1]+r]
        #plt.imshow(solved)
        #plt.colorbar()
        pixmap = enmap.pixmap(solved.shape, solved.wcs)
        fitted_amp, shift_x, shift_y, fitted_fwhm, data_solid_angle, chisred, popt, pcov, radii_data, means_data, means_fit = mu.fit_gauss_pointing(solved, weights, pixmap, make_plots=True)
        
        if planet == "mars":
            radii_mars[ufm][bands[i]].append(radii_data)
            means_datas_mars[ufm][bands[i]].append(means_data)
            means_fits_mars[ufm][bands[i]].append(means_fit)
        elif planet == "saturn":
            radii_saturn[ufm][bands[i]].append(radii_data)
            means_datas_saturn[ufm][bands[i]].append(means_data)
            means_fits_saturn[ufm][bands[i]].append(means_fit)
        else:
            continue

    with open("abscals.pk", "wb") as f:
        pk.dump(cal_dict, f)
        
    rad_dict = {"rad_sat":radii_saturn, "data_sat":means_datas_saturn, "fit_sat":means_fits_saturn, "rad_mars":radii_mars, "data_mars":means_datas_mars, "fit_mars":means_fits_mars}
    with open("mars_saturn.pk", "wb") as f:
        pk.dump(rad_dict, f)
                     
    result_dict = {}

    for key in cal_dict.keys():
        ufm = key.split("_")[0]
        freq = key.split("_")[1]
        if ufm in result_dict.keys():
            continue
        if "090" in freq or "150" in freq:
            result_dict[ufm] = {"090":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[], "source":[], "time":[]},
                                "150":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[], "source":[], "time":[]}
                               }
        else:
            result_dict[ufm] = {"220":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[], "source":[], "time":[]},
                                "280":{"cal":[], "chi":[], "obs":[], "raw_cal":[], "el":[], "pwv":[], "fwhm":[], "raw_opt":[], "cal_opt":[], "omega_data":[], "source":[], "time":[]},
                               }
    for key in cal_dict.keys():
        ufm = key.split("_")[0]
        freq = key.split("_")[1]
        result_dict[ufm][freq]["cal"].append(cal_dict[key]["adj_cal"])
        result_dict[ufm][freq]["raw_cal"].append(cal_dict[key]["raw_cal"])
        result_dict[ufm][freq]["el"].append(cal_dict[key]["el"])
        result_dict[ufm][freq]["pwv"].append(cal_dict[key]["pwv"])
        result_dict[ufm][freq]["obs"].append(key)
        result_dict[ufm][freq]["fwhm"].append(cal_dict[key]["fwhm"])
        result_dict[ufm][freq]["raw_opt"].append(cal_dict[key]["raw_opt"])
        result_dict[ufm][freq]["cal_opt"].append(cal_dict[key]["cal_opt"])
        result_dict[ufm][freq]["omega_data"].append(cal_dict[key]["omega_data"])
        result_dict[ufm][freq]["source"].append(cal_dict[key]["source"])
        result_dict[ufm][freq]["time"].append(cal_dict[key]["time"])
        
    today = dt.date.today()
    date_str = str(today.month).zfill(2)+str(today.day).zfill(2)+str(today.year)
    
    with open("results_{}.pk".format(date_str), "wb") as f:
        pk.dump(result_dict, f)
    
    #Now to write the manifest db
    
    #Load important times in LAT history i.e. slipage/alighnment
    lat_times = {"alignment0": {"start": 1744848000, "stop":1745150000}, "cr_slip0": {"start": 1745150000, "stop":1749355200}, "alignment1": {"start": 1749600000, "stop":1755576000}, "alignment2": {"start": 1756699200, "stop":20000000000}, }
    cals = []
    raw_cals = []
    data_freqs = []
    data_ufms = []
    cals_cmb = []
    raw_cals_cmb = []
    data_solid_angles = []
    omegas = []
    obs = []

    pwv = pwv_interp()

    freqs = ["090", "150", "220", "280"]
    ufms = sorted(result_dict.keys())

    flavor_dict = {"090": "MF_1",
                   "150": "MF_2",
                   "220": "UHF_1",
                   "280": "UHF_2"
                  }
    
    for freq in freqs:
        temp_conv = mu.temp_conv(T_B=2.725*u.Kelvin, flavor=flavor_dict[freq].split("_")[0], ch=flavor_dict[freq], kind='baseline') #Temperature for rj->cmb
        for ufm in ufms:
            for key in result_dict.keys():
                if ufm not in key:
                    continue
                for sub_key in result_dict[key].keys():
                    if freq not in sub_key:
                        continue
                    cur_cals = np.array(result_dict[key][sub_key]["cal"])
                    cur_raw_cals = np.array(result_dict[key][sub_key]["raw_cal"])
                    omega_data = np.array(result_dict[key][sub_key]["omega_data"])
                    cur_obs = np.array(result_dict[key][sub_key]["obs"])
                    for j in range(len(cur_cals)):
                        cals.append(cur_cals[j])
                        raw_cals.append(cur_raw_cals[j])
                        cals_cmb.append(cur_cals[j]*temp_conv)
                        raw_cals_cmb.append(cur_cals[j]*temp_conv)
                        data_freqs.append(freq)
                        data_ufms.append(ufm)
                        omegas.append(omega_data[j])
                        obs.append(cur_obs[j][9:])

    data_freqs = np.array(data_freqs)
    data_ufms = np.array(data_ufms)
    cals = np.array(cals)
    raw_cals = np.array(raw_cals)
    obs = np.array(obs, dtype=float)

    df = pd.DataFrame({'freqs': data_freqs, 'ufms':data_ufms, 'cals': cals,'raw_cals': raw_cals,'cals_cmb':cals_cmb, 'raw_cals_cmb':raw_cals_cmb, 'omegas':omegas, "obs":obs})
    
    #periods we care about
    keys = ["alignment0", "cr_slip0", "alignment1", "alignment2"]

    for key in keys:
        data = []

        #For each period, we're going to compute the average abscal for each ufm and freq
        mfs = ["090", "150"]
        ufs = ["220", "280"]
        for ufm in ufms:
            for freq in freqs:
                if freq in mfs and "uv" in ufm: continue
                if freq in ufs and "mv" in ufm: continue
                if len(np.where((df.freqs == str(freq)) & (df.ufms == str(ufm)))[0]) == 0:
                    print(freq, ufm) #Let me know if there are no obs with this array/freq

                if len(np.where((df.freqs == str(freq)) & (df.ufms == str(ufm)) & (df.obs >= lat_times[key]["start"]) & (df.obs <= lat_times[key]["stop"]))[0]) == 0:
                    #If there are no obs in this particular time range, just use the all time average for that array
                    cur_df = df.where((df.freqs == str(freq)) & (df.ufms == str(ufm)))
                else:   
                    cur_df = df.where((df.freqs == str(freq)) & (df.ufms == str(ufm)) & (df.obs >= lat_times[key]["start"]) & (df.obs <= lat_times[key]["stop"]))
                data.append(("ufm_"+str(ufm), "f"+str(freq), 
                             float(np.nanmean(cur_df.cals)),
                             float(np.nanmean(cur_df.raw_cals)),
                             float(np.nanmean(cur_df.cals_cmb)), 
                             float(np.nanmean(cur_df.raw_cals_cmb)),
                             float(np.nanmean(cur_df.omegas)),))

            data.append(("ufm_"+str(ufm), "NC", np.nan, np.nan, np.nan, np.nan, np.nan))

        # Write to HDF5
        rs = core.metadata.ResultSet(
            keys=['dets:stream_id', 'dets:wafer.bandpass', 'abscal_rj', 'raw_abscal_rj', 'abscal_cmb', 'raw_abscal_cmb', 'beam_solid_angle'])
        rs.rows = data
        io_meta.write_dataset(rs, 'abscals.h5', "abscal_{}".format(key), overwrite=True)
        
    # Record in ManifestDb.
    scheme = core.metadata.ManifestScheme()
    scheme.add_range_match("obs:timestamp")
    scheme.add_data_field('dataset')

    db = core.metadata.ManifestDb(scheme=scheme)
    db.add_entry({"obs:timestamp": (lat_times["alignment0"]["start"], lat_times["alignment0"]["stop"]),
                  "dataset": "abscal_initial_alignment"},
                  filename="abscals.h5")
    db.add_entry({"obs:timestamp": (lat_times["cr_slip0"]["start"], lat_times["cr_slip0"]["stop"]),
                  "dataset": "abscal_corot_slip"},
                  filename="abscals.h5")
    db.add_entry({"obs:timestamp": (lat_times["alignment1"]["start"], alignment1["alignment1"]["stop"]),
                  "dataset": "abscal_first_realignment"},
                  filename="abscals.h5")
    db.add_entry({"obs:timestamp": (lat_times["alignment2"]["start"], alignment1["alignment2"]["stop"]),
                  "dataset": "abscal_second_realignment"},
                  filename="abscals.h5")

    #db.add_entry({'dataset': 'abscal'}, filename='abscals.h5')
    db.to_file('db.sqlite')