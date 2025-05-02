from pixell import utils, enmap, bunch, reproject, colors, coordinates

import numpy as np

import matplotlib.pyplot as plt

import map_utils as u
from optical_loading import pwv_interp, keys_from_wafer

from sotodlib import core

from astropy.convolution import convolve, Gaussian2DKernel

from glob import glob
import datetime as dt
import dill as pk

def data_to_cal_factor(p_meas, beam_solid_angle, bandpass, mars_diameter, obs_id): #TODO: This should go elsewhere
    fiducial_solid_angle = u.angular_diameter_to_solid_angle(mars_diameter)

    timestamp = str(obs_id)
    planet_temp = T_b[timestamp]
    t_eff_planet = planet_temp[bandpass] * (fiducial_solid_angle / (beam_solid_angle * 1e-6))
    cal_factor = t_eff_planet / p_meas # K -> pW

    return cal_factor

T_b = {"1740190512":{"090":214.762, "150":220.089, "220":223.809, "280":225.923},
       "1740190513":{"090":214.762, "150":220.089, "220":223.809, "280":225.923},
       "1744251564":{"090":190.643, "150":192.441, "220":194.069, "280":195.167},
       "1742252709":{"090":186.460, "150":188.254, "220":190.103, "280":191.479},
       "1744255974":{"090":190.382, "150":192.237, "220":194.017, "280":195.256},
       "1744338027":{"090":190.750, "150":192.540, "220":194.132, "280":195.187},
       "1744839257":{"090":185.739, "150":187.463, "220":189.514, "280":191.091}, 
       "1744857536":{"090":189.952, "150":191.927, "220":193.868, "280":195.187}, 
       "1744921072":{"090":185.672, "150":187.267, "220":189.115, "280":190.549}, 
       "1744921073":{"090":185.672, "150":187.267, "220":189.115, "280":190.549}, 
       "1744921074":{"090":185.672, "150":187.267, "220":189.115, "280":190.549},
       "1744923240":{"090":185.522, "150":187.153, "220":189.072, "280":190.562},
       "1744923243":{"090":185.522, "150":187.153, "220":189.072, "280":190.562},
       "1744925494":{"090":185.544, "150":187.217, "220":189.210, "280":190.753},
       "1744925495":{"090":185.544, "150":187.217, "220":189.210, "280":190.753},
       "1744925496":{"090":185.544, "150":187.217, "220":189.210, "280":190.753},
       "1744925498":{"090":185.544, "150":187.217, "220":189.210, "280":190.753},
       "1744932818":{"090":212.559, "150":216.820, "220":220.061, "280":222.010},
       "1744932819":{"090":212.559, "150":216.820, "220":220.061, "280":222.010},
       "1744932820":{"090":212.559, "150":216.820, "220":220.061, "280":222.010},
       "1745009697":{"090":185.689, "150":187.286, "220":189.144, "280":190.587},
       "1745009698":{"090":185.689, "150":187.286, "220":189.144, "280":190.587},
       "1745011939":{"090":185.574, "150":187.204, "220":189.126, "280":190.618},
       "1745011940":{"090":185.574, "150":187.204, "220":189.126, "280":190.618},
       "1745011941":{"090":185.574, "150":187.204, "220":189.126, "280":190.618},
       "1745018947":{"090":186.230, "150":187.994, "220":190.082, "280":191.673},
       "1745027702":{"090":188.398, "150":190.354, "220":192.481, "280":194.009}, 
       "1745030065":{"090":188.859, "150":190.839, "220":192.940, "280":194.428},
       "1745032402":{"090":189.551, "150":191.539, "220":193.567, "280":194.971}, 
       "1745032403":{"090":189.551, "150":191.539, "220":193.567, "280":194.971},
       "1745032405":{"090":189.551, "150":191.539, "220":193.567, "280":194.971}, 
       "1745089225":{"090":187.951, "150":189.376, "220":190.817, "280":191.898},
       "1745091197":{"090":187.398, "150":188.847, "220":190.363, "280":191.515},
       "1745091198":{"090":187.398, "150":188.847, "220":190.363, "280":191.515},
       "1745095052":{"090":186.417, "150":187.928, "220":189.606, "280":190.902}, 
       "1745095053":{"090":186.417, "150":187.928, "220":189.606, "280":190.902},
       "1745095054":{"090":186.417, "150":187.928, "220":189.606, "280":190.902}, 
       "1745098069":{"090":185.783, "150":187.365, "220":189.196, "280":190.619},
       "1745098070":{"090":185.783, "150":187.365, "220":189.196, "280":190.619},
       "1745098071":{"090":185.783, "150":187.365, "220":189.196, "280":190.619},
       "1745105172":{"090":185.904, "150":187.616, "220":189.659, "280":191.230},
       "1745105173":{"090":185.904, "150":187.616, "220":189.659, "280":191.230},
       "1745105177":{"090":185.904, "150":187.616, "220":189.659, "280":191.230},
       "1745111685":{"090":187.141, "150":188.987, "220":191.115, "280":192.703},
       "1745111687":{"090":187.141, "150":188.987, "220":191.115, "280":192.703},
       "1745111689":{"090":187.141, "150":188.987, "220":191.115, "280":192.703},
       "1745115361":{"090":188.068, "150":189.995, "220":192.128, "280":193.680}, 
       "1745182059":{"090":186.843, "150":188.324, "220":189.932, "280":191.169},
       "1745182061":{"090":186.843, "150":188.324, "220":189.932, "280":191.169},
       "1745184306":{"090":186.210, "150":187.740, "220":189.468, "280":190.807},
       "1745184309":{"090":186.210, "150":187.740, "220":189.468, "280":190.807},
       "1745191466":{"090":185.714, "150":187.376, "220":189.361, "280":190.898},
       #{"090":, "150":, "220":, "280":}, 
      }


with open("atmosphere_eff.pk", "rb") as f:
    atmosphere_eff = pk.load(f)
    
fiducial_elevation = 50
fiducial_pwv = 1 # mm
el_key = "50" #hardcoded :(
pwv = pwv_interp()

path = "/so/home/saianeesh/data/beams/lat/source_maps/mars/"
paths = glob(path + "*/*/*_solved.fits")

beam_dict = {}
cal_dict = {}

with open("beams.pk", "rb") as f:
    beam_dict = pk.load(f)

for i in range(len(paths)):
    
    solved_path = paths[i]
    obs = solved_path.split("/")[-2]
    obs_id = obs.split("_")[1]
    wafer = solved_path.split("/")[-1].split("_")[-3]
    band = solved_path.split("/")[-1].split("_")[-2][1:]
    
    if obs_id not in T_b.keys():
        print("No Mars data for obs {}".format(obs_id))
        continue
    
    ufm_type, ufm_band = keys_from_wafer(wafer, band)
    """    
    solved = enmap.read_map(solved_path)[0]
    weights = enmap.read_map(solved_path.replace("solved", "weights"))[0][0]
    binned = enmap.read_map(solved_path.replace("solved", "binned"))[0]
    kernel = Gaussian2DKernel(5)
    smoothed = convolve(solved, kernel)
    cent = np.argmax(smoothed)
    cent = np.unravel_index(cent, solved.shape)
    pixsize = np.abs(solved.wcs.wcs.cdelt[0]*60)

    r = 20
    solved = solved[cent[0]-r:cent[0]+r,cent[1]-r:cent[1]+r]
    weights = weights[cent[0]-r:cent[0]+r,cent[1]-r:cent[1]+r]
    binned = binned[cent[0]-r:cent[0]+r,cent[1]-r:cent[1]+r]

    pixmap = enmap.pixmap(solved.shape, solved.wcs)
    fitted_amp, shift_x, shift_y, fitted_fwhm, data_solid_angle, chisred, popt, pcov = u.fit_gauss_pointing(solved, weights, pixmap, make_plots=False)
    
    beam_dict[str(solved_path.split("/")[-1][4:-12])] = {"amp":fitted_amp, "omega":data_solid_angle, "chisred":chisred, "popt":popt, "pcov":pcov}
    """
    cur_dict = beam_dict[str(solved_path.split("/")[-1][4:-12])]
    fitted_amp = cur_dict["amp"]
    data_solid_angle = cur_dict["omega"]
    chisred = cur_dict["chisred"]
    
    #Get pwv/el adjustment
    start_timestamp = int(obs_id)
    end_timestamp = int(obs_id)

    start_date = dt.datetime.utcfromtimestamp(start_timestamp) - dt.timedelta(days=1)
    end_date = dt.datetime.utcfromtimestamp(end_timestamp) + dt.timedelta(days=1)

    pwv_idx = np.where(np.array([np.abs(pwv - fiducial_pwv) < 0.1 for pwv in atmosphere_eff['pwv']]))[0]
    
    pwv_obs = pwv(obs_id)

    ctx = core.Context('/so/metadata/lat/contexts/smurf_detsets.yaml')
    meta = ctx.get_meta(paths[0].split("/")[-2])

    el_obs = meta.obs_info.el_center
    
    obs_idx_pwv = np.where(np.isclose(np.abs(atmosphere_eff['pwv'] - pwv_obs), np.min(np.abs(atmosphere_eff['pwv'] - pwv_obs))))[0][0]
    obs_key_el = [el for el in atmosphere_eff['LF']["LF_1"].keys() if np.abs(int(el) - el_obs) < 2.5][0]
    t_atm_obs = atmosphere_eff[ufm_type][ufm_band][obs_key_el][obs_idx_pwv]
    t_atm_fiducial = atmosphere_eff[ufm_type][ufm_band][el_key][pwv_idx]
    pwv_adjust = t_atm_fiducial / t_atm_obs
    
    mars_diameter = u.get_planet_diameter(int(obs_id), "Mars") # arcsec, we are using exact temperatures
    
    adjusted_amplitude = fitted_amp * pwv_adjust[0]

    cal_factor = data_to_cal_factor(adjusted_amplitude, data_solid_angle, band, mars_diameter, obs_id)
    raw_factor = data_to_cal_factor(fitted_amp, data_solid_angle, band, mars_diameter, obs_id)
    
    cal_dict[str(solved_path.split("/")[-1][4:-12])] = {"adj_cal": cal_factor, "raw_cal": raw_factor, "chi_beam":chisred, "pwv":pwv_obs, "el":el_obs}
    
    
with open("beams.pk", "wb") as f:
    pk.dump(beam_dict, f)
    
with open("abscals.pk", "wb") as f:
    pk.dump(cal_dict, f)
    

    
