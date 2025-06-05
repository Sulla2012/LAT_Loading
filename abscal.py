from pixell import utils, enmap, bunch, reproject, colors, coordinates

import numpy as np

from scipy.integrate import quad

import matplotlib.pyplot as plt

import map_utils as mu
from optical_loading import pwv_interp, keys_from_wafer, bandpass_interp

from sotodlib import core

from astropy.modeling import models
from astropy import units as u
from astropy import constants as consts
from astropy.convolution import convolve, Gaussian2DKernel

from glob import glob
import datetime as dt
import dill as pk

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

T_b = {"1740190512":{"090":214.762, "150":220.089, "220":223.809, "280":225.923},
       "1740190513":{"090":214.762, "150":220.089, "220":223.809, "280":225.923},
       "1744251564":{"090":190.643, "150":192.441, "220":194.069, "280":195.167},
       "1742252709":{"090":186.460, "150":188.254, "220":190.103, "280":191.479},
       "1744255974":{"090":190.382, "150":192.237, "220":194.017, "280":195.256},
       "1744338027":{"090":190.750, "150":192.540, "220":194.132, "280":195.187},
       "1744321883":{"090":190.658, "150":192.496, "220":194.116, "280":195.170},
       "1744321886":{"090":190.658, "150":192.496, "220":194.116, "280":195.170},
       "1744335555":{"090":190.749, "150":192.560, "220":194.155, "280":195.196},
       "1744839257":{"090":185.739, "150":187.463, "220":189.514, "280":191.091},
       "1744857535":{"090":189.952, "150":191.927, "220":193.868, "280":195.187},
       "1744857536":{"090":189.952, "150":191.927, "220":193.868, "280":195.187},
       "1744921072":{"090":185.672, "150":187.267, "220":189.115, "280":190.549},
       "1744921073":{"090":185.672, "150":187.267, "220":189.115, "280":190.549}, 
       "1744921074":{"090":185.672, "150":187.267, "220":189.115, "280":190.549},
       "1744923236":{"090":185.522, "150":187.153, "220":189.072, "280":190.562},
       "1744923238":{"090":185.522, "150":187.153, "220":189.072, "280":190.562},
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
       "1745018948":{"090":186.230, "150":187.994, "220":190.082, "280":191.673},
       "1745018949":{"090":186.230, "150":187.994, "220":190.082, "280":191.673},
       "1745027699":{"090":188.398, "150":190.354, "220":192.481, "280":194.009}, 
       "1745027702":{"090":188.398, "150":190.354, "220":192.481, "280":194.009},
       "1745030065":{"090":188.859, "150":190.839, "220":192.940, "280":194.428},
       "1745032402":{"090":189.551, "150":191.539, "220":193.567, "280":194.971}, 
       "1745032403":{"090":189.551, "150":191.539, "220":193.567, "280":194.971},
       "1745032405":{"090":189.551, "150":191.539, "220":193.567, "280":194.971},
       "1745032409":{"090":189.551, "150":191.539, "220":193.567, "280":194.971},
       "1745089225":{"090":187.951, "150":189.376, "220":190.817, "280":191.898},
       "1745091197":{"090":187.398, "150":188.847, "220":190.363, "280":191.515},
       "1745091198":{"090":187.398, "150":188.847, "220":190.363, "280":191.515},
       "1745095052":{"090":186.417, "150":187.928, "220":189.606, "280":190.902}, 
       "1745095053":{"090":186.417, "150":187.928, "220":189.606, "280":190.902},
       "1745095054":{"090":186.417, "150":187.928, "220":189.606, "280":190.902}, 
       "1745098069":{"090":185.783, "150":187.365, "220":189.196, "280":190.619},
       "1745098070":{"090":185.783, "150":187.365, "220":189.196, "280":190.619},
       "1745098071":{"090":185.783, "150":187.365, "220":189.196, "280":190.619},
       "1745098073":{"090":185.783, "150":187.365, "220":189.196, "280":190.619},
       "1745105172":{"090":185.904, "150":187.616, "220":189.659, "280":191.230},
       "1745105173":{"090":185.904, "150":187.616, "220":189.659, "280":191.230},
       "1745105177":{"090":185.904, "150":187.616, "220":189.659, "280":191.230},
       "1745111685":{"090":187.141, "150":188.987, "220":191.115, "280":192.703},
       "1745111687":{"090":187.141, "150":188.987, "220":191.115, "280":192.703},
       "1745111689":{"090":187.141, "150":188.987, "220":191.115, "280":192.703},
       "1745115361":{"090":188.068, "150":189.995, "220":192.128, "280":193.680},
       "1745115364":{"090":188.068, "150":189.995, "220":192.128, "280":193.680},
       "1745179289":{"090":187.625, "150":189.062, "220":190.549, "280":191.676},
       "1745179292":{"090":187.625, "150":189.062, "220":190.549, "280":191.676},
       "1745182059":{"090":186.843, "150":188.324, "220":189.932, "280":191.169},
       "1745182060":{"090":186.843, "150":188.324, "220":189.932, "280":191.169},
       "1745182061":{"090":186.843, "150":188.324, "220":189.932, "280":191.169},
       "1745184306":{"090":186.210, "150":187.740, "220":189.468, "280":190.807},
       "1745184307":{"090":186.210, "150":187.740, "220":189.468, "280":190.807},
       "1745184308":{"090":186.210, "150":187.740, "220":189.468, "280":190.807},
       "1745184309":{"090":186.210, "150":187.740, "220":189.468, "280":190.807},
       "1745191464":{"090":185.714, "150":187.376, "220":189.361, "280":190.898},
       "1745191465":{"090":185.714, "150":187.376, "220":189.361, "280":190.898},
       "1745191466":{"090":185.714, "150":187.376, "220":189.361, "280":190.898},
       "1745202116":{"090":187.621, "150":189.502, "220":191.636, "280":193.211},
       "1745543221":{"090":186.216, "150":187.877, "220":189.841, "280":191.362},
       "1745543222":{"090":186.216, "150":187.877, "220":189.841, "280":191.362},
       "1745545818":{"090":186.216, "150":187.877, "220":189.841, "280":191.362},
       "1745547448":{"090":186.310, "150":187.997, "220":189.998, "280":191.544},
       "1745549920":{"090":186.670, "150":188.410, "220":190.469, "280":192.047}, 
       "1745630131":{"090":186.484, "150":188.055, "220":189.853, "280":191.249},
       "1745630134":{"090":186.484, "150":188.055, "220":189.853, "280":191.249},
       "1745632671":{"090":186.351, "150":187.979, "220":189.884, "280":191.362},
       "1745716145":{"090":187.194, "150":188.713, "220":190.388, "280":191.680},
       "1745716147":{"090":187.194, "150":188.713, "220":190.388, "280":191.680},
       "1745718420":{"090":186.716, "150":188.278, "220":190.054, "280":191.430},
       "1745720093":{"090":186.551, "150":188.147, "220":189.996, "280":191.431},
       "1745720094":{"090":186.551, "150":188.147, "220":189.996, "280":191.431},
       "1745720096":{"090":186.551, "150":188.147, "220":189.996, "280":191.431},
       "1745720097":{"090":186.551, "150":188.147, "220":189.996, "280":191.431},
       "1745970733":{"090":190.832, "150":192.289, "220":193.582, "280":194.485},
       "1745970734":{"090":190.832, "150":192.289, "220":193.582, "280":194.485},
       "1745973900":{"090":189.725, "150":191.142, "220":192.502, "280":193.502},
       "1745973902":{"090":189.725, "150":191.142, "220":192.502, "280":193.502},
       "1745975396":{"090":189.725, "150":191.142, "220":192.502, "280":193.502},
       "1745977340":{"090":188.892, "150":190.327, "220":191.784, "280":192.882},
       "1745982286":{"090":187.621, "150":189.130, "220":190.788, "280":192.066},
       "1745982287":{"090":187.621, "150":189.130, "220":190.788, "280":192.066},
       "1746057423":{"090":191.364, "150":192.870, "220":194.170, "280":195.052},
       "1746057425":{"090":191.364, "150":192.870, "220":194.170, "280":195.052},
       "1746059736":{"090":190.995, "150":192.456, "220":193.747, "280":194.646},
       "1746059737":{"090":190.995, "150":192.456, "220":193.747, "280":194.646},
       "1746059739":{"090":190.995, "150":192.456, "220":193.747, "280":194.646}, 
       "1746061770":{"090":190.533, "150":191.964, "220":193.271, "280":194.204},
       "1746063652":{"090":190.196, "150":191.617, "220":192.946, "280":193.910},
       "1746068585":{"090":188.369, "150":189.836, "220":191.386, "280":192.569},
       "1746142610":{"090":191.891, "150":193.509, "220":194.888, "280":195.793},
       "1746142611":{"090":191.891, "150":193.509, "220":194.888, "280":195.793},
       "1746142612":{"090":191.891, "150":193.509, "220":194.888, "280":195.793},
       "1746148087":{"090":191.220, "150":192.690, "220":193.979, "280":194.869},
       "1746148089":{"090":191.220, "150":192.690, "220":193.979, "280":194.869},
       "1746149962":{"090":190.538, "150":191.962, "220":193.271, "280":194.210},
       "1746154880":{"090":189.174, "150":190.600, "220":192.043, "280":193.128},
       "1746228917":{"090":192.035, "150":193.782, "220":195.310, "280":196.308},
       "1746232238":{"090":191.952, "150":193.556, "220":194.921, "280":915.821},
       "1746315232":{"090":192.006, "150":193.837, "220":195.495, "280":196.588},
       "1746315233":{"090":192.006, "150":193.837, "220":195.495, "280":196.588},
       "1746318605":{"090":192.132, "150":193.828, "220":195.292, "280":196.247},
       "1746320747":{"090":192.078, "150":193.703, "220":195.087, "280":195.996},
       "1746322542":{"090":191.903, "150":193.459, "220":194.785, "280":195.669},
       "1746402995":{"090":192.056, "150":193.801, "220":195.592, "280":196.710},
       "1746489159":{"090":191.938, "150":193.876, "220":195.765, "280":197.055},
       "1746489161":{"090":191.938, "150":193.876, "220":195.765, "280":197.055},
       "1746491274":{"090":192.103, "150":193.967, "220":195.685, "280":196.826},
       "1746493391":{"090":192.197, "150":194.003, "220":195.621, "280":196.685},
       "1746499222":{"090":192.107, "150":193.670, "220":195.001, "280":195.885},
       "1746499223":{"090":192.107, "150":193.670, "220":195.001, "280":195.885},
       "1746562890":{"090":192.630, "150":194.390, "220":196.067, "280":197.233},
       "1746575324":{"090":192.010, "150":193.958, "220":195.883, "280":197.209},
       "1746579277":{"090":192.153, "150":194.031, "220":195.775, "280":196.943},
       "1746834604":{"090":192.539, "150":194.427, "220":196.326, "280":197.668},
       "1746908366":{"090":192.972, "150":194.715, "220":196.253, "280":197.258},
       "1746908369":{"090":192.972, "150":194.715, "220":196.253, "280":197.258},
       "1746921207":{"090":193.024, "150":194.748, "220":196.268, "280":197.269},
       "1746993874":{"090":192.923, "150":194.713, "220":196.314, "280":197.360},
       "1746993876":{"090":192.923, "150":194.713, "220":196.314, "280":197.360},
       #{"090":, "150":, "220":, "280":}, 
      }

uhf_raws = ["/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17449/obs_1744925496_latc1_111/obs_1744925496_latc1_111_ufm_uv46_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17449/obs_1744925498_lati5_111/obs_1744925498_lati5_111_ufm_uv31_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17449/obs_1744932819_latc1_111/obs_1744932819_latc1_111_ufm_uv38_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17449/obs_1744932819_latc1_111/obs_1744932819_latc1_111_ufm_uv39_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17449/obs_1744932819_latc1_111/obs_1744932819_latc1_111_ufm_uv46_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17449/obs_1744932819_latc1_111/obs_1744932819_latc1_111_ufm_uv46_f280_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17448/obs_1744839257_lati5_111/obs_1744839257_lati5_111_ufm_uv31_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17448/obs_1744839257_lati5_111/obs_1744839257_lati5_111_ufm_uv31_f280_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17448/obs_1744839257_lati5_111/obs_1744839257_lati5_111_ufm_uv42_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17448/obs_1744839257_lati5_111/obs_1744839257_lati5_111_ufm_uv42_f280_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17448/obs_1744839257_lati5_111/obs_1744839257_lati5_111_ufm_uv47_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17448/obs_1744839257_lati5_111/obs_1744839257_lati5_111_ufm_uv47_f280_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744321883_latc1_111/obs_1744321883_latc1_111_ufm_uv38_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744321883_latc1_111/obs_1744321883_latc1_111_ufm_uv38_f280_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744321883_latc1_111/obs_1744321883_latc1_111_ufm_uv39_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744321883_latc1_111/obs_1744321883_latc1_111_ufm_uv46_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744321883_latc1_111/obs_1744321883_latc1_111_ufm_uv46_f280_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744321886_lati5_101/obs_1744321886_lati5_101_ufm_uv47_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744321886_lati5_101/obs_1744321886_lati5_101_ufm_uv47_f280_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744335555_latc1_111/obs_1744335555_latc1_111_ufm_uv38_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744335555_latc1_111/obs_1744335555_latc1_111_ufm_uv39_f220_solved.fits",
            "/so/home/saianeesh/data/beams/lat/source_maps_raw/mars/17443/obs_1744335555_latc1_111/obs_1744335555_latc1_111_ufm_uv46_f220_solved.fits",
           ]        

fwhm_cuts = {"090": [1.8, 2.3],
             "150": [1.3, 1.6],
             "220": [0.7, 1.1],
             "280": [0.7, 1.0],
            }

bandpasses = {"090": 28.83,
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

    #path_mf = "/so/home/saianeesh/data/beams/lat/source_maps/mars/"
    path_mf = "/so/home/saianeesh/data/beams/lat/source_maps_per_obs/mars/"
    paths_mf = glob(path_mf + "*/*/*_solved.fits")
    paths_mf = [path for path in paths_mf if "090" in path or "150" in path]

    path_uhf = "/so/home/saianeesh/data/beams/lat/source_maps_per_obs/mars/"
    paths_uhf = glob(path_uhf + "*/*/*_solved.fits")
    paths_uhf = [path for path in paths_uhf if "220" in path or "280" in path]
    #paths_uhf += uhf_raws

    paths = paths_mf+paths_uhf

    beam_dict = {}

    cal_dict = {}

    #with open("beams.pk", "rb") as f:
    #    beam_dict = pk.load(f)

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
        try:
            fitted_amp, shift_x, shift_y, fitted_fwhm, data_solid_angle, chisred, popt, pcov = mu.fit_gauss_pointing(solved, weights, pixmap, make_plots=False)
        except:
            continue

        if fwhm_cuts[band][1] < fitted_fwhm or fitted_fwhm < fwhm_cuts[band][0]:
            continue
        if chisred < 25 or chisred > 1000:
            continue
        beam_dict[str(solved_path.split("/")[-1][4:-12])] = {"amp":fitted_amp, "omega":data_solid_angle, "fitted_fwhm":fitted_fwhm, "chisred":chisred, "popt":popt, "pcov":pcov}
        """        
        cur_dict = beam_dict[str(solved_path.split("/")[-1][4:-12])]
        fitted_amp = cur_dict["amp"]
        fitted_fwhm = cur_dict["fitted_fwhm"]
        data_solid_angle = cur_dict["omega"]
        chisred = cur_dict["chisred"]
        """     
        if fwhm_cuts[band][1] < fitted_fwhm or fitted_fwhm < fwhm_cuts[band][0]:
            continue
        if chisred < 25 or chisred > 1000:
            continue
        

        #Get pwv/el adjustment
        start_timestamp = int(obs_id)
        end_timestamp = int(obs_id)

        start_date = dt.datetime.utcfromtimestamp(start_timestamp) - dt.timedelta(days=1)
        end_date = dt.datetime.utcfromtimestamp(end_timestamp) + dt.timedelta(days=1)

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

        adjusted_amplitude = fitted_amp * pwv_adjust[0]

        cal_factor, cal_opt_efc = data_to_cal_factor(adjusted_amplitude, data_solid_angle, band, wafer, mars_diameter, obs_id=obs_id)
        raw_factor, raw_opt_efc = data_to_cal_factor(fitted_amp, data_solid_angle, band, wafer, mars_diameter, obs_id=obs_id)

        cal_dict[str(solved_path.split("/")[-1][4:-12])] = {"adj_cal": cal_factor, "raw_cal": raw_factor, "chi_beam":chisred, "pwv":pwv_obs, "el":el_obs, 
                                                            "omega_data": data_solid_angle, "fwhm":fitted_fwhm, "raw_opt":raw_opt_efc, "cal_opt":cal_opt_efc}
   

    with open("beams.pk", "wb") as f:
        pk.dump(beam_dict, f)

    with open("abscals.pk", "wb") as f:
        pk.dump(cal_dict, f)


    
