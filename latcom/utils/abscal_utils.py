from . import map_utils as mu
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from pixell import enmap

def data_to_cal_factor(
    p_meas, beam_solid_angle, planet_diameter, bandwidth, planet_temp
):
    fiducial_solid_angle = mu.angular_diameter_to_solid_angle(planet_diameter)

    fill_factor = fiducial_solid_angle / (beam_solid_angle)
    t_eff_planet = planet_temp * fill_factor
    cal_factor = t_eff_planet / p_meas  # K -> pW

    opt_eff = ((1 / (cal_factor) * u.pW / u.K) / (consts.k_B * bandwidth * u.GHz)).to(1)

    return cal_factor, opt_eff


# Uniform +/- 20% from nominal. TODO: Make the percentage offset a variable
fwhm_cuts = {
    "030": [6.0, 8.8],
    "040": [4.1, 6.1],
    "090": [1.8, 2.6],
    "150": [1.1, 1.6],
    "220": [0.8, 1.2],
    "280": [0.7, 1.0],
}

# These are roughly 20% - 500% the expected beam value
beam_volume_cuts = {
    "090": [8e-8, 2e-6],
    "150": [4e-8, 1e-6],
    "220": [1e-8, 2e-7],
    "280": [1e-8, 2e-7],
}


