import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import datetime as dt
import ephem

import so3g.proj as proj





rad_to_arcsec = (180 * 3600) / np.pi
rad_to_arcmin = (180 * 60) / np.pi
rad_to_deg = 180 / np.pi

so3gsite = proj.coords.SITES['so']
site = so3gsite.ephem_observer()



def get_radial_mask(data, pixsize, radius):
    x0, y0 = int(data.shape[0] / 2), int(data.shape[1] / 2)  # TODO: subpixel alignment
    X, Y = np.arange(data.shape[0]), np.arange(data.shape[1])
    XX, YY = np.meshgrid(Y, X)

    dist = np.sqrt((XX - x0) ** 2 + (YY - y0) ** 2) * pixsize

    return dist < radius

def get_rand_offsets(r: float=10., num: int=50):
    angles = np.random.rand(num) * 2 * np.pi
    offsets = np.array([[r*np.cos(angle), r*np.sin(angle)] for angle in angles])
    return offsets

def get_aperture_phot(imap, cent, pixsize, r = 2.4, r_rand=10, num_rand=50):
    r = 20
    stamp = imap[cent[0]-r:cent[0]+r,cent[1]-r:cent[1]+r]
    
    stamp_mask = get_radial_mask(stamp, pixsize, 2.4)
    flux = np.sum(stamp[stamp_mask])
    
    rands = get_rand_offsets(r=r_rand, num=num_rand)
    cents = cent+rands

    aperture_phots = np.zeros(len(rands))

    for i, cur_cent in enumerate(cents):
        cur_stamp = imap[int(cur_cent[0])-r:int(cur_cent[0])+r,int(cur_cent[1])-r:int(cur_cent[1])+r]
        mask = get_radial_mask(cur_stamp, pixsize, 2.4)
        aperture_phots[i] = np.sum(cur_stamp[mask])
        
    return flux - np.mean(aperture_phots)

#########################################################################################################################################################
# Beam funcs from F2F Tutorials https://github.com/simonsobs/pwg-tutorials/blob/master/2024_F2F_Tutorials/05_Planet_Data_Processing_and_Mapmaking.ipynb #
#########################################################################################################################################################

def gauss(pixmap, mu_x, mu_y, amp, var):
    '''
    Evaluate 2d Gaussian in pixel space. (From Adri's pwg-tutorials scripts)

    Parameters
    ----------
    pixmap : (2, ny, nx) array
        Y and X pixel indices for each pixel.
    mu_x : float
        Mean pixel index in X.
    mu_y : float
        Mean pixel index in Y.
    amp : float
        Ampltude of Gaussian
    var : float
        Variance of Gaussian in pixel indices^2,

    Returns
    -------
    out : (nx, ny) array
        Gaussian evaluated on input geometry.
    '''

    xx = pixmap[1]
    yy = pixmap[0]

    return np.array(
        amp * np.exp(-0.5 * ((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / var)).ravel()

def rad_avg(x_arr, y_arr, power, rmin, rmax, inc, shift_center=None):
    """Compute radial average of measured power over range [0,deg] of radii.
    
    x_arr,y_arr,rmin,rmax,inc [arcmin]

    """

    # calculate radius from center.
    if shift_center is not None:
        r_i = np.sqrt((x_arr-shift_center[0]) ** 2 + ((y_arr-shift_center[1]) ** 2))
    else:
        r_i = np.sqrt((x_arr) ** 2 + ((y_arr) ** 2))

    # calculate the mean
    def rad_bins(r_b):
        return (power)[(r_i >= r_b - (inc)) & (r_i < r_b + (
            inc))].mean()

    r_phi = np.arange(rmin, rmax, inc)
    means = np.vectorize(rad_bins)(r_phi)
    return r_phi, means

def get_fwhm_radial_bins(r, y, interpolate=False):
    half_point = np.max(y) * .5

    if interpolate:
        r_diff = r[1] - r[0]
        interp_func = interp1d(r, y)
        r_interp = np.arange(np.min(r), np.max(r) - r_diff, r_diff / 100)
        y_interp = interp_func(r_interp)
        r, y = (r_interp, y_interp)
    d = y - half_point
    inds = np.where(d > 0)[0]
    fwhm = 2 * (r[inds[-1]])
    return fwhm

def normalize(data):
    return data / np.max(data)

def arcmin2rad(x):
    return (x / 60 / (180 / np.pi))

def solid_angle(az, el, beam):
    '''Compute the integrated solid angle of a beam map.

    az and el are given in ARCMIN

    return value is in steradians  (sr)
    '''

    # convert from arcmin to rad
    az = arcmin2rad(az)
    el = arcmin2rad(el)
    integrand = beam
    # perform the solid angle integral
    integral = np.trapz(np.trapz(integrand, el, axis=0), az, axis=0)
    return integral

def fit_gauss_pointing(imap, ivar, pixmap, make_plots=True):
    '''
    Fit 2d Gaussian to input map.

    Arguments
    ---------
    imap : (ny, nx) enmap
        Input map.
    ivar : (ny, nx) enmap
        Inverse-variance map.
    pixmap : (2, ny, nx) array
        X and Y pixel indices for each pixel.

    Returns
    -------
    amp : float
        Amplitude of Gaussian.
    shift_x : float
        X shift needed to center the Gaussian in middle of map
    shift_y : float
        Y shift needed to center the Gaussian in middle of map
    fwhm    
        Fitted FWHM of Gaussian
    '''

    ny, nx = imap.shape[-2:]
    mid_y, mid_x = np.asarray(imap.shape[-2:]) / 2

    sigma = np.sqrt(ivar)
    sigma = np.divide(1, sigma, where=sigma != 0)

    # Set to numerically large value.
    sigma[ivar == 0] = sigma[~(ivar == 0)].max() * 1e5


    # Start at middle pixel, using max value as guess for amp.
    max_pix = enmap.argmax(imap, unit='pix')

    guess = [max_pix[1],
             max_pix[0],
             imap[max_pix[0], max_pix[1]],
             10]

    # You may need to adjust these bounds as needed depending on your source.
    bounds = ([0, 0, 0, 1], [nx, ny, np.inf, 50])

    popt, pcov = curve_fit(gauss, pixmap.astype(float), imap.ravel(),
                           p0=guess, sigma=sigma.ravel(), bounds=bounds)

    shift_x = -(popt[0] - mid_x)
    shift_y = -(popt[1] - mid_y)

    # factor to convert units of pixels to meaningful (arcmin) units
    mult_factor = (imap.wcs.wcs.cdelt[1] * (60))
    
    # convert sigma to FWHM in arcmin
    fwhm = np.sqrt(popt[3]) * 2 * np.sqrt(2 * np.log(2)) * mult_factor
    
    data_fitted = gauss(pixmap.astype(float), *popt).reshape(imap.shape)
    max_val = np.max(imap)
    min_val = np.min(imap)
    
    res = pixmap.wcs.wcs.cdelt[1] * 60 # convert to arcmin
    n_pix = imap.shape[0]
    span = n_pix * res
    x_vals = np.arange(-span / 2, span / 2, res)
    y_vals = np.arange(-span / 2, span / 2, res)
    x_vals, y_vals = np.meshgrid(x_vals, y_vals)
    
    radii_data, means_data = rad_avg(x_vals, y_vals, imap, 0, span/2, 0.5, 
                                 shift_center=[-shift_x * res, -shift_y * res])
    radii_fit, means_fit = rad_avg(x_vals, y_vals, data_fitted, 0, span/2, 0.5, 
                                   shift_center=[-shift_x * res, -shift_y * res])
    data_fwhm = get_fwhm_radial_bins(radii_data, means_data, interpolate=True)
    
    y_centered_integrand = y_vals[:, 0] + shift_y * res
    x_centered_integrand = x_vals[0] + shift_x * res

    data_solid_angle = solid_angle(x_centered_integrand, y_centered_integrand, normalize(imap)) * (1e6)
    
    fit_sigma = arcmin2rad(np.sqrt(popt[3]) * mult_factor)
    fit_solid_angle = 2 * np.pi * (fit_sigma ** 2) * (1e6)
    # For plotting the fits
    if make_plots:
        plt.subplots(1, 3, figsize=(13, 3))
        plt.subplot(1, 3, 1)
        
        plt.gca().set_aspect('equal')
        
        plt.pcolormesh(x_vals, y_vals, imap, vmin=min_val, vmax=np.max(imap))
        plt.xlim(-span/2, span/2)
        plt.ylim(-span/2, span/2)
        plt.xlabel('[arcmin]')
        plt.ylabel('[arcmin]')
        plt.colorbar(label='[pW]')

        levels=[max_val * 0.1, max_val * .25, max_val * .5,
                max_val * 0.75, max_val]
        
        cs = plt.contour(x_vals, y_vals, data_fitted,
                         cmap='viridis', vmin=min_val,
                         vmax=max_val, levels=levels)
        
        plt.subplot(1, 3, 2)

        plt.plot(radii_data, means_data, label=f"Data: {data_fwhm:.2f}")
        plt.plot(radii_fit, means_fit, label=f"Fit: {fwhm:.2f}")
        plt.title(f'FWHM [arcmin]')
        plt.axhline(0, ls="--", color="black")
        plt.xlabel('[arcmin]')
        plt.xlim(0, span/2)
        plt.legend()
        
        plt.subplot(1, 3, 3)

        plt.plot(radii_data, normalize(means_data), label=f"Data: {data_solid_angle:.2f}")
        plt.plot(radii_data, normalize(means_fit), label=f"Fit: {fit_solid_angle:.2f}")
        plt.title(fr'beam solid angle [$\cdot 10^{{-6}}$ sr]')
        plt.legend()
        plt.ylim(1e-4, 1)
        plt.yscale('log')
        plt.xlabel('[arcmin]')
        plt.xlim(0, span/2)

        plt.subplots_adjust(wspace=0.55)
        plt.show()
        
    chisred = np.sum((imap-data_fitted)**2/(sigma**2))/imap.shape[0]**2

    return popt[2], shift_x, shift_y, fwhm, np.max(np.array([data_solid_angle, fit_solid_angle])), chisred, popt, pcov

def get_planet_diameter(obs_id, planet):
    """get angular diameter (in ARCSEC) of a planet from obs id"""
    timestamp = int(obs_id)
    date = dt.datetime.utcfromtimestamp(timestamp)
    site.date  = ephem.Date(date)
    saturn = getattr(ephem, planet)(site)
    dia = getattr(saturn, "size")
    return dia

def angular_diameter_to_solid_angle(angular_diameter):
    """Angular diameter is in ARCSEC"""
    return 2 * np.pi * (1 - np.cos(np.deg2rad(angular_diameter / 3600) / 2))

def get_pwv_obs(obs_id, pwv_data, n_hours_before, n_hours_after):
    ## factor needed to get the maps at 1 mm PWV
    timestamp = int(obs_id)
    msk = np.all([pwv_data[
        'timestamp'] > timestamp - n_hours_before * 3600, pwv_data[
            'timestamp'] < timestamp + n_hours_after * 3600], axis=0)
    pwv_obs = np.nanmean(pwv_data['pwv'][msk])
    return pwv_obs
