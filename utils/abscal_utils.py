from . import map_utils as mu
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from pixell import enmap


def make_planet_profiles(planet, obs_id, stream_id, band, ufm):
    solved_file = (
        "/so/home/saianeesh/data/beams/lat/source_maps/per_obs/{}/".format(planet)
        + str(obs_id[:5])
        + "/"
        + str(obs_ids[i])
        + "/"
        + str(obs_ids[i])
        + "_"
        + str(stream_ids[i])
        + "_"
        + str(bands[i])
        + "_solved.fits"
    )
    weight_file = solved_file.replace("solved", "weights")
    binned_file = solved_file.replace("solved", "binned")
    try:
        solved = enmap.read_map(solved_file)[0]
        weights = enmap.read_map(weight_file)[0][0]
        binned = enmap.read_map(binned_file)[0]

    except FileNotFoundError:
        return None, None, None

    kernel = Gaussian2DKernel(5)
    smoothed = convolve(solved, kernel)
    cent = np.argmax(smoothed)
    cent = np.unravel_index(cent, solved.shape)

    r = 20
    solved = solved[cent[0] - r : cent[0] + r, cent[1] - r : cent[1] + r]
    weights = weights[cent[0] - r : cent[0] + r, cent[1] - r : cent[1] + r]
    binned = binned[cent[0] - r : cent[0] + r, cent[1] - r : cent[1] + r]
    # plt.imshow(solved)
    # plt.colorbar()
    pixmap = enmap.pixmap(solved.shape, solved.wcs)
    (
        fitted_amp,
        shift_x,
        shift_y,
        fitted_fwhm,
        data_solid_angle,
        chisred,
        popt,
        pcov,
        radii_data,
        means_data,
        means_fit,
    ) = mu.fit_gauss_pointing(solved, weights, pixmap, make_plots=True)

    return radii_data, means_data, means_fit
