    arrays = bandcenters.keys()

    # Set up arrays to hold our measurements
    radii_saturn = {array: {} for array in arrays}
    means_datas_saturn = {array: {} for array in arrays}
    means_fits_saturn = {array: {} for array in arrays}

    for array in arrays:
        if "ln" in array:
            radii_saturn[array] = {"f030": [], "f040": []}
            means_datas_saturn[array] = {"f030": [], "f040": []}
            means_fits_saturn[array] = {"f030": [], "f040": []}
        elif "mv" in array:
            radii_saturn[array] = {"f090": [], "f150": []}
            means_datas_saturn[array] = {"f090": [], "f150": []}
            means_fits_saturn[array] = {"f090": [], "f150": []}
        elif "uv" in array:
            radii_saturn[array] = {"f220": [], "f280": []}
            means_datas_saturn[array] = {"f220": [], "f280": []}
            means_fits_saturn[array] = {"f220": [], "f280": []}

            radii_mars = copy.deepcopy(radii_saturn)

    means_datas_mars = copy.deepcopy(means_datas_saturn)
    means_fits_mars = copy.deepcopy(means_fits_saturn)
def make_planet_profiles(solved_file: str) -> tuple(np.array, np.array, np.array):
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
