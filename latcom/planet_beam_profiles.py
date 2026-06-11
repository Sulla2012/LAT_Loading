"""
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

        if args.make_profiles:
            solved_file = (
                "/so/home/saianeesh/data/beams/lat/source_maps/per_obs/{}/".format(
                    planet
                )
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
            radii_data, means_data, means_fit = au.make_planet_profiles(
                solved_file=solved_file,
            )
            if radii_data is None:
                continue
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

    if parser.parse_args().make_profiles:
        rad_dict = {
            "rad_sat": radii_saturn,
            "data_sat": means_datas_saturn,
            "fit_sat": means_fits_saturn,
            "rad_mars": radii_mars,
            "data_mars": means_datas_mars,
            "fit_mars": means_fits_mars,
        }
        with open("mars_saturn.pk", "wb") as f:
            pk.dump(rad_dict, f)
"""