def get_saturn_temp(band: str):
    """
    Get the temperature of Saturn for a given band.

    Parameters
    ----------
    band : str
        The band for which to get the temperature.

    Returns
    -------
    saturn_temp : float
        The temperature of Saturn in Kelvin.

    Raises
    ------
    ValueError
        If the band is not valid.
    """
    if band == "030":
        # Revised planet brightness temperatures using the Planck/LFI 2018 data release, table 10, T_d,c Channel 30
        saturn_temp = 140.0
    elif band == "040":
        # Revised planet brightness temperatures using the Planck/LFI 2018 data release, table 10, T_d,c Channel 44
        saturn_temp = 148.3
    if band == "090":
        # Seven-Year Wilkinson Microwave Anisotropy Probe (WMAP1) Observations: Planets and Celestial Calibration Sources, table 9. Consistency with MF SATs. Effective bandcenter 93GHz for WMAP
        saturn_temp = 143.3
    elif band == "150":
        # Planck intermediate results. LII. Planet flux densities, table 6. Effective planck bandcenter 143GHz. Note this is the combined planet ring system. Consistency with MF SATs.
        saturn_temp = 143.6
    elif band == "220":
        # Planck intermediate results. LII. Planet flux densities, table 6. Effective planck bandcenter 143GHz.
        saturn_temp = 139.7
    elif band == "280":
        # Fit k=2 interp to Planck LLI table 6, using 143, 217, 353 points, interpolated at the average bandcenter for the Nominal SO tubes of 284.5
        saturn_temp = 136.3
    else:
        raise ValueError("Error: invalid band {}".format(band))
    return saturn_temp
