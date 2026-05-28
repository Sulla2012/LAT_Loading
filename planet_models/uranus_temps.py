from astropy.io import fits
from scipy.interpolate import make_interp_spline


def get_uranus_temp(band: str):
    """
    Get the temperature of Uranus for a given band.

    Parameters
    ----------
    band : str
        The band for which to get the temperature.

    Returns
    -------
    uranus_temp : float
        The temperature of Uranus in Kelvin.

    Raises
    ------
    ValueError
        If the band is not valid.
    """
    hdu = fits.open("uranus_esa4.fits")
    data = hdu[1].data
    freqs = data["wave"]
    T_rj = data["T_rj"]
    interp = make_interp_spline(freqs, T_rj)

    # TODO: accurate bandcenters
    if band == "030":
        uranus_temp = interp(30)
    elif band == "040":
        uranus_temp = interp(40)
    elif band == "090":
        uranus_temp = interp(90)
    elif band == "150":
        uranus_temp = interp(150)
    elif band == "220":
        uranus_temp = interp(220)
    elif band == "280":
        uranus_temp = interp(280)
    else:
        raise ValueError("Error: invalid band {}".format(band))
    return uranus_temp
