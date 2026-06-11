from functools import cache

from astropy.io import fits
from scipy.interpolate import make_interp_spline

from ..bands.bands import bandcenters


@cache  # We could pre-calculate the bandcenter adjusted temepratures and save it but this is more robust to changing the bandcenters and should be nearly as fast.
def get_uranus_temp(band: str, ufm: str) -> float:
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
    KeyError
        If ufm or band is invalid
    """
    hdu = fits.open(
        "/global/u2/j/jorlo/dev/LAT_Loading/latcom/planet_models/uranus_esa4.fits"
    )
    data = hdu[1].data
    freqs = data["wave"]
    T_rj = data["T_rj"]
    interp = make_interp_spline(freqs, T_rj)
    try:
        bandcenter = bandcenters[ufm][band]
    except KeyError:
        raise KeyError(f"Error: invalid ufm {ufm} or band {band}")

    # TODO: accurate bandcenters
    if band == "030":
        uranus_temp = interp(27.9)
    elif band == "040":
        uranus_temp = interp(40.2)
    elif band == "090" or band == "150" or band == "220" or band == "280":
        uranus_temp = interp(bandcenter)
    else:
        raise ValueError(f"Error: invalid band {band}")
    return uranus_temp
