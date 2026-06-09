from .mars_temps import get_mars_temp
from .saturn_temps import get_saturn_temp
from .uranus_temps import get_uranus_temp


def get_planet_temp(planet: str, obs_id: str, band: str, ufm: str) -> float:
    """
    Get the planet temperature for a given planet, observation ID, and band.

    Parameters
    ----------
    planet : str
        The name of the planet.
    obs_id : str
        The observation ID.
    band : str
        The band for which to get the temperature.

    Returns
    -------
    planet_temp : float
        The temperature of the planet in Kelvin.

    Raises
    ------
    ValueError
        If the planet is not valid.
    """
    if planet.lower() == "mars":
        planet_temp = get_mars_temp(obs_id, band)
    elif planet.lower() == "saturn":
        planet_temp = get_saturn_temp(band)
    elif planet.lower() == "uranus":
        planet_temp = get_uranus_temp(band=band, ufm=ufm)
    else:
        raise ValueError("Error: invalid planet {}".format(planet))
    return planet_temp
