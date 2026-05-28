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
    if band == "090":
        saturn_temp = 142.9
    elif band == "150":
        saturn_temp = 142.6
    elif band == "220":
        saturn_temp = 139.7
    elif band == "280":
        saturn_temp = 138.7
    else:
        raise ValueError("Error: invalid freq {}".format(freq))
    return saturn_temp
