import numpy as np
import dill as pk
from marsmodel import mars_brightness


def get_mars_temp(obs_id: str, band: str):
    """
    Get the brightness temperature of Mars for a given observation ID and band.

    Parameters:
    -----------
    obs_id : str
           The observation ID for which to retrieve the brightness temperature.
    band : str
           The band for which to retrieve the brightness temperature.

    Returns:
    --------
    T_b[tb_time][band] : float
           The brightness temperature of Mars for the given observation ID and band.
    Raises:
    -------
    ValueError
           If no Mars data is available for the given observation ID.
    """
    bands = ["030", "040", "090", "150", "220", "280"]
    with open("/global/u2/j/jorlo/dev/LAT_Loading/latcom/planet_models/mars_temps.pk", "rb") as f:
        T_b = pk.load(f)
    if band not in bands:
        raise ValueError(
            "Invalid band. Must be one of '030', '040', '090', '150', '220', '280'."
        )
    tb_time = 0
    for key in T_b:
        if np.abs(int(obs_id) - int(key)) <= 300:  # Within 5 minutes is OK
            tb_time = key
            return T_b[tb_time][band]
    if not tb_time:
        T_b[int(obs_id)] = {}
        r = mars_brightness(int(obs_id), hpbw=30, roughness=12,
                    penetration=12, dielectric=2.25,
                    frequencies=[30, 40, 90, 150, 220, 280])
        for i, cur_band in enumerate(bands):
            T_b[int(obs_id)][cur_band] = r.Tb[i]
        with open("/global/u2/j/jorlo/dev/LAT_Loading/latcom/planet_models/mars_temps.pk", "wb") as f:
            pk.dump(T_b, f)

        return T_b[int(obs_id)][band]
    


