import os
import tempfile
from pathlib import Path

import dill as pk
import numpy as np
from marsmodel import mars_brightness

CACHE_PATH = Path(__file__).with_name("mars_temps.pk")


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
    bands = ["f030", "f040", "f090", "f150", "f220", "f280"]
    # mars_temps.pk is keyed by the band without the "f", e.g. "090"
    cache_key = band.removeprefix("f")
    with open(CACHE_PATH, "rb") as f:
        T_b = pk.load(f)
    if band not in bands:
        raise ValueError(
            "Invalid band. Must be one of 'f030', 'f040', 'f090', 'f150', 'f220', 'f280'."
        )
    tb_time = 0
    for key in T_b:
        if np.abs(int(obs_id) - int(key)) <= 300:  # Within 5 minutes is OK
            tb_time = key
            return T_b[tb_time][cache_key]
    if not tb_time:
        T_b[int(obs_id)] = {}
        r = mars_brightness(
            int(obs_id),
            hpbw=30,
            roughness=12,
            penetration=12,
            dielectric=2.25,
            frequencies=[30, 40, 90, 150, 220, 280],
        )
        for i, cur_band in enumerate(bands):
            T_b[int(obs_id)][cur_band.removeprefix("f")] = r.Tb[i]
        # Write to a unique temp file and rename it over the cache, so threads
        # and MPI ranks reading the cache never see a partially written file.
        # Concurrent writers can drop each other's new entries, which only costs
        # recomputing them later.
        fd, tmp = tempfile.mkstemp(dir=CACHE_PATH.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pk.dump(T_b, f)
        os.chmod(tmp, 0o660)
        os.replace(tmp, CACHE_PATH)

        return T_b[int(obs_id)][cache_key]
