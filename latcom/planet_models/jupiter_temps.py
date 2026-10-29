import numpy as np

from ..bands.bands import bandcenters


def get_jupiter_temp(band: str, ufm: str) -> float:
    """
    Get the temperature of Jupiter for a given band.

    Parameters
    ----------
    band : str
        The band for which to get the temperature.
    ufm : str
        The UFM identifier for the band.

    Returns
    -------
    jupiter_temp : float
        The temperature of Jupiter in Kelvin.

    Raises
    ------
    ValueError
        If the band is not valid.
    """
    if band != "f030" and band != "f040":
        raise ValueError("Error: Jupiter is only for use with LF")
    # Maris et. al. 2021 (A&A 647, A104), Table 6: T_b,rj at Planck LFI centers
    # Plancks f40 is pretty different in bandcenter from ours so use quadratic to estimate offset.
    # Note linear appx differs by >0.5K = >0.2%
    PLANCK_NU = np.array([28.40, 44.10, 70.40])  # GHz
    PLANCK_TB_RJ = np.array([144.93, 159.76, 170.50])  # K
    return float(
        np.polyval(np.polyfit(PLANCK_NU, PLANCK_TB_RJ, 2), bandcenters[ufm][band])
    )
