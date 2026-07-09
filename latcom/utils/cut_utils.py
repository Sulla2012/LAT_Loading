import numpy as np
from sotodlib.core.axisman import AxisManager


def band_from_ufm(ufm: str, band: str) -> int:
    """
    Get the band number from a UFM name.

    Parameters
    ----------
    ufm : str
        The UFM identifier.
    band : str
        The band identifier.

    Returns
    -------
    int
        The band number.

    Raises
    ------
    ValueError
        If the UFM or band is not recognized.
    """
    if "ln" in ufm:
        if band == "030":
            ufm_band = 1
        elif band == "040":
            ufm_band = 2
        else:
            raise ValueError(f"Unknown band {band} for UFM {ufm}")
    elif "mv" in ufm:
        if band == "090":
            ufm_band = 1
        elif band == "150":
            ufm_band = 2
        else:
            raise ValueError(f"Unknown band {band} for UFM {ufm}")
    elif "uv" in ufm:
        if band == "220":
            ufm_band = 1
        elif band == "280":
            ufm_band = 2
        else:
            raise ValueError(f"Unknown band {band} for UFM {ufm}")

    else:
        raise ValueError(f"Unknown UFM {ufm}")

    return ufm_band


def get_wnoise_yield(meta: AxisManager, ufm: str, band: str) -> float:
    """
    Get the number of dets with white noise estimates
    from a metadata AxisManager.

    Parameters
    ----------
    meta : AxisManager
        The metadata containing white noise information.
    ufm : str
        The UFM identifier.
    band : str
        The band identifier.

    Returns
    -------
    len(np.where(wnoise != 0)[0]) : int
        Number of dets with non-zero white noise estimates.

    Raises
    ------
    ValueError
        If no valid noise key is found in the metadata.
    """
    wafer_flag = np.array([ufm in _ufm for _ufm in meta.det_info.stream_id])
    obs_id = meta.obs_info.obs_id
    if len(wafer_flag) == 0:
        raise ValueError(f"No det info for obs {obs_id}.")

    bp = (meta.det_cal.bg % 4) // 2
    try:
        ufm_band = band_from_ufm(ufm=ufm, band=band)
    except ValueError as e:
        raise ValueError(
            f"Error occurred while getting band number for UFM {ufm} and band {band}: {e}"
        )

    if ufm_band == 1:
        net_flag = wafer_flag * (bp == 0)
    elif ufm_band == 2:
        net_flag = wafer_flag * (bp == 1)
    if "noise" in meta.preprocess:
        wnoise = meta.preprocess.noise.white_noise[net_flag]
    elif "noiseT" in meta.preprocess:
        wnoise = meta.preprocess.noiseT.white_noise[net_flag]
    else:
        raise ValueError(f"Error: no valid noise key in {meta.preprocess.keys()}")

    return len(np.where(wnoise != 0)[0])


def get_yield_stages(
    meta: AxisManager, ufm: str, band: str
) -> tuple[list[int], list[str]]:
    """
    Get the yield at each stage of the analysis.

    Parameters
    ----------
    meta : AxisManager
        The metadata containing preprocessing information.
    ufm : str
        The UFM identifier.
    band : str
        The band identifier.

    Returns
    -------
    ndets, stage_name : tuple[list[int], list[str]]
        The the stages and the corresponding number of dets at that stage.
    """
    ndet = []
    stage_name = ["initial"]

    for key in meta.preprocess.keys():
        try:
            if "valid" in meta.preprocess[key].keys():
                cur_ndet = len(
                    [
                        flag
                        for flag in np.all(meta.preprocess[key].valid.mask(), axis=-1)
                        if flag
                    ]
                )
                ndet.append(cur_ndet)
                stage_name.append(key)
        except AttributeError:
            continue
    try:
        ndet.append(get_wnoise_yield(meta=meta, ufm=ufm, band=band))
    except ValueError as e:
        print(f"Error occurred while getting wnoise yield: {e}")

    return ndet, stage_name


def get_cut_sources(
    meta: AxisManager, ufm: str, band: str
) -> tuple[list[int], list[str]]:
    """
    Get the number of cuts from each source
    in a preprocessing metadata. Note this assumes
    that the preprocessing steps were applied in
    the same order as they appear in the metadata.
    As far as I can tell this is true.

    Parameters
    ----------
    meta : AxisManager
        The metadata containing preprocessing information.
    ufm : str
        The UFM identifier.
    band : str
        The band identifier.

    Returns
    -------
    ncut, cut_name : tuple[list[int], list[str]]
        The number of cuts and their corresponding names.

    Raises
    ------
    ValueError
        If error occurs while getting white noise yield.
    """
    ndet, stage_name = get_yield_stages(meta, ufm, band)

    return -1 * np.diff(ndet), stage_name[1:]


def get_det_cal_cuts(meta: AxisManager) -> tuple[list[int], list[str]]:
    """
    Get the number of dets with bad det cal
    stemming from different sources. Dets
    are considered to have bad det cal if the
    value of the corresponding parameter is nan
    or 0.

    Parameters
    ----------
    meta : AxisManager
        The metadata containing detector calibration information.

    Returns
    -------
    ncut, cut_name : tuple[list[int], list[str]]
        The number of cuts and their corresponding names.
    """
    ncut = [0]
    cut_name = []

    for key in meta.det_cal.keys():
        try:
            ncut = len(np.where(np.isnan(meta.det_cal[key][net_flag]))[0]) + len(
                np.where(np.isnan(meta.det_cal[key][net_flag]))[0]
            )
            ncut.append(ncut)
            cut_name.append(key)
        except:
            continue

    return np.diff(ncut), cut_name
