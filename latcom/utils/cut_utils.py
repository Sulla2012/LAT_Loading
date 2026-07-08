import numpy as np
from sotodlib.core.axisman import AxisManager


def get_cut_sources(meta: AxisManager) -> tuple[list[int], list[str]]:
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

    Returns
    -------
    ncut, cut_name : tuple[list[int], list[str]]
        The number of cuts and their corresponding names.
    """
    ncut = [0]
    cut_name = []

    for key in meta.preprocess.keys():
        try:
            if "valid" in meta.preprocess[key].keys():
                cur_cut = len(
                    [
                        flag
                        for flag in np.all(~meta.preprocess[key].valid.mask(), axis=-1)
                        if flag
                    ]
                )
                ncut.append(cur_cut)
                cut_name.append(key)
        except AttributeError:
            continue

    return np.diff(ncut), cut_name


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
