import numpy as np
from scipy import interpolate
from sotodlib import core
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.tod_ops.flags import get_det_bias_flags

from latcom.utils.optical_loading import ufm_dict


def gen_empty_net_dict() -> dict:
    """
    Generate an emtpy net_dict with the appropriate keys from an abscal dict.

    Returns
    -------
    net_dict : dict
        Empty NET dictionary.
    """
    net_dict = {}
    for ufm_list in ufm_dict.values():
        for ufm in ufm_list:
            if "ln" in ufm:
                net_dict[ufm] = {
                    "030": {
                        "obs": [],
                        "ndets": [],
                        "nets": [],
                        "raw_cal": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                    "040": {
                        "obs": [],
                        "ndets": [],
                        "nets": [],
                        "raw_cal": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                }
            elif "mv" in ufm:
                net_dict[ufm] = {
                    "090": {
                        "obs": [],
                        "ndets": [],
                        "nets": [],
                        "raw_cal": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                    "150": {
                        "obs": [],
                        "ndets": [],
                        "nets": [],
                        "raw_cal": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                }
            elif "uv" in ufm:
                net_dict[ufm] = {
                    "220": {
                        "obs": [],
                        "ndets": [],
                        "nets": [],
                        "raw_cal": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                    "280": {
                        "obs": [],
                        "ndets": [],
                        "nets": [],
                        "raw_cal": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                }

    return net_dict


def gen_empty_nep_dict() -> dict:
    """
    Generate an emtpy nep_dict with the appropriate keys from an abscal dict.


    Returns
    -------
    nep_dict : dict
        Empty NEP dictionary.
    """
    nep_dict = {}
    for ufm_list in ufm_dict.values():
        for ufm in ufm_list:
            if "ln" in ufm:
                nep_dict[ufm] = {
                    "030": {
                        "obs": [],
                        "ndets": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                    "040": {
                        "obs": [],
                        "ndets": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                }
            elif "mv" in ufm:
                nep_dict[ufm] = {
                    "090": {
                        "obs": [],
                        "ndets": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                    "150": {
                        "obs": [],
                        "ndets": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                }
            elif "uv" in ufm:
                nep_dict[ufm] = {
                    "220": {
                        "obs": [],
                        "ndets": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                    "280": {
                        "obs": [],
                        "ndets": [],
                        "el": [],
                        "pwv": [],
                        "neps": [],
                        "phiconv": [],
                    },
                }

    return nep_dict


def get_nets(
    obs_id: str,
    abscal_list: list,
    pwv: interpolate.interp1d,
    ctx_path: str,
) -> tuple[list, list, list, list, list, list, list, list, list, list] | None:
    """
    Function which computes the NET as well as NEP and some other parameters
    for a given observation. Computes these parameters over all available
    wafers and bands. Returns none if there is no metadata for the obs.

    Parameters
    ----------
    obs_id : str
        Obs id of the observation
    abscal_list : list
        List of array/band combinations we have abscals for.
    pwv : interpolate-interp1d
        Interpolation function for pwv.
    ctx : str
        Path to context object for loading metadata

    Returns
    -------
    arrays : list
        List of arrays corresponding to each return entry
    ret_bands : list
        List of bands corresponding to each return entry
    raw_cals : list
        Raw abscal for array, band at time of obs
    obs_ids : list
        Obs ids corresponding to each return entry
    ndets : list
        Number of functioning detectors
    array_nets : list
        Inverse variance averaged NET over the array
    pwvs : list
        PWVs corresponding to each return entry
    els : list
        Central elevations corresponding to each return entry
    neps : list[list]
        List of per detector NEPs
    phiconvs : lsit
        Phi to pW conversions corresponding to each return entry

    """

    arrays = []
    ret_bands = []

    raw_cals = []
    obs_ids = []
    ndets = []
    array_nets = []
    pwvs = []
    els = []
    neps = []
    phiconvs = []

    ctx = core.Context(ctx_path)
    try:  # Much faster than ctx.get_meta
        det_info = ctx.get_det_info(obs_id)
    except LoaderError:
        print(f"No meta data for obs {obs_id}")
        return None
    wafers = np.unique(det_info["stream_id"])
    bands = np.unique(det_info["wafer.bandpass"])
    bands = np.array([b[1:] for b in bands if len(b) > 1 and b[0] == "f"])

    for j in range(len(wafers)):
        cur_wafer = wafers[j].split("_")[-1]

        if cur_wafer not in abscal_list:
            print(f"No abscal for ufm {cur_wafer}")
            continue

        for band in bands:
            if "mv" in cur_wafer:
                if band == "090":
                    ufm_band = 1
                elif band == "150":
                    ufm_band = 2
            if "uv" in cur_wafer:
                if band == "220":
                    ufm_band = 1
                elif band == "280":
                    ufm_band = 2

            try:
                meta = ctx.get_meta(
                    obs_id,
                    dets={
                        "dets:stream_id": "ufm_" + str(cur_wafer),
                        "dets:wafer.bandpass": "f" + str(band),
                    },
                )
            except LoaderError:
                print(f"No meta data for obs {obs_id}")
                continue

            flags = get_det_bias_flags(meta).det_bias_flags
            meta.restrict("dets", ~core.flagman.has_any_cuts(flags))
            wafer_flag = np.array([cur_wafer in ufm for ufm in meta.det_info.stream_id])

            if len(wafer_flag) == 0:
                print(f"No det_info for obs {obs_id}")
                continue

            bp = (meta.det_cal.bg % 4) // 2

            if ufm_band == 1:
                net_flag = wafer_flag * (bp == 0)
            elif ufm_band == 2:
                net_flag = wafer_flag * (bp == 1)

            raw_cal = np.nanmedian(meta.abscal.raw_abscal_rj[net_flag])
            if "noise" in meta.preprocess:
                wnoise = meta.preprocess.noise.white_noise[net_flag]
            elif "noiseT" in meta.preprocess:
                wnoise = meta.preprocess.noiseT.white_noise[net_flag]
            else:
                print(f"Error: no valid noise ken in {meta.preprocess.keys()}")
                continue
            ndet = len(np.where(wnoise != 0)[0])

            net_mes = 1 / np.sqrt(2) * wnoise * raw_cal
            clean_nets = []
            for net in net_mes:
                if net * 1e6 > 100:
                    clean_nets.append(net)
            clean_nets = np.array(clean_nets)
            array_net = np.nansum((clean_nets * 1e6) ** (-2)) ** (-1 / 2)

            arrays.append(cur_wafer)
            ret_bands.append(band)

            raw_cals.append(raw_cal)
            obs_ids.append(obs_id)
            ndets.append(ndet)
            array_nets.append(array_net)
            pwvs.append(pwv(obs_id.split("_")[1]))
            els.append(meta.obs_info.el_center)
            neps.append(wnoise)
            phiconvs.append(meta.det_cal.phase_to_pW[net_flag])

    return (
        arrays,
        ret_bands,
        raw_cals,
        obs_ids,
        ndets,
        array_nets,
        pwvs,
        els,
        neps,
        phiconvs,
    )


def get_neps(
    obs_id: str,
    pwv: interpolate.interp1d,
    ctx_path: str,
) -> tuple[list, list, list, list, list, list, list, list] | None:
    """
    Function which computes just the NEPs and some other parameters
    for a given observation. Computes these parameters over all available
    wafers and bands. Returns none if there is no metadata for the obs.
    Note that NEPs don't rely on abscals so this function exists to
    provide NEP measurements when abscals are not available. Otherwise
    get_nets should be used as it also returns NEPs.

    Parameters
    ----------
    obs_id : str
        Obs id of the observation
    pwv : interpolate-interp1d
        Interpolation function for pwv.
    ctx : str
        Path to context object for loading metadata

    Returns
    -------
    arrays : list
        List of arrays corresponding to each return entry
    ret_bands : list
        List of bands corresponding to each return entry
    obs_ids : list
        Obs ids corresponding to each return entry
    ndets : list
        Number of functioning detectors
    pwvs : list
        PWVs corresponding to each return entry
    els : list
        Central elevations corresponding to each return entry
    neps : list[list]
        List of per detector NEPs
    phiconvs : lsit
        Phi to pW conversions corresponding to each return entry

    """

    arrays = []
    ret_bands = []

    obs_ids = []
    ndets = []
    pwvs = []
    els = []
    neps = []
    phiconvs = []

    ctx = core.Context(ctx_path)
    try:  # Much faster than ctx.get_meta
        det_info = ctx.get_det_info(obs_id)
    except LoaderError:
        print(f"No meta data for obs {obs_id}")
        return None
    wafers = np.unique(det_info["stream_id"])
    bands = np.unique(det_info["wafer.bandpass"])
    bands = np.array([b[1:] for b in bands if len(b) > 1 and b[0] == "f"])

    for j in range(len(wafers)):
        cur_wafer = wafers[j].split("_")[-1]

        for band in bands:
            if "mv" in cur_wafer:
                if band == "090":
                    ufm_band = 1
                elif band == "150":
                    ufm_band = 2
            if "uv" in cur_wafer:
                if band == "220":
                    ufm_band = 1
                elif band == "280":
                    ufm_band = 2

            try:
                meta = ctx.get_meta(
                    obs_id,
                    dets={
                        "dets:stream_id": "ufm_" + str(cur_wafer),
                        "dets:wafer.bandpass": "f" + str(band),
                    },
                )
            except LoaderError:
                print(f"No meta data for obs {obs_id}")
                continue

            flags = get_det_bias_flags(meta).det_bias_flags
            meta.restrict("dets", ~core.flagman.has_any_cuts(flags))
            wafer_flag = np.array([cur_wafer in ufm for ufm in meta.det_info.stream_id])

            if len(wafer_flag) == 0:
                print(f"No det_info for obs {obs_id}")
                continue

            bp = (meta.det_cal.bg % 4) // 2

            if ufm_band == 1:
                net_flag = wafer_flag * (bp == 0)
            elif ufm_band == 2:
                net_flag = wafer_flag * (bp == 1)

            if "noise" in meta.preprocess:
                wnoise = meta.preprocess.noise.white_noise[net_flag]
            elif "noiseT" in meta.preprocess:
                wnoise = meta.preprocess.noiseT.white_noise[net_flag]
            else:
                print(f"Error: no valid noise ken in {meta.preprocess.keys()}")
                continue
            ndet = len(np.where(wnoise != 0)[0])

            arrays.append(cur_wafer)
            ret_bands.append(band)

            obs_ids.append(obs_id)
            ndets.append(ndet)
            pwvs.append(pwv(obs_id.split("_")[1]))
            els.append(meta.obs_info.el_center)
            neps.append(wnoise)
            phiconvs.append(meta.det_cal.phase_to_pW[net_flag])
    if len(arrays) == 0:
        return None

    return (
        arrays,
        ret_bands,
        obs_ids,
        ndets,
        pwvs,
        els,
        neps,
        phiconvs,
    )


def parse_net_results(results: list) -> dict:
    """
    Function which parses the list based results of get_nets
    into the expected results dictionary. get_nets is set to
    be parallelizable, so it has to return list instead of the
    desired dict.

    Parameters
    ----------
    results : list
        List of results by obs as produced by get_nets


    Returns
    -------
    net_dict : dict
        Result dictionary of results reorganized by array/band

    """
    net_dict = gen_empty_net_dict()

    for result in results:
        if result is None:
            continue
        (
            arrays,
            bands,
            raw_cals,
            obs_ids,
            ndets,
            array_nets,
            pwvs,
            els,
            neps,
            phiconvs,
        ) = result
        for i in range(len(arrays)):
            array = arrays[i]
            band = bands[i]
            net_dict[array][band]["raw_cal"].append(raw_cals[i])

            net_dict[array][band]["obs"].append(obs_ids[i])
            net_dict[array][band]["ndets"].append(ndets[i])
            net_dict[array][band]["nets"].append(array_nets[i])
            net_dict[array][band]["pwv"].append(pwvs[i])
            net_dict[array][band]["el"].append(els[i])
            net_dict[array][band]["neps"].append(neps[i])
            net_dict[array][band]["phiconv"].append(phiconvs[i])

    return net_dict


def parse_nep_results(results: list) -> dict:
    """
    Function which parses the list based results of get_neps
    into the expected results dictionary. get_neps is set to
    be parallelizable, so it has to return list instead of the
    desired dict.

    Parameters
    ----------
    results : list
        List of results by obs as produced by get_neps

    Returns
    -------
    nep_dict : dict
        Result dictionary of results reorganized by array/band

    """
    nep_dict = gen_empty_nep_dict()

    for result in results:
        if result is None:
            continue
        (
            arrays,
            bands,
            obs_ids,
            ndets,
            pwvs,
            els,
            neps,
            phiconvs,
        ) = result
        for i in range(len(arrays)):
            array = arrays[i]
            band = bands[i]

            nep_dict[array][band]["obs"].append(obs_ids[i])
            nep_dict[array][band]["ndets"].append(ndets[i])
            nep_dict[array][band]["pwv"].append(pwvs[i])
            nep_dict[array][band]["el"].append(els[i])
            nep_dict[array][band]["neps"].append(neps[i])
            nep_dict[array][band]["phiconv"].append(phiconvs[i])

    return nep_dict
