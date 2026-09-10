import astropy.units as u
import numpy as np
import pandas as pd
from scipy import interpolate
from sotodlib import core
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.tod_ops.flags import get_det_bias_flags

from latcom.utils.map_utils import temp_conv
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
    obs_ctx: tuple[str, str],
    abscal_list: list,
    pwv: interpolate.interp1d,
) -> tuple[list, list, list, list, list, list, list, list, list, list] | None:
    """
    Function which computes the NET as well as NEP and some other parameters
    for a given observation. Computes these parameters over all available
    wafers and bands. Returns none if there is no metadata for the obs.

    Parameters
    ----------
    obs_ctx : tuple(str, str)
        Tuple of Obs id, ctx_path of the observation
    abscal_list : list
        List of array/band combinations we have abscals for.
    pwv : interpolate-interp1d
        Interpolation function for pwv.

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
    obs_id, ctx_path = obs_ctx
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
            if "ln" in cur_wafer:
                if band == "030":
                    ufm_band = 1
                elif band == "040":
                    ufm_band = 2
            elif "mv" in cur_wafer:
                if band == "090":
                    ufm_band = 1
                elif band == "150":
                    ufm_band = 2
            elif "uv" in cur_wafer:
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
            net_flag = net_flag * (meta.relcal.rel_factor_error < 0.1)

            relcal = meta.relcal.rel_factor[net_flag]

            raw_cal = np.nanmedian(meta.abscal.raw_abscal_rj[net_flag])
            if "noise" in meta.preprocess:
                wnoise = meta.preprocess.noise.white_noise[net_flag]
            elif "noiseT" in meta.preprocess:
                wnoise = meta.preprocess.noiseT.white_noise[net_flag]
            else:
                print(f"Error: no valid noise ken in {meta.preprocess.keys()}")
                continue

            ndet = len(np.where(wnoise != 0)[0])
            net_mes = 1 / np.sqrt(2) * wnoise * raw_cal * relcal
            factor = 1e6
            clean_nets = []
            for net in net_mes:
                if net * factor > 50:
                    clean_nets.append(net)
            clean_nets = np.array(clean_nets)
            array_net = np.nansum((clean_nets * factor) ** (-2)) ** (-1 / 2)

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
    obs_ctx: tuple[str, str],
    pwv: interpolate.interp1d,
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
    obs_ctx : tuple(str, str)
        Tuple of Obs id, ctx_path of the observation
    pwv : interpolate-interp1d
        Interpolation function for pwv.


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
    obs_id, ctx_path = obs_ctx

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
        if "lf" in ctx_path:
            return None
            # det_info = ctx.get_det_info(obs_id, dets={'wafer.wafer_slot': 'ws0'}) #TODO: need per-wafer logic for LF
        else:
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
            except (LoaderError, KeyError):
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
                print(f"Error: no valid noise key in {meta.preprocess.keys()}")
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


def get_all_times(tobs: np.ndarray) -> np.ndarray:
    """
    Get all unique times from the input array.

    Parameters
    ----------
    tobs : np.ndarray
        Array of observation times

    Returns
    -------
    all_times : np.ndarray
        Array of unique times
    """
    all_times = [float(tobs[0][0])]  # initialize with 1 time
    for i, cur_times in enumerate(tobs):
        for cur_time in cur_times:
            isclose = False
            for time in all_times:
                if np.isclose(time, cur_time, rtol=0, atol=300):
                    isclose = True
                    continue
            if not isclose:
                all_times.append(cur_time)

    return np.array(all_times)


def get_matching_stats(
    all_times: np.ndarray,
    unmatched_nets: np.ndarray,
    unmatched_ndets: np.ndarray,
    unmatched_obs_ids: np.ndarray,
    unmatched_tobs: np.ndarray,
    unmatched_pwvs: np.ndarray,
    unmatched_els: np.ndarray,
    temp_conv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get matching statistics for the given times and unmatched data.

    Parameters
    ----------
    all_times : np.ndarray
        Array of unique times
    unmatched_nets : np.ndarray
        Array of unmatched nets
    unmatched_ndets : np.ndarray
        Array of unmatched ndets
    unmatched_obs_ids : np.ndarray
        Array of unmatched observation IDs
    unmatched_tobs : np.ndarray
        Array of unmatched observation times
    unmatched_pwvs : np.ndarray
        Array of unmatched pwvs
    unmatched_els : np.ndarray
        Array of unmatched els
    temp_conv : float
        Temperature conversion factor

    Returns
    -------
    [matched_nets, matched_ndets, matched_obs_ids, matched_t_obs, matched_pwvs, matched_pwvs_sinel] : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple of matching statistics
    """
    nets = [[] for i in range(len(all_times))]
    ndets = [[] for i in range(len(all_times))]
    obs_ids = []
    t_obs = np.zeros(len(all_times))
    pwvs_sinel = np.zeros(len(all_times))
    pwvs = np.zeros(len(all_times))

    for obs_index, time in enumerate(all_times):
        saved_aux_data = False
        for array_index in range(len(unmatched_obs_ids)):
            flag = np.where(np.abs(np.array(unmatched_tobs[array_index]) - time) < 300)[
                0
            ]
            if len(flag) > 1:
                print(obs_index, array_index)
            if len(flag) != 1:
                continue
            flag = flag[0]
            nets[obs_index].append(unmatched_nets[array_index][flag] * temp_conv)
            ndets[obs_index].append(unmatched_ndets[array_index][flag])

            if not saved_aux_data:
                obs_ids.append(str(unmatched_obs_ids[array_index][flag]))
                t_obs[obs_index] = unmatched_tobs[array_index][flag]
                pwvs[obs_index] = unmatched_pwvs[array_index][flag]
                pwvs_sinel[obs_index] = pwvs[obs_index] / np.sin(
                    np.deg2rad(unmatched_els[array_index][flag])
                )
            saved_aux_data = True

    return nets, ndets, np.array(obs_ids), t_obs, pwvs, pwvs_sinel


def comb_by_freq(net_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine net results by frequency.

    Parameters
    ----------
    net_df : pd.DataFrame
        DataFrame containing net results

    Returns
    -------
    df_freq : pd.DataFrame
        DataFrame with combined results by frequency
    """
    # Implementation for combining by frequency
    set_labels = np.unique(net_df.labels.to_numpy()).astype(str)
    tmp_obs_ids_090 = [
        net_df[net_df.labels.str.contains(label)].obs.to_numpy()
        for label in set_labels
        if "090" in label
    ]
    tmp_obs_ids_150 = [
        net_df[net_df.labels.str.contains(label)].obs.to_numpy()
        for label in set_labels
        if "150" in label
    ]
    tmp_obs_ids_220 = [
        net_df[net_df.labels.str.contains(label)].obs.to_numpy()
        for label in set_labels
        if "220" in label
    ]
    tmp_obs_ids_280 = [
        net_df[net_df.labels.str.contains(label)].obs.to_numpy()
        for label in set_labels
        if "280" in label
    ]

    tmp_nets_090 = [
        net_df[net_df.labels.str.contains(label)].nets.to_numpy()
        for label in set_labels
        if "090" in label
    ]
    tmp_nets_150 = [
        net_df[net_df.labels.str.contains(label)].nets.to_numpy()
        for label in set_labels
        if "150" in label
    ]
    tmp_nets_220 = [
        net_df[net_df.labels.str.contains(label)].nets.to_numpy()
        for label in set_labels
        if "220" in label
    ]
    tmp_nets_280 = [
        net_df[net_df.labels.str.contains(label)].nets.to_numpy()
        for label in set_labels
        if "280" in label
    ]

    tmp_ndets_090 = [
        net_df[net_df.labels.str.contains(label)].ndets.to_numpy()
        for label in set_labels
        if "090" in label
    ]
    tmp_ndets_150 = [
        net_df[net_df.labels.str.contains(label)].ndets.to_numpy()
        for label in set_labels
        if "150" in label
    ]
    tmp_ndets_220 = [
        net_df[net_df.labels.str.contains(label)].ndets.to_numpy()
        for label in set_labels
        if "220" in label
    ]
    tmp_ndets_280 = [
        net_df[net_df.labels.str.contains(label)].ndets.to_numpy()
        for label in set_labels
        if "280" in label
    ]

    tmp_tobs_090 = [
        net_df[net_df.labels.str.contains(label)].t_obs.to_numpy()
        for label in set_labels
        if "090" in label
    ]
    tmp_tobs_150 = [
        net_df[net_df.labels.str.contains(label)].t_obs.to_numpy()
        for label in set_labels
        if "150" in label
    ]
    tmp_tobs_220 = [
        net_df[net_df.labels.str.contains(label)].t_obs.to_numpy()
        for label in set_labels
        if "220" in label
    ]
    tmp_tobs_280 = [
        net_df[net_df.labels.str.contains(label)].t_obs.to_numpy()
        for label in set_labels
        if "280" in label
    ]

    tmp_pwv_090 = [
        net_df[net_df.labels.str.contains(label)].pwv.to_numpy()
        for label in set_labels
        if "090" in label
    ]
    tmp_pwv_150 = [
        net_df[net_df.labels.str.contains(label)].pwv.to_numpy()
        for label in set_labels
        if "150" in label
    ]
    tmp_pwv_220 = [
        net_df[net_df.labels.str.contains(label)].pwv.to_numpy()
        for label in set_labels
        if "220" in label
    ]
    tmp_pwv_280 = [
        net_df[net_df.labels.str.contains(label)].pwv.to_numpy()
        for label in set_labels
        if "280" in label
    ]

    tmp_els_090 = [
        net_df[net_df.labels.str.contains(label)].el.to_numpy()
        for label in set_labels
        if "090" in label
    ]
    tmp_els_150 = [
        net_df[net_df.labels.str.contains(label)].el.to_numpy()
        for label in set_labels
        if "150" in label
    ]
    tmp_els_220 = [
        net_df[net_df.labels.str.contains(label)].el.to_numpy()
        for label in set_labels
        if "220" in label
    ]
    tmp_els_280 = [
        net_df[net_df.labels.str.contains(label)].el.to_numpy()
        for label in set_labels
        if "280" in label
    ]

    all_times_090 = get_all_times(tmp_tobs_090)
    all_times_150 = get_all_times(tmp_tobs_150)
    all_times_220 = get_all_times(tmp_tobs_220)
    all_times_280 = get_all_times(tmp_tobs_280)

    temp_conv_090 = temp_conv(
        T_B=2.725 * u.Kelvin, flavor="MF", ch="MF_1", kind="baseline"
    )
    temp_conv_150 = temp_conv(
        T_B=2.725 * u.Kelvin, flavor="MF", ch="MF_2", kind="baseline"
    )
    temp_conv_220 = temp_conv(
        T_B=2.725 * u.Kelvin, flavor="UHF", ch="UHF_1", kind="baseline"
    )
    temp_conv_280 = temp_conv(
        T_B=2.725 * u.Kelvin, flavor="UHF", ch="UHF_2", kind="baseline"
    )

    nets_090, ndets_090, obs_ids_090, t_obs_090, pwvs_sinel_090, pwvs_090 = (
        get_matching_stats(
            all_times=all_times_090,
            unmatched_nets=tmp_nets_090,
            unmatched_ndets=tmp_ndets_090,
            unmatched_obs_ids=tmp_obs_ids_090,
            unmatched_tobs=tmp_tobs_090,
            unmatched_pwvs=tmp_pwv_090,
            unmatched_els=tmp_els_090,
            temp_conv=temp_conv_090,
        )
    )

    nets_150, ndets_150, obs_ids_150, t_obs_150, pwvs_sinel_150, pwvs_150 = (
        get_matching_stats(
            all_times=all_times_150,
            unmatched_nets=tmp_nets_150,
            unmatched_ndets=tmp_ndets_150,
            unmatched_obs_ids=tmp_obs_ids_150,
            unmatched_tobs=tmp_tobs_150,
            unmatched_pwvs=tmp_pwv_150,
            unmatched_els=tmp_els_150,
            temp_conv=temp_conv_150,
        )
    )

    nets_220, ndets_220, obs_ids_220, t_obs_220, pwvs_sinel_220, pwvs_220 = (
        get_matching_stats(
            all_times=all_times_220,
            unmatched_nets=tmp_nets_220,
            unmatched_ndets=tmp_ndets_220,
            unmatched_obs_ids=tmp_obs_ids_220,
            unmatched_tobs=tmp_tobs_220,
            unmatched_pwvs=tmp_pwv_220,
            unmatched_els=tmp_els_220,
            temp_conv=temp_conv_220,
        )
    )

    nets_280, ndets_280, obs_ids_280, t_obs_280, pwvs_sinel_280, pwvs_280 = (
        get_matching_stats(
            all_times=all_times_280,
            unmatched_nets=tmp_nets_280,
            unmatched_ndets=tmp_ndets_280,
            unmatched_obs_ids=tmp_obs_ids_280,
            unmatched_tobs=tmp_tobs_280,
            unmatched_pwvs=tmp_pwv_280,
            unmatched_els=tmp_els_280,
            temp_conv=temp_conv_280,
        )
    )

    nets_090_comb = np.zeros(len(all_times_090))
    for i in range(len(nets_090)):
        nets_090_comb[i] = np.sum(1 / np.array(nets_090[i]) ** 2) ** (-1 / 2)
        ndets_090[i] = np.sum(ndets_090[i])

    nets_150_comb = np.zeros(len(all_times_150))
    for i in range(len(nets_150)):
        nets_150_comb[i] = np.sum(1 / np.array(nets_150[i]) ** 2) ** (-1 / 2)
        ndets_150[i] = np.sum(ndets_150[i])

    nets_220_comb = np.zeros(len(all_times_220))
    for i in range(len(nets_220)):
        nets_220_comb[i] = np.sum(1 / np.array(nets_220[i]) ** 2) ** (-1 / 2)
        ndets_220[i] = np.sum(ndets_220[i])

    nets_280_comb = np.zeros(len(all_times_280))
    for i in range(len(nets_280)):
        nets_280_comb[i] = np.sum(1 / np.array(nets_280[i]) ** 2) ** (-1 / 2)
        ndets_280[i] = np.sum(ndets_280[i])

    nets = np.concatenate([nets_090_comb, nets_150_comb, nets_220_comb, nets_280_comb])
    pwvs = np.concatenate([pwvs_090, pwvs_150, pwvs_220, pwvs_280])
    pwvs_sinel = np.concatenate(
        [pwvs_sinel_090, pwvs_sinel_150, pwvs_sinel_220, pwvs_sinel_280]
    )
    labels = np.concatenate(
        [
            ["f090"] * len(pwvs_090),
            ["f150"] * len(pwvs_150),
            ["f220"] * len(pwvs_220),
            ["f280"] * len(pwvs_280),
        ]
    )
    t_obs = np.concatenate([t_obs_090, t_obs_150, t_obs_220, t_obs_280])
    ndets = np.concatenate([ndets_090, ndets_150, ndets_220, ndets_280])
    obs_ids = np.concatenate([obs_ids_090, obs_ids_150, obs_ids_220, obs_ids_280])

    df_freq = pd.DataFrame(
        {
            "labels": labels,
            "nets": nets,
            "pwv": pwvs,
            "pwvs_sinel": pwvs_sinel,
            "t_obs": t_obs,
            "ndets": ndets,
            "obs_ids": obs_ids,
        }
    )

    return df_freq
