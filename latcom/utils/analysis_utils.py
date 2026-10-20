"""
Helpers not called by any top-level script, used to analyze their outputs
in the notebooks: parsing NET/NEP/yield results, cut statistics, and bootstraps.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sotodlib.core.axisman import AxisManager

from latcom.utils.common import (
    BANDS,
    cmb_conv,
    get_band_flag,
    get_white_noise,
    pwv_interp,
    ufm_dict,
)

ALL_UFMS = [ufm for ufms in ufm_dict.values() for ufm in ufms]
NDETS_PER_BAND = 860

################################################################
# Parsing results of nets.py / neps.py                         #
################################################################


def _iter_entries(net_dict: dict, freqs: list[str]) -> Iterator[tuple[str, str, dict]]:
    """
    Iterate over the [ufm][freq] entries of a result dict, sorted by freq then ufm.
    Keys of net_dict which are not UFMs (e.g. an index) are skipped.

    Parameters
    ----------
    net_dict : dict
        Result dict organized by [ufm][freq].
    freqs : list[str]
        Freqs to iterate over.

    Yields
    ------
    freq : str
        Frequency of the entry.
    ufm : str
        UFM of the entry.
    entry : dict
        net_dict[ufm][freq].
    """
    ufms = sorted(ufm for ufm in net_dict if ufm in ALL_UFMS)
    for freq in freqs:
        for ufm in ufms:
            if freq in net_dict[ufm]:
                yield freq, ufm, net_dict[ufm][freq]


def _good_net(net: float, ndet: int) -> bool:
    """
    Whether a NET measurement is plausible; very large nets are not real.

    Parameters
    ----------
    net : float
        Array NET.
    ndet : int
        Number of detectors.

    Returns
    -------
    good : bool
        True if the measurement passes cuts.
    """
    return net <= 100 and ndet > 100


def parse_dict(net_dict: dict, cmb: bool = False) -> pd.DataFrame:
    """
    Flatten the output of nets.py into a dataframe with one row per array/band/obs.

    Parameters
    ----------
    net_dict : dict
        Output of nets.py, organized by [ufm][freq].
    cmb : bool, default False
        Convert NETs to uK_CMB rather than uK_RJ.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of NETs, NEPs, yields, etc.
    """
    pwv = pwv_interp()
    rows = []
    for freq, ufm, sub in _iter_entries(net_dict, BANDS):
        conv = cmb_conv(freq) if cmb else 1
        for net, obs, ndet, cal, el in zip(
            sub["nets"], sub["obs"], sub["ndets"], sub["raw_cal"], sub["el"]
        ):
            if not _good_net(net, ndet):
                continue
            t_obs = float(obs.split("_")[1])
            rows.append(
                {
                    "labels": f"{freq}_{ufm}",
                    "nets": net * conv,
                    "obs": obs,
                    "ndets": ndet,
                    "pwv": pwv(t_obs),
                    "neps": net / cal * np.sqrt(2) * np.sqrt(ndet),
                    "cals": cal,
                    "el": el,
                    "yields": ndet / NDETS_PER_BAND,
                    "t_obs": t_obs,
                    "indv_nets": net * np.sqrt(ndet),
                    "ufms": ufm,
                    "freqs": freq,
                }
            )
    return pd.DataFrame(rows)


def parse_neps(net_dict: dict) -> pd.DataFrame:
    """
    Flatten the output of neps.py into a dataframe with one row per detector/obs.

    Parameters
    ----------
    net_dict : dict
        Output of neps.py (or nets.py), organized by [ufm][freq].

    Returns
    -------
    nep_df : pd.DataFrame
        Dataframe of per detector NEPs and NEIs.
    """
    pwv = pwv_interp()
    rows = []
    for freq, ufm, sub in _iter_entries(net_dict, ["090", "150", "220", "280"]):
        for obs, ndet, el, neps, phicals in zip(
            sub["obs"], sub["ndets"], sub["el"], sub["neps"], sub["phiconv"]
        ):
            t_obs = float(obs.split("_")[1])
            cur_pwv = pwv(t_obs)
            for nep, phical in zip(neps, phicals):
                rows.append(
                    {
                        "labels": f"{freq}_{ufm}",
                        "obs": obs,
                        "pwv": cur_pwv,
                        "neps": nep,
                        "neis": nep / phical * 9e6 / (2 * np.pi),
                        "el": el,
                        "pwv_sinel": cur_pwv / np.sin(np.deg2rad(el)),
                        "yields": ndet / NDETS_PER_BAND,
                        "t_obs": t_obs,
                    }
                )
    return pd.DataFrame(rows)


def parse_yield(net_dict: dict) -> pd.DataFrame:
    """
    Parse a result dict containing nets/yields and return a yield dataframe,
    with one row per array/band/obs followed by one row per band/obs combining
    all arrays.

    Parameters
    ----------
    net_dict : dict
        Dictionary with results per obs of NETs/yields, etc.
        Output of nets.py

    Returns
    -------
    df_yield : pd.DataFrame
        Pandas DF with yield information.
    """
    pwv = pwv_interp()
    rows = []
    obs_ids = set()

    def _row(label: str, obs: str, el: float, yld: float) -> dict:
        cur_pwv = pwv(float(obs.split("_")[1]))
        return {
            "labels": label,
            "pwv": cur_pwv,
            "el": el,
            "yields": yld,
            "pwvs_sinel": cur_pwv / np.sin(np.deg2rad(el)),
        }

    for freq, ufm, sub in _iter_entries(net_dict, BANDS):
        for obs, ndet, el, net in zip(sub["obs"], sub["ndets"], sub["el"], sub["nets"]):
            if _good_net(net, ndet):
                rows.append(_row(f"{freq}_{ufm}", obs, el, ndet / NDETS_PER_BAND))
                obs_ids.add(obs)

    times = sorted(float(obs.split("_")[1]) for obs in obs_ids)
    for freq in BANDS:
        for time in times:
            ndets, narrays, last = 0, 0, None
            for _, _, sub in _iter_entries(net_dict, [freq]):
                for obs, ndet, el in zip(sub["obs"], sub["ndets"], sub["el"]):
                    if np.isclose(time, float(obs.split("_")[1])):
                        ndets += ndet
                        narrays += NDETS_PER_BAND
                        last = (obs, el)
            if narrays:
                rows.append(_row(freq, last[0], last[1], ndets / narrays))

    return pd.DataFrame(rows)


def get_all_times(tobs: list[np.ndarray], atol: float = 300) -> np.ndarray:
    """
    Get all unique times (up to atol seconds) from the input arrays.

    Parameters
    ----------
    tobs : list[np.ndarray]
        Arrays of observation times
    atol : float, default 300
        Tolerance in seconds for two times to be the same.

    Returns
    -------
    all_times : np.ndarray
        Array of unique times
    """
    all_times = [float(tobs[0][0])]
    for cur_times in tobs:
        for cur_time in cur_times:
            if not np.any(np.isclose(all_times, cur_time, rtol=0, atol=atol)):
                all_times.append(cur_time)
    return np.array(all_times)


def get_matching_stats(
    all_times: np.ndarray,
    unmatched_nets: list[np.ndarray],
    unmatched_ndets: list[np.ndarray],
    unmatched_obs_ids: list[np.ndarray],
    unmatched_tobs: list[np.ndarray],
    unmatched_pwvs: list[np.ndarray],
    unmatched_els: list[np.ndarray],
    temp_conv: float,
) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match per array statistics to a common set of observation times.

    Parameters
    ----------
    all_times : np.ndarray
        Array of unique times
    unmatched_nets : list[np.ndarray]
        Per array nets
    unmatched_ndets : list[np.ndarray]
        Per array ndets
    unmatched_obs_ids : list[np.ndarray]
        Per array observation IDs
    unmatched_tobs : list[np.ndarray]
        Per array observation times
    unmatched_pwvs : list[np.ndarray]
        Per array pwvs
    unmatched_els : list[np.ndarray]
        Per array els
    temp_conv : float
        Temperature conversion factor applied to the nets

    Returns
    -------
    nets : list
        For each time, the nets of each matching array.
    ndets : list
        For each time, the ndets of each matching array.
    obs_ids : np.ndarray
        Obs id of each time.
    t_obs : np.ndarray
        Obs time of each time.
    pwvs : np.ndarray
        PWV of each time.
    pwvs_sinel : np.ndarray
        PWV / sin(el) of each time.
    """
    n = len(all_times)
    nets = [[] for _ in range(n)]
    ndets = [[] for _ in range(n)]
    obs_ids = []
    t_obs, pwvs, pwvs_sinel = np.zeros(n), np.zeros(n), np.zeros(n)

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
    Combine per array NET results into per band results by inverse variance
    combining the arrays observing at the same time.

    Parameters
    ----------
    net_df : pd.DataFrame
        DataFrame containing net results, as from parse_dict

    Returns
    -------
    df_freq : pd.DataFrame
        DataFrame with combined results by frequency
    """
    set_labels = np.unique(net_df.labels.to_numpy()).astype(str)
    dfs = []
    for freq in ["090", "150", "220", "280"]:
        subs = [
            net_df[net_df.labels.str.contains(label)]
            for label in set_labels
            if freq in label
        ]
        cols = {
            col: [sub[col].to_numpy() for sub in subs]
            for col in ("obs", "nets", "ndets", "t_obs", "pwv", "el")
        }
        all_times = get_all_times(cols["t_obs"])
        nets, ndets, obs_ids, t_obs, pwvs, pwvs_sinel = get_matching_stats(
            all_times=all_times,
            unmatched_nets=cols["nets"],
            unmatched_ndets=cols["ndets"],
            unmatched_obs_ids=cols["obs"],
            unmatched_tobs=cols["t_obs"],
            unmatched_pwvs=cols["pwv"],
            unmatched_els=cols["el"],
            temp_conv=cmb_conv(freq),
        )
        dfs.append(
            pd.DataFrame(
                {
                    "labels": f"f{freq}",
                    "nets": [np.sum(1 / np.array(n) ** 2) ** (-1 / 2) for n in nets],
                    "pwv": pwvs,
                    "pwvs_sinel": pwvs_sinel,
                    "t_obs": t_obs,
                    "ndets": [np.sum(n) for n in ndets],
                    "obs_ids": obs_ids,
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


def get_matching_obs(obs_id_list: list[list[str]], atol: float = 300) -> np.ndarray:
    """
    Given a list of lists of obs ids, find the set of unique observation times
    and which obs in each list correspond to each time.

    Parameters
    ----------
    obs_id_list : list[list[str]]
        List of obs ID lists, e.g. one per array.
    atol : float, default 300
        Time tolerance, in seconds, for observations to be considered the same

    Returns
    -------
    array_matched_times : np.ndarray
        (len(obs_id_list), n_times) array of indices into each obs list,
        -1 where a list has no obs (within 40 s) at that time.
    """

    def _t(obs_id: str) -> float:
        return float(obs_id.split("_")[1])

    all_times = get_all_times([[_t(ob) for ob in obs] for obs in obs_id_list], atol)
    array_matched_times = -1 * np.ones((len(obs_id_list), len(all_times)), dtype=int)
    for i, obs_ids in enumerate(obs_id_list):
        for j, time in enumerate(all_times):
            for k, obs in enumerate(obs_ids):
                if np.isclose(_t(obs), time, rtol=0, atol=40):
                    array_matched_times[i, j] = k
    return array_matched_times


################################################################
# Bootstraps                                                   #
################################################################


def bootstrap(cals: np.ndarray, samps: int = 1000) -> np.ndarray:
    """
    Bootstrap the mean of cals.

    Parameters
    ----------
    cals : np.ndarray
        Data to bootstrap.
    samps : int, default 1000
        Number of bootstrap samples.

    Returns
    -------
    samples : np.ndarray
        Mean of each bootstrap sample.
    """
    rng = np.random.default_rng()
    return rng.choice(cals, size=(samps, len(cals))).mean(axis=1)


def bootstrap_err(
    cals: np.ndarray, q: list[float] | None = None, samps: int = 1000
) -> np.ndarray:
    """
    Get bootstrapped percentiles of the mean of cals.

    Parameters
    ----------
    cals : np.ndarray
        Data to bootstrap.
    q : list[float] | None, default None
        Percentiles to compute, defaults to +/- 1 sigma [15.9, 84.1].
    samps : int, default 1000
        Number of bootstrap samples.

    Returns
    -------
    percentiles : np.ndarray
        Percentiles of the bootstrapped mean.
    """
    if q is None:
        q = [15.9, 84.1]
    return np.percentile(bootstrap(cals, samps), q)


################################################################
# Yield and cut statistics                                     #
################################################################


def get_wnoise_yield(meta: AxisManager, ufm: str, band: str) -> int:
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
    ndets : int
        Number of dets with non-zero white noise estimates.

    Raises
    ------
    ValueError
        If there are no dets, the band is invalid for the UFM,
        or no valid noise key is found in the metadata.
    """
    flag = get_band_flag(meta, ufm, band)
    if len(flag) == 0:
        raise ValueError(f"No det info for obs {meta.obs_info.obs_id}.")
    return int(np.count_nonzero(get_white_noise(meta, flag)))


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
    ndets : list[int]
        The number of dets at each stage.
    stage_name : list[str]
        The stages, starting with "initial".
    """
    ndet = []
    stage_name = ["initial"]
    for key in meta.preprocess:
        try:
            if "valid" in meta.preprocess[key]:
                valid = np.all(meta.preprocess[key].valid.mask(), axis=-1)
                ndet.append(int(np.count_nonzero(valid)))
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
) -> tuple[np.ndarray, list[str]]:
    """
    Get the number of cuts from each source
    in a preprocessing metadata. Note this assumes
    that the preprocessing steps were applied in
    the same order as they appear in the metadata.

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
    ncut : np.ndarray
        The number of dets cut at each stage.
    cut_name : list[str]
        The corresponding stage names.
    """
    ndet, stage_name = get_yield_stages(meta, ufm, band)
    return -1 * np.diff(ndet), stage_name[1:]


def get_det_bias_cuts(meta: AxisManager) -> dict[str, float]:
    """
    Get the fraction of dets cut by bad bias, by reason.

    Parameters
    ----------
    meta : AxisManager
        Metadata which contains preprocess.det_bias_flags

    Returns
    -------
    cut_sources : dict[str, float]
        The fractional number of cuts and their corresponding names.
    """
    flags = meta.preprocess.det_bias_flags

    def n_cut(key: str) -> int:
        return int(np.count_nonzero(np.any(flags[key].mask(), axis=-1)))

    keys = ["bg_flags", "r_tes_flags", "r_frac_gt_flags", "r_frac_lt_flags"]
    valid = n_cut("valid")
    if valid == 0:
        return dict.fromkeys([*keys, "r_frac_nan_flags"], 0)

    nan_cuts = int(
        np.count_nonzero(
            flags["r_frac_gt_flags"].mask().T[0] & flags["r_frac_lt_flags"].mask().T[0]
        )
    )
    cut_sources = {
        key: (n_cut(key) - (nan_cuts if "r_frac" in key else 0)) / valid for key in keys
    }
    cut_sources["r_frac_nan_flags"] = nan_cuts / valid
    return cut_sources


################################################################
# Reference yields                                             #
################################################################

aso_toby_yields = {
    "090_mv11": 329 / 860,
    "090_mv13": 406 / 860,
    "090_mv14": 301.56 / 860,
    "090_mv15r2": 541.2989130434783 / 860,
    "090_mv20": 218.12 / 860,
    "090_mv21": 464.128 / 860,
    "090_mv24": 705.1230158730159 / 860,
    "090_mv25": 672.4333333333333 / 860,
    "090_mv26": 522.9151291512915 / 860,
    "090_mv28": 621.1854838709677 / 860,
    "090_mv29": 572.3918918918919 / 860,
    "090_mv32": 539.5990099009902 / 860,
    "090_mv34": 239.39086294416245 / 860,
    "090_mv49": 492.3690476190476 / 860,
    "090_mv63": 621.4718309859155 / 860,
    "090_mv64": 488.7892720306513 / 860,
    "090_mv65": 525.1842105263158 / 860,
    "090_mv67": 610.788679245283 / 860,
    "090_mv68": 705.1143911439115 / 860,
    "090_mv70": 392.6513409961686 / 860,
    "090_mv73": 368.1 / 860,
    "090_mv75": 723.29296875 / 860,
    "090_mv76": 619.1269841269841 / 860,
    "090_mv77": 514.319391634981 / 860,
    "150_mv11": 418.77186311787074 / 860,
    "150_mv13": 526.8343949044586 / 860,
    "150_mv14": 359.5532994923858 / 860,
    "150_mv15r2": 617.5326086956521 / 860,
    "150_mv20": 275.43055555555554 / 860,
    "150_mv21": 531.18 / 860,
    "150_mv24": 705.1230158730159 / 860,
    "150_mv25": 694.2 / 860,
    "150_mv26": 213.12546125461256 / 860,
    "150_mv28": 476.2258064516129 / 860,
    "150_mv29": 643.3333333333334 / 860,
    "150_mv32": 609.6534653465346 / 860,
    "150_mv34": 512.0812182741116 / 860,
    "150_mv49": 587.8452380952381 / 860,
    "150_mv63": 700.4119718309859 / 860,
    "150_mv64": 683.3716475095786 / 860,
    "150_mv65": 621.2255639097745 / 860,
    "150_mv67": 607.8490566037735 / 860,
    "150_mv68": 725.4022140221402 / 860,
    "150_mv70": 577.8659003831417 / 860,
    "150_mv73": 428.0 / 860,
    "150_mv75": 709.8359375 / 860,
    "150_mv76": 587.0119047619048 / 860,
    "150_mv77": 630.1444866920152 / 860,
    "220_uv31": 684.9250936329588 / 860,
    "220_uv38": 660.6954887218045 / 860,
    "220_uv39": 623.455223880597 / 860,
    "220_uv42": 645.4659498207885 / 860,
    "220_uv46": 689.4028268551236 / 860,
    "220_uv47": 707.0153846153846 / 860,
    "220_uv54": 605.1223776223776 / 860,
    "220_uv57": 394.23790322580646 / 860,
    "220_uv58": 569.4181184668989 / 860,
    "220_uv59": 389.85887096774195 / 860,
    "220_uv62": 571.7391304347826 / 860,
    "280_uv31": 665.0636704119851 / 860,
    "280_uv38": 447.9248120300752 / 860,
    "280_uv39": 477.7014925373134 / 860,
    "280_uv42": 439.3763440860215 / 860,
    "280_uv46": 594.1236749116608 / 860,
    "280_uv47": 652.1192307692307 / 860,
    "280_uv54": 455.65034965034965 / 860,
    "280_uv57": 315.89516129032256 / 860,
    "280_uv58": 381.77351916376307 / 860,
    "280_uv59": 389.1774193548387 / 860,
    "280_uv62": 365.30434782608694 / 860,
    "030_ln": 195.72627737226279 / 354,
    "040_ln": 111.56934306569343 / 354,
}
aso_toby_df = pd.DataFrame(
    {"labels": aso_toby_yields.keys(), "yields": aso_toby_yields.values()}
)

# Pton yields are the same for both bands of a UFM
_ptown_ufm_yields = {
    "mv11": 0.72,
    "mv13": 0.75,
    "mv14": 0.88,
    "mv20": 0.90,
    "mv21": 0.79,
    "mv24": 0.85,
    "mv25": 0.90,
    "mv26": 0.89,
    "mv28": 0.81,
    "mv32": 0.88,
    "mv34": 0.72,
    "mv49": np.nan,
    "uv31": 0.81,
    "uv38": 0.71,
    "uv39": 0.87,
    "uv42": 0.81,
    "uv46": 0.82,
    "uv47": 0.82,
}
ptown_yields = {
    f"{band}_{ufm}": yld
    for band in ("090", "220", "150", "280")
    for ufm, yld in _ptown_ufm_yields.items()
    if ufm[:2] == ("mv" if band in ("090", "150") else "uv")
}
ptown_df = pd.DataFrame(
    {"labels": ptown_yields.keys(), "yields": ptown_yields.values()}
)
