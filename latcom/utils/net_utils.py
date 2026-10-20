"""Helpers for nets.py and neps.py."""

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy import interpolate
from sotodlib import core
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.tod_ops.flags import get_det_bias_flags

from latcom.utils.common import (
    CTX_DIR,
    bands_for_ufm,
    get_band_flag,
    get_white_noise,
    lf_tubes,
    ufm_dict,
)

NEP_FIELDS = ("obs", "ndets", "el", "pwv", "neps", "phiconv")
NET_FIELDS = ("obs", "ndets", "nets", "raw_cal", "el", "pwv", "neps", "phiconv")

# Preprocess config to use for obs before / after the switch date
PREPROC_SWITCH = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
PREPROC_OLD = CTX_DIR / "preprocess_nominal_260616.yaml"
PREPROC_NEW = CTX_DIR / "preprocess_260604.yaml"


def get_obs_ctx_list(
    ctx: core.Context, start: dt.datetime, end: dt.datetime
) -> list[tuple[str, str]]:
    """
    Query the CMB observations between start and end and pair each with
    the preprocess context it should be loaded with. LF tubes are skipped.

    Parameters
    ----------
    ctx : core.Context
        Context with the obsdb to query.
    start : dt.datetime
        Start time.
    end : dt.datetime
        End time.

    Returns
    -------
    obs_ctx_list : list[tuple[str, str]]
        List of (obs_id, ctx_path).
    """
    obs_list = ctx.obsdb.query(
        f"{end.timestamp()} > timestamp and timestamp > {start.timestamp()}"
        " and type=='obs' and subtype=='cmb'"
    )
    obs_ctx_list = []
    for obs in obs_list:
        obs_id = str(obs["obs_id"])
        _, ctime, tel_tube = obs_id.split("_")[:3]
        if tel_tube[3:] in lf_tubes:
            continue  # Skip LF tubes for now
        timestamp = dt.datetime.fromtimestamp(int(ctime), tz=dt.timezone.utc)
        ctx_path = PREPROC_OLD if timestamp < PREPROC_SWITCH else PREPROC_NEW
        obs_ctx_list.append((obs_id, str(ctx_path)))
    return obs_ctx_list


def _iter_wafer_band_meta(
    ctx: core.Context, obs_id: str, ufms: list[str] | None = None
) -> Iterator[tuple[str, str, core.AxisManager, np.ndarray]]:
    """
    Load the bias-flag-cut metadata of an observation for each wafer and band.

    Parameters
    ----------
    ctx : core.Context
        Context to load metadata from.
    obs_id : str
        Observation to load.
    ufms : list[str] | None, default None
        If given, only wafers in this list are loaded.

    Yields
    ------
    ufm : str
        Wafer of the metadata.
    band : str
        Band of the metadata.
    meta : core.AxisManager
        Metadata restricted to the wafer/band and det bias cuts.
    flag : np.ndarray
        Detector mask for the wafer/band.
    """
    try:  # Much faster than ctx.get_meta
        det_info = ctx.get_det_info(obs_id)
    except LoaderError:
        print(f"No meta data for obs {obs_id}")
        return

    for stream_id in np.unique(det_info["stream_id"]):
        ufm = stream_id.split("_")[-1]
        if ufms is not None and ufm not in ufms:
            print(f"No abscal for ufm {ufm}")
            continue
        for band in bands_for_ufm(ufm):
            try:
                meta = ctx.get_meta(
                    obs_id,
                    dets={
                        "dets:stream_id": f"ufm_{ufm}",
                        "dets:wafer.bandpass": f"f{band}",
                    },
                )
            except (LoaderError, KeyError):
                print(f"No meta data for obs {obs_id}")
                continue

            flags = get_det_bias_flags(meta).det_bias_flags
            meta.restrict("dets", ~core.flagman.has_any_cuts(flags))
            flag = get_band_flag(meta, ufm, band)
            if len(flag) == 0:
                print(f"No det_info for obs {obs_id}")
                continue
            yield ufm, band, meta, flag


def get_nets(
    obs_ctx: tuple[str, str],
    abscal_list: list,
    pwv: interpolate.interp1d,
) -> list[dict] | None:
    """
    Function which computes the NET as well as NEP and some other parameters
    for a given observation. Computes these parameters over all available
    wafers and bands. Returns none if there is no metadata for the obs.

    Parameters
    ----------
    obs_ctx : tuple(str, str)
        Tuple of Obs id, ctx_path of the observation
    abscal_list : list
        List of arrays we have abscals for.
    pwv : interpolate.interp1d
        Interpolation function for pwv.

    Returns
    -------
    results : list[dict] | None
        One dict per wafer/band with keys "ufm", "band" and NET_FIELDS:
        obs id, number of functioning dets, inverse variance averaged array NET,
        raw abscal, central elevation, pwv, per det NEPs, and phase to pW conversions.
    """
    obs_id, ctx_path = obs_ctx
    ctx = core.Context(ctx_path)
    results = []
    for ufm, band, meta, flag in _iter_wafer_band_meta(ctx, obs_id, abscal_list):
        flag = flag * (meta.relcal.rel_factor_error < 0.1)
        try:
            wnoise = get_white_noise(meta, flag)
        except ValueError as e:
            print(e)
            continue

        raw_cal = np.nanmedian(meta.abscal.raw_abscal_rj[flag])
        net_mes = 1e6 / np.sqrt(2) * wnoise * raw_cal * meta.relcal.rel_factor[flag]
        clean_nets = net_mes[net_mes > 50]
        results.append(
            {
                "ufm": ufm,
                "band": band,
                "obs": obs_id,
                "ndets": int(np.count_nonzero(wnoise)),
                "nets": np.nansum(clean_nets ** (-2)) ** (-1 / 2),
                "raw_cal": raw_cal,
                "el": meta.obs_info.el_center,
                "pwv": pwv(float(obs_id.split("_")[1])),
                "neps": wnoise,
                "phiconv": meta.det_cal.phase_to_pW[flag],
            }
        )
    return results or None


def get_neps(
    obs_ctx: tuple[str, str],
    pwv: interpolate.interp1d,
) -> list[dict] | None:
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
    pwv : interpolate.interp1d
        Interpolation function for pwv.

    Returns
    -------
    results : list[dict] | None
        One dict per wafer/band with keys "ufm", "band" and NEP_FIELDS:
        obs id, number of functioning dets, central elevation, pwv,
        per det NEPs, and phase to pW conversions.
    """
    obs_id, ctx_path = obs_ctx
    if "lf" in Path(ctx_path).name:
        return None  # TODO: need per-wafer logic for LF
    ctx = core.Context(ctx_path)
    results = []
    for ufm, band, meta, flag in _iter_wafer_band_meta(ctx, obs_id):
        try:
            wnoise = get_white_noise(meta, flag)
        except ValueError as e:
            print(e)
            continue
        results.append(
            {
                "ufm": ufm,
                "band": band,
                "obs": obs_id,
                "ndets": int(np.count_nonzero(wnoise)),
                "el": meta.obs_info.el_center,
                "pwv": pwv(float(obs_id.split("_")[1])),
                "neps": wnoise,
                "phiconv": meta.det_cal.phase_to_pW[flag],
            }
        )
    return results or None


def parse_results(results: list[list[dict] | None], fields: tuple[str, ...]) -> dict:
    """
    Parse the per obs results of get_nets/get_neps into a dictionary
    organized by [ufm][band][field]. get_nets/get_neps are run in
    parallel, so they return per obs lists instead of the desired dict.

    Parameters
    ----------
    results : list[list[dict] | None]
        List of results by obs as produced by get_nets or get_neps.
    fields : tuple[str, ...]
        Fields to collect, i.e. NET_FIELDS or NEP_FIELDS.

    Returns
    -------
    result_dict : dict
        Results reorganized by [ufm][band][field].
    """
    result_dict = {
        ufm: {band: {field: [] for field in fields} for band in bands_for_ufm(ufm)}
        for ufms in ufm_dict.values()
        for ufm in ufms
    }
    for obs_results in results:
        for res in obs_results or []:
            entry = result_dict[res["ufm"]][res["band"]]
            for field in fields:
                entry[field].append(res[field])
    return result_dict
