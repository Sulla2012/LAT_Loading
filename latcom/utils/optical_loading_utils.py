"""Helpers for optical_loading.py."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import sotodlib.io.load_book as lb
from sotodlib import core
from sotodlib.io import hkdb

from latcom.utils.common import DATA_DIR, pwv_interp

HKDB_CFG = DATA_DIR / "hkdb-lat.cfg"

# Dict that tracks UXM measurements. Low is the low freq channel, high is the high
UXM_dict = {
    "low": {
        "uv42": {"psat_dark": 28.2, "kappa": 27154, "G": 669, "n": 3.8},
        "uv47": {"psat_dark": 31.4, "kappa": 26600, "G": 780, "n": 3.8},
        "uv31": {"psat_dark": 31.3, "kappa": 31214, "G": 817, "n": 3.8},
        "uv39": {"psat_dark": 22.3, "kappa": 35091, "G": 708, "n": 3.8},
        "uv38": {"psat_dark": 26.9, "kappa": 25104, "G": 668, "n": 3.8},
        "uv46": {"psat_dark": 33.9, "kappa": 26696, "G": 808, "n": 3.8},
        "mv32": {"psat_dark": 3.1, "kappa": 978, "G": 77, "n": 3.0},
        "mv49": None,
        "mv14": {"psat_dark": 2.7, "kappa": 657, "G": 59, "n": 3.0},
        "mv20": {"psat_dark": 3.2, "kappa": 849, "G": 71, "n": 3.0},
        "mv13": {"psat_dark": 2.9, "kappa": 752, "G": 66, "n": 3.0},
        "mv34": {"psat_dark": 2.8, "kappa": 887, "G": 69, "n": 3.0},
        "mv11": {"psat_dark": 3, "kappa": 1004, "G": 80, "n": 3.0},
        "mv25": {"psat_dark": 3.5, "kappa": 944, "G": 78, "n": 3.0},
        "mv26": {"psat_dark": 3.8, "kappa": 1004, "G": 80, "n": 3.0},
        "mv21": {"psat_dark": 3.0, "kappa": 1042, "G": 80, "n": 3.0},
        "mv24": {"psat_dark": 3.7, "kappa": 980, "G": 84, "n": 3.0},
        "mv28": {"psat_dark": 3.7, "kappa": 1004, "G": 86, "n": 3.0},
    },
    "high": {
        "uv42": {"psat_dark": 30.3, "kappa": 34881, "G": 669, "n": 3.9},
        "uv47": {"psat_dark": 33.6, "kappa": 34363, "G": 780, "n": 3.9},
        "uv31": {"psat_dark": 33.3, "kappa": 39978, "G": 817, "n": 3.9},
        "uv39": {"psat_dark": 23.9, "kappa": 45475, "G": 708, "n": 3.9},
        "uv38": {"psat_dark": 28.8, "kappa": 31606, "G": 668, "n": 3.9},
        "uv46": {"psat_dark": 36.4, "kappa": 34126, "G": 808, "n": 3.9},
        "mv32": {"psat_dark": 8.8, "kappa": 8911, "G": 77, "n": 3.7},
        "mv49": None,
        "mv14": {"psat_dark": 6.6, "kappa": 4502, "G": 59, "n": 3.7},
        "mv20": {"psat_dark": 8.7, "kappa": 7327, "G": 71, "n": 3.7},
        "mv13": {"psat_dark": 8.6, "kappa": 6269, "G": 66, "n": 3.7},
        "mv34": {"psat_dark": 8.6, "kappa": 8125, "G": 69, "n": 3.7},
        "mv11": {"psat_dark": 8.1, "kappa": 8257, "G": 80, "n": 3.7},
        "mv25": {"psat_dark": 9.4, "kappa": 8653, "G": 78, "n": 3.7},
        "mv26": {"psat_dark": 10.1, "kappa": 8902, "G": 80, "n": 3.7},
        "mv21": {"psat_dark": 7.5, "kappa": 7920, "G": 80, "n": 3.7},
        "mv24": {"psat_dark": 9.5, "kappa": 8310, "G": 84, "n": 3.7},
        "mv28": {"psat_dark": 9.7, "kappa": 8278, "G": 86, "n": 3.7},
    },
}

# Dict mapping OTs to housekeeping channels for level 3 hk database.
therm_dict = {
    ot: f"cryo-tauhk-1.tauhk_data_full.RTD_OT{n}_100mK_2_temperature"
    for ot, n in {
        "c1": 4,
        "i1": 1,
        "i2": 2,
        "i3": 13,
        "i4": 14,
        "i5": 5,
        "i6": 6,
        "o1": 11,
        "o2": 12,
        "o3": 9,
        "o4": 10,
        "o5": 7,
        "o6": 8,
    }.items()
}


def get_fpa_temps(obs_list: list[OrderedDict]) -> np.ndarray:
    """
    Function that gets UFM temp for obs.
    Gets the UFM therm for each OT from level 3 housekeeping.

    Parameters
    ----------
    obs_list : list[OrderedDict]
        List of obsdb entries to get fpa temps for

    Returns
    -------
    fpa_temps : np.ndarray
       Temperature for each obs
    """
    fpa_temps = np.zeros((len(obs_list),))
    cfg = hkdb.HkConfig.from_yaml(str(HKDB_CFG))
    for o, obs in enumerate(obs_list):
        field = therm_dict[obs["tube_slot"]]
        lspec = hkdb.LoadSpec(
            cfg=cfg, start=obs["start_time"], end=obs["stop_time"], fields=[field]
        )
        result = hkdb.load_hk(lspec, show_pb=False)
        try:
            # tauHK introduces these weird spikes to >1K, not physical. Cut them
            temps = result.data[field][1]
            fpa_temps[o] = np.mean(temps[temps < 0.3])
        except KeyError:
            fpa_temps[o] = np.nan
    return fpa_temps


def add_iv_info(meta: core.AxisManager, ctx: core.Context) -> None:
    """
    Function which adds iv data to a meta data AM.
    Function modifies meta in place, with iv data
    being added to a new axis called iv.

    Parameters
    ----------
    meta : core.AxisManager
        Meta data we wish to add iv data to.
    ctx : core.Context
        Context file corresponding to meta data
    """
    fields = ["p_sat", "R_n", "bgmap"]
    iv = core.AxisManager(meta.dets)
    for f in fields:
        iv.wrap_new(f, ("dets",))
        iv[f] *= np.nan
    iv_data = lb.load_smurf_npy_data(ctx, meta.obs_info.obs_id, "iv")

    for d in range(iv_data["nchans"]):
        idx = np.where(
            (meta.det_info.smurf.band == iv_data["bands"][d])
            & (meta.det_info.smurf.channel == iv_data["channels"][d])
        )[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        iv.bgmap[idx] = iv_data["bgmap"][d]

        if iv.bgmap[idx] not in iv_data["bias_groups"]:
            # iv did not include bias group detector is attached to
            continue

        iv.p_sat[idx] = iv_data["p_sat"][d] * 1e12
        iv.R_n[idx] = iv_data["R_n"][d]
    meta.wrap("iv", iv)


def get_all_iv_data(obs_list: list[OrderedDict], ctx_path: str | Path) -> dict | None:
    """
    Function which gets all iv data from a list of observations of one UFM,
    along with PWVs and UFM temps

    Parameters
    ----------
    obs_list : list[OrderedDict]
        A list of observation dictionaries.
    ctx_path : str | Path
        Path to the context to load the observations with.

    Returns
    -------
    iv_dict : dict | None
        A dictionary containing the IV data, PWVs, and UFM temps for each observation,
        or None if obs_list is empty.
    """
    if len(obs_list) == 0:
        print("obs_list is empty")
        return None
    ufm = obs_list[0]["stream_ids_list"].split("_")[-1]
    print("Starting ", ufm)

    pwv = pwv_interp()
    ctx = core.Context(str(ctx_path))
    iv_dict = {
        key: [] for key in ("psats", "bgmaps", "obs_ids", "pwvs", "ufm_temps", "els")
    }

    for i, obs in enumerate(obs_list):
        meta = ctx.get_meta(obs)
        try:
            add_iv_info(meta, ctx)
        except FileNotFoundError:
            print(meta.obs_info.obs_id)
            continue

        iv_dict["psats"].append(meta.iv["p_sat"])
        iv_dict["bgmaps"].append(meta.iv["bgmap"][...])
        iv_dict["obs_ids"].append(obs["obs_id"])
        iv_dict["pwvs"].append(float(pwv((obs["start_time"] + obs["stop_time"]) / 2)))
        iv_dict["ufm_temps"].append(float(get_fpa_temps(obs_list[i : i + 1])[0]))
        iv_dict["els"].append(meta.obs_info.el_center)

    print("Finished ", ufm)
    return iv_dict
