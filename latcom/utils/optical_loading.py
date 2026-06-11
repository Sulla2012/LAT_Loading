from functools import cache
from os.path import join

import h5py
import numpy as np
import pandas as pd
import sotodlib.coords.det_match as dm
import sotodlib.io.load_book as lb
from scipy import interpolate
from so3g.hk import load_range
from sotodlib import core
from sotodlib.io import hkdb
from sotodlib.io.ancil.pwv import apex_to_tocolin_250701

# Dict mapping OTs to UFMs
ufm_dict = {
    "c1": ["uv38", "uv39", "uv46"],
    "i1": ["mv21", "mv24", "mv28"],
    "i2": ["uv54", "uv58", "uv60"],
    "i3": ["mv13", "mv20", "mv34"],
    "i4": ["mv14", "mv32", "mv49"],
    "i5": ["uv31", "uv42", "uv47"],
    "i6": ["mv11", "mv25", "mv26"],
    "o1": ["uv57", "uv59", "uv62"],
    "o2": ["mv29", "mv68", "mv73"],
    "o3": ["mv65", "mv67", "mv75"],
    "o4": ["mv15r2", "mv64", "mv70"],
    "o5": ["mv63", "mv76", "mv77"],
    "o6": ["ln2", "ln3", "ln4"],
}


def ot_from_ufm(ufm: str) -> str:
    """
    Convert from ufm to the OT the ufm is in.

    Parameters
    ----------
    ufm : str
        UFM of interest.

    Returns
    -------
    ot : str
        OT the UFM is in

    Raises
    ------
    ValueError
        If no OT is found for the UFM.
    """

    for ot, ufms in ufm_dict:
        if ufm in ufms:
            return ot
    raise ValueError(f"Error: no OT found for UFM {ufm}")


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

# Dict mapping OTs to house keeping channels for level 2 hk data base.
# Eepreciated in favor of level 3 hk data base.
_therm_dict = {
    "c1": "lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_03_T",
    "i1": "lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_15_T",
    "i3": "lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_09_T",
    "i4": "lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_11_T",
    "i5": "lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_01_T",
    "i6": "lat.cryo-ls372-lsa22vr.feeds.temperatures.Channel_14_T",
}

# Dict mapping OTs to housekeeping channels for level 3 hk database.
therm_dict = {
    "c1": "cryo-ls372-lsa22vr.temperatures.Channel_03_T",
    "i1": "cryo-ls372-lsa22vr.temperatures.Channel_15_T",
    "i3": "cryo-ls372-lsa22vr.temperatures.Channel_09_T",
    "i4": "cryo-ls372-lsa22vr.temperatures.Channel_11_T",
    "i5": "cryo-ls372-lsa22vr.temperatures.Channel_01_T",
    "i6": "cryo-ls372-lsa22vr.temperatures.Channel_14_T",
}


def keys_from_wafer(wafer: str, band: str):
    if "mv" in wafer:
        ufm_type = "MF"
        if band == "090":
            ufm_band = "MF_1"
        elif band == "150":
            ufm_band = "MF_2"
        else:
            raise ValueError(f"Error: bad band {band} for ufm {wafer}")
    elif "uv" in wafer:
        ufm_type = "UHF"
        if band == "220":
            ufm_band = "UHF_1"
        elif band == "280":
            ufm_band = "UHF_2"
        else:
            raise ValueError(f"Error: bad band {band} for ufm {wafer}")
    elif "lv" in wafer:
        ufm_type = "LF"
        if band == "030":
            ufm_band = "LF_1"
        elif band == "040":
            ufm_band = "LF_2"
        else:
            raise ValueError(f"Error: bad band {band} for ufm {wafer}")

    else:
        raise ValueError(f"Error: bad ufm {wafer}.")

    return ufm_type, ufm_band


def pwv_interp(
    filepath: str = "/global/u2/j/jorlo/dev/LAT_Loading/latcom/utils/apex_pwv_data.npz",
    time_cut: float = 17410 * 1e5,
) -> interpolate.interp1d:
    """
    Interpolates APEX pwv data. Should be replaced by SO radiometer when it becomes available

    Parameters
    ----------
    filepath : str
        Path to apex pwv data
    time cut : float, Default = 17410*1e5
        Time in unix below which to exclude data.

    Returns
    -------
    pwv : scipy.interpolate.interp1d
        Interp object of pwv vs unix time
    """
    data = {}
    with np.load(filepath, allow_pickle=True) as x:
        for k in x:
            data[k] = x[k]

    flags = np.where(data["timestamp"] >= time_cut)[0]

    for key in data:
        data[key] = data[key][flags]

    data["pwv"] = apex_to_tocolin_250701(data["pwv"])  # APEX to CLASS best fit from Max

    pwv = interpolate.interp1d(data["timestamp"], data["pwv"])
    return pwv


@cache
def bandpass_interp(
    band: str, ufm: str, path: str = "/so/home/jorlo/data/lat_bandpasses/"
) -> interpolate.interp1d:
    if band == "090" or band == "150":
        df = pd.read_csv(path + "LAT_MF_bands.csv")
    elif band == "220" or band == "280":
        df = pd.read_csv(path + "LAT_UHF_bands.csv")
    else:
        raise ValueError(f"ERROR: band {band} not valid")

    x = df["frequency"].to_numpy()
    if str(ufm + "_f" + band) in df:
        y = df[str(ufm + "_f" + band)].to_numpy()

    else:
        ys = []
        for key in df:
            if str(band) in key:
                ys.append(df[key])
        ys = np.array(ys)
        y = np.mean(ys, axis=0)

    return interpolate.interp1d(x, y, bounds_error=False, fill_value=0)


@cache
def get_bandwidth(band: str, ufm: str, path: str = "./bands") -> float:
    """
    Function which gets the site measured bandwidth for a given ufm and band.
    If no data is available for the particular ufm of interest, the average
    across all ufm's is used.

    Parameters
    ----------
    band : str
        String identifying the band of interest
    ufm : str
        UFM of interest
    path : str
        Path to the bandpass files

    Returns
    -------
    bandwidth : float
        The bandwidth, in GHz

    Raises
    ------
    ValueError
        If invalid band is passed.
    """
    if band == "090" or band == "150":
        df = pd.read_csv(path + "LAT_MF_bands.csv")
    elif band == "220" or band == "280":
        df = pd.read_csv(path + "LAT_UHF_bands.csv")
    else:
        raise ValueError(f"ERROR: band {band} not valid")

    x = np.linspace(20, 375, 10000)
    if str(ufm + "_f" + band) in df:
        band = bandpass_interp(band=band, ufm=ufm, path=path)
        bandwidth = np.trapz(band(x), x)

    else:
        arrays = [key.split("_")[0] for key in df if key != "frequency"]
        passes = np.zeros(len(arrays))
        for i, array in enumerate(arrays):
            bandpass = bandpass_interp(band, array, path=path)
            x = np.linspace(50, 350, 10000)
            y = bandpass(x)
            passes[i] = np.trapezoid(y, x)
        bandwidth = np.mean(passes)
    return bandwidth


def get_fpa_temps(obs_list: list[core.axisman.AxisManager]) -> np.array:
    """
    Function that gets UFM temp for obs.
    Gets the UFM therm for each OT from level 3 housekeeping.
    Parameters
    ----------
    obs_list : list[AxisManager]
        List of observations to get fpa temps for

    Returns
    -------
    fpa_temps : np.array
       Temperatue for each obs
    """
    fpa_temps = np.zeros((len(obs_list),))
    cfg = hkdb.HkConfig.from_yaml("/so/home/jorlo/dev/LAT_analysis/hkdb-lat.cfg")
    for o, obs in enumerate(obs_list):
        field = therm_dict[obs["tube_slot"]]
        lspec = hkdb.LoadSpec(
            cfg=cfg,
            start=obs["start_time"],
            end=obs["stop_time"],
            fields=[field],
        )
        result = hkdb.load_hk(lspec, show_pb=False)
        try:
            fpa_temps[o] = np.mean(result.data[field][1])
        except KeyError:
            fpa_temps[o] = np.nan
    return fpa_temps


def _get_fpa_temps(obs_list: list[core.axisman.AxisManager]) -> np.array:
    """
    Function that gets UFM temp for obs.
    Gets the UFM therm for each OT from level 2 housekeeping.
    Depreciated in favor of level 2 housekeeping.
    Kept around as sometimes accessing level 2 is necessary.

    Parameters
    ----------
    obs_list : list[AxisManager]
        List of observations to get fpa temps for

    Returns
    -------
    fpa_temps : np.array
        Temperatue for each obs
    """
    fpa_temps = np.zeros((len(obs_list),))
    for o, obs in enumerate(obs_list):
        field = _therm_dict[obs["tube_slot"]]
        data = load_range(
            obs["start_time"],
            obs["stop_time"],
            fields=[field],
            alias=["fpa_temp"],
            data_dir="/so/level2-daq/lat/hk/",
        )
        try:
            fpa_temps[o] = np.mean(data["fpa_temp"][1])
        except KeyError:
            fpa_temps[o] = np.nan


def add_iv_info(meta: core.axisman.AxisManager, ctx: core.Context):
    """
    Function which adds iv data to a meta data AM.
    Function modifies meta in place, with iv data
    being added to a new axis called iv.

    Parameters
    ----------
    meta : core.axisman.AxisManager
        Meta data we wish to add iv data to.
    ctx : core.Context
        Context file corresponding to meta data

    Returns
    -------
    none
    """
    fields = ["p_sat", "R_n", "bgmap"]
    iv = core.AxisManager(meta.dets)

    for f in fields:
        iv.wrap_new(f, ("dets",))
        iv[f] *= np.nan
    iv_data = lb.load_smurf_npy_data(ctx, meta.obs_info.obs_id, "iv")

    for d in range(iv_data["nchans"]):
        idx = np.where(
            np.all(
                [
                    meta.det_info.smurf.band == iv_data["bands"][d],
                    meta.det_info.smurf.channel == iv_data["channels"][d],
                ],
                axis=0,
            )
        )[0]
        if len(idx) == 0:
            print(f"Cannot find ({iv_data['bands'][d]},{iv_data['channels'][d]})")
            continue
        idx = idx[0]
        iv.bgmap[idx] = iv_data["bgmap"][d]

        if iv.bgmap[idx] not in iv_data["bias_groups"]:
            ## iv did not include bias group detector is attached to
            continue

        iv.p_sat[idx] = iv_data["p_sat"][d] * 1e12
        iv.R_n[idx] = iv_data["R_n"][d]
    meta.wrap("iv", iv)


def get_obs_biases(iva: dict) -> dict:
    """
    This function is exactly how we currently choose biases from IV's.

    Do not change these parameters or you will not get the correct chosen bias!
    """

    rfrac_range = (0.3, 0.6)
    bias_groups = iva["bias_groups"]

    biases = {}
    Rfrac = (iva["R"].T / iva["R_n"]).T
    in_range = (rfrac_range[0] < Rfrac) & (Rfrac < rfrac_range[1])
    Rn_range = (5e-3, 12e-3)

    for bg in bias_groups:
        m = iva["bgmap"] == bg
        m = m & (Rn_range[0] < iva["R_n"]) & (iva["R_n"] < Rn_range[1])

        if not m.any():
            continue

        nchans_in_range = np.sum(in_range[m, :], axis=0)
        target_idx = np.nanargmax(nchans_in_range)
        biases[bg] = iva["v_bias"][target_idx]

    return biases


def get_dark_cal(stream_id: str, platform: str) -> np.array:
    """
    THIS FUNCTION CURRENTLY REFERENCES FILES THAT ONLY EXIST FOR THE SAT
    """
    dark_cal_dir = "/so/home/mrandall/Analysis/IVs/Dark_Cal"
    dark_cal_file = f"{platform}_dark_cal.h5"
    with h5py.File(join(dark_cal_dir, dark_cal_file), "r") as f:
        return np.array(f[stream_id])


def get_dark_rset(stream_id: str, ot: str) -> dm.ResSet:
    data = get_dark_cal(stream_id, ot)

    north_is_highband = dm.get_north_is_highband(data["band"], data["bg"])
    resonances = []
    for idx, x in enumerate(data):
        is_north = north_is_highband ^ (x["band"] < 4)
        res = dm.Resonator(
            idx=idx,
            smurf_band=x["band"],
            smurf_channel=x["channel"],
            bg=x["bg"],
            res_freq=x["freq"],
            is_north=is_north,
            is_optical=~x["masked"],
        )
        if res.res_freq >= 6000:
            res.res_freq -= 2000
        resonances.append(res)
    rset = dm.ResSet(resonances)
    rset.name = f"{stream_id} pton"
    return rset


def get_obs_from_obs_ids(
    obs_ids: int, ot: str
) -> list[core.metadata.resultset.ResultSet]:
    ctx = core.Context("/so/metadata/lat/contexts/smurf_detcal.yaml")

    obs_list = []
    for obs_id in obs_ids:  # TODO: why is this a loop and not just a querry?
        obs = ctx.obsdb.query(
            f"(obs_id=='{obs_id}') and (type=='oper') and (subtype=='iv') and tube_slot == '{ot}'"
        )
        obs_list.append(obs[0])

    return obs_list
