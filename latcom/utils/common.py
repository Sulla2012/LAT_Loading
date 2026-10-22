"""Constants and helpers shared by more than one top-level latcom script."""

import argparse as ap
import datetime as dt
import os
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

import astropy.constants as const
import astropy.units as u
import dill as pk
import numpy as np
import yaml
from scipy import interpolate
from sotodlib import core
from sotodlib.io.ancil.pwv import apex_to_tocolin_250701

REPO_DIR = Path(__file__).resolve().parents[2]
CTX_DIR = REPO_DIR / "ctxs"
DATA_DIR = REPO_DIR / "data"
APEX_PWV_PATH = Path(__file__).resolve().parent / "apex_pwv_data.npz"
JBOLO_MODELS_PATH = "/global/u2/j/jorlo/dev/bolocalc-so-model"

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

so_nominal_tubes = ["c1", "i1", "i3", "i4", "i5", "i6"]
aso_tubes = ["i2", "o1", "o2", "o3", "o4", "o5"]
lf_tubes = ["o6"]

# Bands (low, high) for each UFM type, keyed by the UFM name prefix
UFM_BANDS = {"ln": ("f030", "f040"), "mv": ("f090", "f150"), "uv": ("f220", "f280")}
BANDS = [band for pair in UFM_BANDS.values() for band in pair]

# Band to bolocalc channel name
flavor_dict = {
    "f030": "LF_1",
    "f040": "LF_2",
    "f090": "MF_1",
    "f150": "MF_2",
    "f220": "UHF_1",
    "f280": "UHF_2",
}

sim_list = {
    kind: {
        flavor: os.path.join(
            JBOLO_MODELS_PATH,
            f"V3r8/V3r8_{name}/LAT/V3r8_{name}_LAT_{flavor}.yaml",
        )
        for flavor in ("LF", "MF", "UHF")
    }
    for kind, name in (("baseline", "Baseline"), ("goal", "Goal"))
}


################################################################
# UFM / band bookkeeping                                       #
################################################################


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
    for ot, ufms in ufm_dict.items():
        if ufm in ufms:
            return ot
    raise ValueError(f"Error: no OT found for UFM {ufm}")


def bands_for_ufm(ufm: str) -> tuple[str, str]:
    """
    Get the (low, high) bands observed by a UFM.

    Parameters
    ----------
    ufm : str
        UFM of interest, e.g. "mv11".

    Returns
    -------
    bands : tuple[str, str]
        The low and high band of the UFM, e.g. ("f090", "f150").

    Raises
    ------
    ValueError
        If the UFM type is not recognized.
    """
    for prefix, bands in UFM_BANDS.items():
        if prefix in ufm:
            return bands
    raise ValueError(f"Error: bad ufm {ufm}.")


def ufm_band_index(ufm: str, band: str) -> int:
    """
    Get the band index (1 for the low band, 2 for the high band) of a UFM.

    Parameters
    ----------
    ufm : str
        UFM of interest.
    band : str
        Band of interest, e.g. "f090".

    Returns
    -------
    ufm_band : int
        1 for the low band, 2 for the high band.

    Raises
    ------
    ValueError
        If the band is not observed by the UFM.
    """
    bands = bands_for_ufm(ufm)
    if band not in bands:
        raise ValueError(f"Error: bad band {band} for ufm {ufm}")
    return bands.index(band) + 1


def keys_from_wafer(wafer: str, band: str) -> tuple[str, str]:
    """
    Get the bolocalc flavor and channel keys for a UFM and band.

    Parameters
    ----------
    wafer : str
        UFM of interest.
    band : str
        Band of interest.

    Returns
    -------
    ufm_type : str
        Flavor of the UFM, i.e. "LF", "MF", or "UHF".
    ufm_band : str
        Channel of the band, e.g. "MF_1".

    Raises
    ------
    ValueError
        If the band is not observed by the UFM.
    """
    ufm_band_index(wafer, band)
    ufm_band = flavor_dict[band]
    return ufm_band.split("_")[0], ufm_band


def get_band_flag(meta: core.AxisManager, ufm: str, band: str) -> np.ndarray:
    """
    Get the mask of detectors on a UFM which belong to a given band.

    Parameters
    ----------
    meta : core.AxisManager
        Metadata containing det_info and det_cal.
    ufm : str
        UFM of interest.
    band : str
        Band of interest.

    Returns
    -------
    flag : np.ndarray
        Boolean mask over dets. Empty if meta has no detectors.
    """
    wafer_flag = np.array([ufm in stream_id for stream_id in meta.det_info.stream_id])
    if len(wafer_flag) == 0:
        return wafer_flag
    bp = (meta.det_cal.bg % 4) // 2
    return wafer_flag * (bp == ufm_band_index(ufm, band) - 1)


def get_white_noise(meta: core.AxisManager, flag: np.ndarray) -> np.ndarray:
    """
    Get the white noise of the flagged detectors from the preprocess metadata.

    Parameters
    ----------
    meta : core.AxisManager
        Metadata containing preprocess noise information.
    flag : np.ndarray
        Detector mask.

    Returns
    -------
    wnoise : np.ndarray
        White noise of the flagged detectors.

    Raises
    ------
    ValueError
        If no valid noise key is found in the metadata.
    """
    for key in ("noise", "noiseT"):
        if key in meta.preprocess:
            return meta.preprocess[key].white_noise[flag]
    raise ValueError(f"Error: no valid noise key in {meta.preprocess.keys()}")


################################################################
# PWV                                                          #
################################################################


def pwv_interp(
    filepath: str | Path = APEX_PWV_PATH,
    time_cut: float = 17410 * 1e5,
) -> interpolate.interp1d:
    """
    Interpolates APEX pwv data. Should be replaced by SO radiometer when it becomes available

    Parameters
    ----------
    filepath : str | Path
        Path to apex pwv data
    time_cut : float, Default = 17410*1e5
        Time in unix below which to exclude data.

    Returns
    -------
    pwv : scipy.interpolate.interp1d
        Interp object of pwv vs unix time
    """
    with np.load(filepath, allow_pickle=True) as x:
        timestamp = x["timestamp"]
        pwv = x["pwv"]

    flags = timestamp >= time_cut
    # APEX to CLASS best fit from Max
    return interpolate.interp1d(timestamp[flags], apex_to_tocolin_250701(pwv[flags]))


################################################################
# RJ -> CMB temperature conversion                             #
################################################################


def tag_substr(dest: object, tags: dict, max_recursion: int = 20) -> object:
    """
    Do string substitution of all tags into dest (in-place if dest is a dict).
    'Borrowed' from sotodlib; used to replace tags within yaml files.

    Parameters
    ----------
    dest : object
        String, list, tuple or dict to substitute tags into.
    tags : dict
        Tags to substitute.
    max_recursion : int, default 20
        Maximum dictionary depth.

    Returns
    -------
    dest : object
        dest with tags substituted.
    """
    assert max_recursion > 0  # Too deep this dictionary.
    if isinstance(dest, str):
        # Keep subbing until it doesn't change any more...
        new = dest.format(**tags)
        while dest != new:
            dest = new
            new = dest.format(**tags)
        return dest
    if isinstance(dest, list):
        return [tag_substr(x, tags) for x in dest]
    if isinstance(dest, tuple):
        return (tag_substr(x, tags) for x in dest)
    if isinstance(dest, dict):
        for k, v in dest.items():
            dest[k] = tag_substr(v, tags, max_recursion - 1)
        return dest
    return dest


def load_sim(filename: str) -> dict:
    """
    Load a bolocalc sim yaml, substituting its tags.

    Parameters
    ----------
    filename : str
        Path to the sim yaml.

    Returns
    -------
    sim : dict
        The loaded sim.
    """
    with open(filename) as f:
        s = yaml.safe_load(f)
    if "tags" in s:
        tag_substr(s, s["tags"])
    return s


def load_band_file(fname: str) -> np.ndarray:
    """
    Load a bolocalc band file relative to JBOLO_MODELS_PATH.

    Parameters
    ----------
    fname : str
        Band file path relative to JBOLO_MODELS_PATH.

    Returns
    -------
    band : np.ndarray
        (freq, response) arrays.
    """
    return np.loadtxt(os.path.join(JBOLO_MODELS_PATH, fname), unpack=True)


def x_naught(nu: u.Quantity, tb: u.Quantity) -> u.Quantity:
    """
    Compute the dimensionless frequency h nu / k T.

    Parameters
    ----------
    nu : u.Quantity
        Frequency.
    tb : u.Quantity
        Brightness temperature.

    Returns
    -------
    x : u.Quantity
        h nu / (k_B tb).
    """
    return const.h * nu / (const.k_B * tb)


def sigma(nu: u.Quantity, tb: u.Quantity) -> u.Quantity:
    """
    Compute the blackbody derivative weighting used for the band center.

    Parameters
    ----------
    nu : u.Quantity
        Frequency.
    tb : u.Quantity
        Brightness temperature.

    Returns
    -------
    sigma : u.Quantity
        nu^2 e^x / (e^x - 1)^2.
    """
    x = x_naught(nu, tb)
    return nu**2 * np.exp(x) / (np.exp(x) - 1) ** 2


def temp_conv(
    T_B: u.Quantity, flavor: str, ch: str, kind: str = "baseline"
) -> u.Quantity:
    """
    Convert temperature in RJ to Boltzmann temperature at T_B.
    Set T_B = T_cmb to get into CMB units.

    Parameters
    ----------
    T_B : u.Quantity
        Brightness temperature, e.g. 2.725 K.
    flavor : str
        "LF", "MF", or "UHF".
    ch : str
        Channel, e.g. "MF_1".
    kind : str, default "baseline"
        Which bolocalc sim to use, "baseline" or "goal".

    Returns
    -------
    dTB_dTrj : u.Quantity
        Dimensionless conversion factor.
    """
    sim = load_sim(sim_list[kind][flavor])
    freq, band = load_band_file(sim["channels"][ch]["band_response"]["fname"])
    nu = freq * u.GHz
    nu_c = np.trapezoid(nu * band * sigma(nu, T_B), nu) / np.trapezoid(
        band * sigma(nu, T_B), nu
    )
    x_0 = x_naught(nu_c, T_B)
    return ((np.exp(x_0) - 1) ** 2 / (x_0**2 * np.exp(x_0))).to(1)


@cache
def cmb_conv(band: str) -> float:
    """
    Get the K_RJ -> K_CMB conversion for a band.

    Parameters
    ----------
    band : str
        Band of interest, e.g. "f090".

    Returns
    -------
    conv : float
        Dimensionless conversion factor.
    """
    ch = flavor_dict[band]
    return float(temp_conv(T_B=2.725 * u.Kelvin, flavor=ch.split("_")[0], ch=ch))


################################################################
# Script plumbing                                              #
################################################################


def parse_date(d: str) -> dt.datetime:
    """
    Parse a YYYY-MM-DD string into a UTC datetime.

    Parameters
    ----------
    d : str
        Date string.

    Returns
    -------
    date : dt.datetime
        UTC datetime.
    """
    return dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def add_time_args(parser: ap.ArgumentParser, start: str, end: str) -> None:
    """
    Add --start/--end date arguments to a parser.

    Parameters
    ----------
    parser : ap.ArgumentParser
        Parser to add arguments to.
    start : str
        Default start date, YYYY-MM-DD.
    end : str
        Default end date, YYYY-MM-DD.
    """
    parser.add_argument(
        "--start", "-s", type=parse_date, default=start, help="Start time for obs"
    )
    parser.add_argument(
        "--end", "-e", type=parse_date, default=end, help="End time for obs"
    )


def date_str() -> str:
    """
    Get today's date (America/New_York) as MMDDYYYY, used to tag outputs.

    Returns
    -------
    date_str : str
        Today's date.
    """
    return dt.datetime.now(tz=ZoneInfo("America/New_York")).strftime("%m%d%Y")


def save_pickle(obj: object, path: str | Path) -> None:
    """
    Pickle an object with dill, creating the parent directory if needed.

    Parameters
    ----------
    obj : object
        Object to save.
    path : str | Path
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pk.dump(obj, f)
    print(f"Saved {path}")
