"""Helpers for abscal.py."""

import os
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import sotodlib.io.metadata as io_meta
from astropy import constants as consts
from astropy.table import Table
from astropy.time import Time
from pixell import enmap
from sotodlib import core

from latcom.bands.bands import bandwidths
from latcom.planet_models.core import get_planet_temp
from latcom.utils.common import (
    BANDS,
    DATA_DIR,
    bands_for_ufm,
    cmb_conv,
    get_band_flag,
    keys_from_wafer,
    ot_from_ufm,
    ufm_dict,
)

# Uniform +/- 20% from nominal was:
# {"030": [6.0, 8.8], "040": [4.1, 6.1], "090": [1.8, 2.6],
#  "150": [1.1, 1.6], "220": [0.8, 1.2], "280": [0.7, 1.0]}
fwhm_cuts = {
    "030": [5.1, 9.7],
    "040": [3.5, 6.8],
    "090": [1.8, 3.0],
    "150": [1.1, 2.0],
    "220": [0.8, 1.8],
    "280": [0.7, 1.6],
}

# These are roughly 20% - 500% the expected beam value
beam_volume_cuts = {
    "030": [2e-7, 6e-6],
    "040": [1e-7, 5e-6],
    "090": [8e-8, 2e-6],
    "150": [4e-8, 1e-6],
    "220": [1e-8, 2e-7],
    "280": [1e-8, 2e-7],
}

# Important times in LAT history i.e. slipage/alignment
lat_times = {
    "alignment0": {"start": 1744848000, "stop": 1745150000},
    "cr_slip0": {"start": 1745150000, "stop": 1749355200},
    "alignment1": {"start": 1749600000, "stop": 1755576000},
    "alignment2": {"start": 1756699200, "stop": 20000000000},
}

# (polar, equatorial) radii in km. Jupiter/Saturn from Weiland et al. 2011,
# others from the nssdc fact sheets https://nssdc.gsfc.nasa.gov/planetary/factsheet/
planet_radii = {
    "jupiter": (66854, 71492),
    "saturn": (54364, 60268),
    "mars": (3376.2, 3396.2),
    "uranus": (24973, 25559),
    "neptune": (24341, 24764),
}

AU_KM = 1.495978707e8
EPH_PATH = DATA_DIR / "JPL_ephem.fits"
_cached_eph_table = None

RESULT_FIELDS = (
    "cal",
    "chi",
    "obs",
    "raw_cal",
    "el",
    "pwv",
    "fwhm",
    "raw_opt",
    "cal_opt",
    "omega_data",
    "source",
    "time",
    "relcal",
)
# Abscal columns of the per obs results dataframe
CAL_COLS = ("cals", "raw_cals", "cals_cmb", "raw_cals_cmb")
# result_dict field -> cal_dict field, where they differ
_RESULT_FROM_CAL = {"cal": "adj_cal"}


################################################################
# Observation loading and cuts                                 #
################################################################


def load_amans(
    f: h5py.File,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the beam fit AxisManagers from a beam_pars h5 file.

    Parameters
    ----------
    f : h5py.File
        Open beam_pars h5 file, organized as [obs_id][stream_id][band].

    Returns
    -------
    amans : np.ndarray
        Beam fit AxisManagers.
    obs_ids : np.ndarray
        Obs id of each AxisManager.
    stream_ids : np.ndarray
        Stream id of each AxisManager.
    bands : np.ndarray
        Band of each AxisManager, e.g. "f090".
    """
    entries = [
        (o, s, b, "full" if "full" in f[o][s][b] else "")
        for o in f
        for s in f[o]
        for b in f[o][s]
        if b[1:] in BANDS and b[0] == "f"
    ]

    amans, keep = [], []
    for o, s, b, p in entries:
        try:
            amans.append(core.AxisManager.load(f[os.path.join(o, s, b, p)]))
            keep.append((o, s, b))
        except KeyError:
            continue

    obs_ids, stream_ids, bands = (np.array(x) for x in zip(*keep))
    return np.array(amans), obs_ids, stream_ids, bands


def get_inner_radial_mask(
    resid_map: enmap.ndmap, pixsize: float, radius: float
) -> np.ndarray:
    """
    Get a boolean mask for the pixels in resid_map that are within radius arcmin of the center of the map.

    Parameters
    ----------
    resid_map : enmap.ndmap
        Residual map to get the mask for.
    pixsize : float
        Pixel size of the map in arcmin.
    radius : float
        Radius in arcmin to use for the mask.

    Returns
    -------
    mask : np.ndarray
        Boolean mask for the pixels in resid_map that are within radius arcmin of the center of the map.
    """
    x0, y0 = int(resid_map.shape[0] / 2), int(resid_map.shape[1] / 2)
    XX, YY = np.meshgrid(np.arange(resid_map.shape[1]), np.arange(resid_map.shape[0]))
    return np.sqrt((XX - x0) ** 2 + (YY - y0) ** 2) * pixsize < radius


def get_beam_keepout(band: str) -> float:
    """
    Get the keepout radius in arcmin for the given band, 0.2 deg scaled from 90 GHz.

    Parameters
    ----------
    band : str
        Frequency of the residual map.

    Returns
    -------
    keepout : float
        Keepout radius in arcmin for the given band.

    Raises
    ------
    ValueError
        If the band is not one of "030", "040", "090", "150", "220", or "280".
    """
    if band not in BANDS:
        raise ValueError(f"Invalid band: {band}")
    return 0.2 * 60 * 90 / int(band)


def get_resid_rmse(resid_path: str, band: str) -> float:
    """
    Get the RMSE of the residual map at resid_path for the given band.

    Parameters
    ----------
    resid_path : str
        Path to the residual map.
    band : str
        Frequency of the residual map.

    Returns
    -------
    rmse : float
        RMSE of the residual map.
    """
    resid_map = enmap.read_map(resid_path)
    mask = get_inner_radial_mask(
        resid_map=resid_map,
        pixsize=np.abs(resid_map.wcs.wcs.cdelt[0]) * 60,
        radius=get_beam_keepout(band=band),
    )
    return np.sqrt(np.mean(resid_map[mask] ** 2))


def get_relcal(meta: core.AxisManager, ufm: str, band: str) -> float:
    """
    Get the relcal for a given observation, ufm, and frequency.

    Parameters
    ----------
    meta : AxisManager
        The metadata containing preprocessing information.
    ufm : str
        The ufm ID to get relcal for.
    band : str
        The frequency band to get relcal for.

    Returns
    -------
    relcal : float
        The relcal for the given observation, ufm, and frequency, averaged over all dets.
    """
    flag = get_band_flag(meta, ufm, band)
    if len(flag) == 0:
        print(f"No relcal info for obs {meta.obs_info.obs_id}, ufm {ufm}, band {band}")
        return np.nan
    flag = flag * (meta.relcal.rel_factor_error < 0.1)
    return np.nanmean(meta.relcal.rel_factor[flag])


################################################################
# Functions to get cal factor                                  #
################################################################


def data_to_cal_factor(
    p_meas: float,
    omega_beam: float,
    omega_planet: float,
    bandwidth: float,
    planet_temp: float,
) -> tuple[float, float]:
    """
    Convert a measured planet amplitude to a calibration factor and optical efficiency.

    Parameters
    ----------
    p_meas : float
        Measured planet amplitude in pW.
    omega_beam : float
        Beam solid angle in sr.
    omega_planet : float
        Planet solid angle in sr.
    bandwidth : float
        Bandwidth in GHz.
    planet_temp : float
        Planet brightness temperature in K_RJ.

    Returns
    -------
    cal_factor : float
        Calibration factor in K/pW.
    opt_eff : float
        Optical efficiency.
    """
    t_eff_planet = planet_temp * omega_planet / omega_beam
    cal_factor = t_eff_planet / p_meas
    opt_eff = ((1 / cal_factor * u.pW / u.K) / (consts.k_B * bandwidth * u.GHz)).to(1)
    return cal_factor, opt_eff


def interp_JPL_Horizons(
    planet: str,
    ctimes: float | np.ndarray,
    eph: Table,
    cols: tuple = ("ang_width", "delta", "PDObsLat"),
) -> dict:
    """
    Interpolate JPL Horizons ephemerides to the requested times.

    Parameters
    ----------
    planet : str
        Planet name, as stored in the eph "planet" column.
    ctimes : float | np.ndarray
        Unix timestamp(s).
    eph : Table
        Ephemerides table with a "ctime" column.
    cols : tuple
        Names of Horizons.ephemerides() columns to interpolate.

    Returns
    -------
    ephem_dict : dict
        Dictionary of ephemerides data, interpolated to the input ctimes.
    """
    tab = eph[eph["planet"] == planet]
    tab.sort("ctime")
    ephem_dict = {}
    for c in cols:
        y = np.asarray(tab[c], dtype=float)
        # unwrap so 359.9 -> 0.1 doesn't blow up
        if c == "RA":
            y = np.rad2deg(np.unwrap(np.deg2rad(y)))
        ephem_dict[c] = np.interp(ctimes, tab["ctime"], y)
    if "RA" in cols:
        ephem_dict["RA"] = np.mod(ephem_dict["RA"], 360.0)
    return ephem_dict


def get_jpl_planet_omega(
    planet: str, ctime: float, eph_path: str | Path = EPH_PATH
) -> float:
    """
    Get omega_planet using JPL Horizons Ephemerides.

    Parameters
    ----------
    planet : str
        Planet to get solid angle for.
    ctime : float
        Unix timestamp for evaluation of the ephemerides data.
    eph_path : str | Path
        Path to astropy Table fits file with the JPL Horizons ephemerides,
        sampled every 20 minutes (see query_jpl.py).

    Returns
    -------
    omega_planet : float
        Planet solid angle in [sr]

    Raises
    ------
    ValueError
        If the planet argument is not one of jupiter, saturn, mars, uranus, or neptune.
    """
    global _cached_eph_table
    if _cached_eph_table is None:
        eph = Table.read(eph_path)
        eph["ctime"] = Time(eph["datetime_jd"], format="jd", scale="utc").unix
        _cached_eph_table = eph

    if planet not in planet_radii:
        raise ValueError(
            f"Planet {planet} not recognized. Must be one of {list(planet_radii)}"
        )
    r_pol, r_eq = planet_radii[planet]

    jpl_ephem = interp_JPL_Horizons(
        eph=_cached_eph_table, planet=planet, ctimes=ctime, cols=("delta", "PDObsLat")
    )
    # Planetodetic latitude of the center of the observed disk
    Dw = np.deg2rad(jpl_ephem["PDObsLat"])
    # Projected polar radius along line of sight (Planck LII intermediate results;
    # Weiland 2011 has this formula but flips r_eq and r_pol in the ratio)
    rproj = r_pol * np.sqrt(np.cos(Dw) ** 2 + (r_eq / r_pol) ** 2 * np.sin(Dw) ** 2)
    return np.pi * r_eq * rproj / (jpl_ephem["delta"] * AU_KM) ** 2


def get_single_abscal(
    amp: float,
    planet: str,
    timestamp: str,
    band: str,
    ufm: str,
    el_obs: float,
    omega_beam: float,
    pwv_obs: float,
    atmosphere_eff: dict,
    fiducial_pwv: float = 1.0,
    fiducial_el: str = "50",
) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
    """
    Get the abscal for a single observation.

    Parameters
    ----------
    amp : float
        The amplitude of the abscal fit, in pW.
    planet : str
        Planet that this abscal is computed from.
    timestamp : str
        Time of observation
    band : str
        The band for which to get the temperature.
    ufm : str
        The ufm associated with the abscal.
    el_obs : float
        Elevation of observation.
    omega_beam : float
        Beam solid angle in SR.
    pwv_obs : float
        PWV of obs.
    atmosphere_eff : dict
        Dictionary which gives atm efficiency modification for pwv/el.
    fiducial_pwv : float, default 1.0
        PWV in mm to adjust the amplitude to.
    fiducial_el : str, default "50"
        Elevation key of atmosphere_eff to adjust the amplitude to.

    Returns
    -------
    abscal : float
        PWV/elevation adjusted abscal in K/pW.
    opt_eff : float
        PWV/elevation adjusted optical efficiency.
    raw_abscal : float
        Unadjusted abscal in K/pW.
    raw_opt_eff : float
        Unadjusted optical efficiency.
        All four are None if the planet temperature or elevation is unavailable.
    """
    planet_temp = get_planet_temp(planet=planet, obs_id=timestamp, band=band, ufm=ufm)
    if planet_temp is None:
        return None, None, None, None

    if el_obs > 90:
        el_obs = 180 - el_obs
    try:
        obs_key_el = next(
            el for el in atmosphere_eff["LF"]["LF_1"] if np.abs(int(el) - el_obs) < 2.5
        )
    except StopIteration:
        return None, None, None, None

    # Get pwv/el adjustment
    pwvs = np.asarray(atmosphere_eff["pwv"])
    pwv_idx = np.where(np.abs(pwvs - fiducial_pwv) < 0.1)[0]
    obs_idx_pwv = np.argmin(np.abs(pwvs - pwv_obs))

    ufm_type, ufm_band = keys_from_wafer(ufm, band)
    t_atm = atmosphere_eff[ufm_type][ufm_band]
    pwv_adjust = t_atm[fiducial_el][pwv_idx] / t_atm[obs_key_el][obs_idx_pwv]

    omega_planet = get_jpl_planet_omega(planet=planet.lower(), ctime=int(timestamp))

    def _cal(p_meas: float) -> tuple[float, float]:
        return data_to_cal_factor(
            p_meas=p_meas,
            omega_beam=omega_beam,
            omega_planet=omega_planet,
            bandwidth=bandwidths[ufm][band],
            planet_temp=planet_temp,
        )

    # By default we use the pwv/sin(el) adjusted abscals, but keep the unadjusted ones
    abscal, opt_eff = _cal(amp * pwv_adjust[0])
    raw_abscal, raw_opt_eff = _cal(amp)
    return abscal, opt_eff, raw_abscal, raw_opt_eff


################################################################
# Collating results                                            #
################################################################


def make_results_dict(cal_dict: dict) -> dict:
    """
    Function which makes the result dict from the cal dict.
    Mostly this just reorganizes data; cal_dict has primary keys [obs_id],
    this function reorganizes them to [ufm][freq], which is more useful.

    Parameters
    ----------
    cal_dict : dict
        Dictionary of results keyed by {ufm}_{band}_{obs_id}

    Returns
    -------
    result_dict : dict
        Dictionary of results by [ufm][freq]
    """
    result_dict = {}
    for key, cal in cal_dict.items():
        ufm, freq = key.split("_")[:2]
        if ufm not in result_dict:
            result_dict[ufm] = {
                band: {field: [] for field in RESULT_FIELDS}
                for band in bands_for_ufm(ufm)
            }
        entry = result_dict[ufm][freq]
        entry["obs"].append(key)
        for field in RESULT_FIELDS:
            cal_field = _RESULT_FROM_CAL.get(field, field)
            if cal_field in cal:
                entry[field].append(cal[cal_field])
    return result_dict


def get_tube_abscal(
    ot: str, full_df: pd.DataFrame
) -> tuple[float, float, float, float]:
    """
    Get the relcal-corrected abscals for a given tube, averaged over its UFMs.

    Parameters
    ----------
    ot : str
        The tube to get abscals for.
    full_df : pd.DataFrame
        Dataframe of observations. Should contain a single freq and epoch,
        otherwise they are averaged together.

    Returns
    -------
    cal : float
        The abscal in K/pW.
    raw_cal : float
        The raw abscal in K/pW.
    cal_cmb : float
        The abscal in K_CMB/pW.
    raw_cal_cmb : float
        The raw abscal in K_CMB/pW.
    """
    ot_df = full_df[full_df.ufms.isin(ufm_dict[ot])]
    return tuple(float(np.nanmean(ot_df[col] * ot_df.relcals)) for col in CAL_COLS)


def compute_tube_abscals(ot: str, tube_df: pd.DataFrame) -> dict[str, tuple]:
    """
    Compute the abscals of each UFM in a tube: the tube averaged relcal-corrected
    abscal divided by the UFM's mean relcal. UFMs without observations get the
    average of their tube-mates.

    Parameters
    ----------
    ot : str
        The tube to compute abscals for.
    tube_df : pd.DataFrame
        Dataframe of observations of the tube at a single freq and epoch.

    Returns
    -------
    abscals : dict[str, tuple]
        For each UFM in the tube, (cal, raw_cal, cal_cmb, raw_cal_cmb, omega),
        with cals in K/pW or K_CMB/pW and omega the beam solid angle in sr.
    """
    tube_cals = get_tube_abscal(ot, tube_df)
    abscals = {}
    for ufm in ufm_dict[ot]:
        ufm_df = tube_df[tube_df.ufms == ufm]
        if len(ufm_df):
            relcal = np.nanmean(ufm_df.relcals)
            abscals[ufm] = (
                *(float(c / relcal) for c in tube_cals),
                float(np.nanmean(ufm_df.omegas)),
            )

    mates = np.array(list(abscals.values())).reshape(-1, 5)
    for ufm in ufm_dict[ot]:
        if ufm not in abscals:
            print(f"No obs of {ufm}, using the average of its tube-mates")
            abscals[ufm] = tuple(float(x) for x in np.nanmean(mates, axis=0))
    return abscals


def _results_df(result_dict: dict) -> pd.DataFrame:
    """
    Flatten a result_dict into a dataframe with one row per observation.

    Parameters
    ----------
    result_dict : dict
        Dictionary of results sorted by [ufm][freq]

    Returns
    -------
    df : pd.DataFrame
        Dataframe of per obs results sorted by freq then ufm.
    """
    rows = []
    for freq in BANDS:
        conv = cmb_conv(freq)  # Temperature for rj->cmb
        for ufm in sorted(result_dict):
            sub = result_dict[ufm].get(freq)
            if sub is None:
                continue
            for cal, raw_cal, omega, obs, relcal in zip(
                sub["cal"], sub["raw_cal"], sub["omega_data"], sub["obs"], sub["relcal"]
            ):
                rows.append(
                    {
                        "freqs": freq,
                        "ufms": ufm,
                        "cals": cal,
                        "raw_cals": raw_cal,
                        "cals_cmb": cal * conv,
                        "raw_cals_cmb": raw_cal * conv,
                        "omegas": omega,
                        "obs": float(obs.rsplit("_", 1)[-1]),
                        "relcals": relcal,
                    }
                )
    return pd.DataFrame(rows)


def make_db(
    result_dict: dict, h5_path: str | Path, verbose: bool = True
) -> core.metadata.ManifestDb:
    """
    Write per-epoch abscals to h5_path and make the ManifestDB indexing them.

    Parameters
    ----------
    result_dict : dict
        Dictionary of results sorted by [ufm][freq]
    h5_path : str | Path
        Path of the h5 file to write the abscals to.
    verbose : bool, default True
        Print tube/freq combinations with no observations in an epoch.

    Returns
    -------
    db : core.metadata.ManifestDb
        Manifest db with abscal results for each LAT epoch.
    """
    df = _results_df(result_dict)
    tubes = sorted({ot_from_ufm(ufm) for ufm in result_dict})

    for epoch, times in lat_times.items():
        in_epoch = (df.obs >= times["start"]) & (df.obs <= times["stop"])
        abscals = {}
        for freq in BANDS:
            for ot in tubes:
                tube_mask = (df.freqs == freq) & df.ufms.isin(ufm_dict[ot])
                if not tube_mask.any():
                    continue  # Tube doesn't observe this freq
                tube_df = df[tube_mask & in_epoch]
                if tube_df.empty:
                    # No obs of the tube in this epoch, so use the all time average
                    if verbose:
                        print(f"No {ot} f{freq} obs in {epoch}, using all epochs")
                    tube_df = df[tube_mask]
                for ufm, cals in compute_tube_abscals(ot, tube_df).items():
                    abscals[ufm, freq] = cals

        data = []
        for ufm in sorted({ufm for ufm, _ in abscals}):
            for freq in bands_for_ufm(ufm):
                cals = abscals.get((ufm, freq), (np.nan,) * 5)
                data.append((f"ufm_{ufm}", f"f{freq}", *cals))
            data.append((f"ufm_{ufm}", "NC", *[np.nan] * 5))

        rs = core.metadata.ResultSet(
            keys=[
                "dets:stream_id",
                "dets:wafer.bandpass",
                "abscal_rj",
                "raw_abscal_rj",
                "abscal_cmb",
                "raw_abscal_cmb",
                "beam_solid_angle",
            ]
        )
        rs.rows = data
        io_meta.write_dataset(rs, str(h5_path), epoch, overwrite=True)

    # Record in ManifestDb.
    scheme = core.metadata.ManifestScheme()
    scheme.add_range_match("obs:timestamp")
    scheme.add_data_field("dataset")
    db = core.metadata.ManifestDb(scheme=scheme)
    for epoch, times in lat_times.items():
        db.add_entry(
            {"obs:timestamp": (times["start"], times["stop"]), "dataset": epoch},
            filename=Path(h5_path).name,
        )
    return db


def get_abscals_from_perobs(
    abscal_perobs: core.metadata.ResultSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the abscals from a ResultSet of per observation abscals.

    Parameters
    ----------
    abscal_perobs : core.metadata.ResultSet
        ResultSet containing per observation abscals.

    Returns
    -------
    abscal_rj : np.ndarray
        The per observation abscal.
    raw_abscal_rj : np.ndarray
        The per observation raw abscal.
    abscal_cmb : np.ndarray
        The per observation abscal to Kcmb.
    raw_abscal_cmb : np.ndarray
        The per observation raw abscal to Kcmb.
    """
    df = abscal_perobs.to_pandas()
    abscal_rj = df["abscal"].to_numpy()
    raw_abscal_rj = df["raw_abscal_rj"].to_numpy()
    conv = np.array([cmb_conv(band) for band in df["band"]], dtype=float)
    return abscal_rj, raw_abscal_rj, abscal_rj * conv, raw_abscal_rj * conv
