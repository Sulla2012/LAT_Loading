import datetime as dt
import os
from sqlite3 import Time
from zoneinfo import ZoneInfo

import astropy.units as u
import dill as pk
import h5py
import numpy as np
import pandas as pd
import sotodlib.io.metadata as io_meta
from astropy import constants as consts
from astropy.table import Table
from astropy.time import Time
from sotodlib import core

import latcom.utils.optical_loading as ol
from latcom.bands.bands import bandwidths
from latcom.planet_models.core import get_planet_temp
from latcom.utils import map_utils as mu
from latcom.utils.optical_loading import keys_from_wafer


# Uniform +/- 20% from nominal. TODO: Make the percentage offset a variable
# fwhm_cuts = {
#    "030": [6.0, 8.8],
#    "040": [4.1, 6.1],
#    "090": [1.8, 2.6],
#    "150": [1.1, 1.6],
#    "220": [0.8, 1.2],
#    "280": [0.7, 1.0],
# }

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

flavor_dict = {
    "030": "LF_1",
    "040": "LF_2",
    "090": "MF_1",
    "150": "MF_2",
    "220": "UHF_1",
    "280": "UHF_2",
}

rad_to_arcsec = (180 * 3600) / np.pi
rad_to_arcmin = (180 * 60) / np.pi
rad_to_deg = 180 / np.pi

#so3gsite = proj.coords.SITES["so"]
#site = so3gsite.ephem_observer()

EPH_PATH = '/global/u2/j/jorlo/dev/LAT_Loading/data/JPL_ephem.fits'
_cached_eph_table = None

################################################################
# Functions to get cal factor                                  #
################################################################

def data_to_cal_factor(
    p_meas: float, omega_beam: float, omega_planet: float, bandwidth: float, planet_temp: float
):
    fill_factor = omega_planet / (omega_beam)
    t_eff_planet = planet_temp * fill_factor
    cal_factor = t_eff_planet / p_meas  # K -> pW

    opt_eff = ((1 / (cal_factor) * u.pW / u.K) / (consts.k_B * bandwidth * u.GHz)).to(1)

    return cal_factor, opt_eff

def get_planet_diameter(obs_id, planet):
    """
    Get angular diameter (in ARCSEC) of a planet from obs id.
    Depreciated in favor of get_jpl_planet_omega, which uses 
    JPL Horizons ephemerides to get the solid angle of the planet.
    """
    raise DeprecationWarning(
        "get_planet_diameter is depreciated. Use get_jpl_planet_omega instead."
    )
    timestamp = int(obs_id)
    date = dt.datetime.fromtimestamp(timestamp, tz=ZoneInfo("UTC"))
    site.date = ephem.Date(date)
    saturn = getattr(ephem, planet)(site)
    dia = saturn.size
    return dia

def interp_JPL_Horizons(
        planet: str, ctimes: np.ndarray, eph: Table, cols: tuple = ('ang_width', 'delta', 'PDObsLat')
    ):
    '''
    arguments:
    eph: loaded astropy Table fits file (database for the ephemerides)
    planet: either saturn or jupiter
    ctimes: array of unix timestamps
    cols: names of Horizons.ephemerides() columns desired for query

    returns:
    ephem_dict: dictionary of ephemerides data, interpolated to the input ctimes
    '''
    tab = eph[eph['planet'] == planet]
    tab.sort('ctime')
    ephem_dict = {}
    for c in cols:
        y = np.asarray(tab[c], dtype=float)
        # unwrap so 359.9 -> 0.1 doesn't blow up
        if c == 'RA':                      
            y = np.rad2deg(np.unwrap(np.deg2rad(y)))
        ephem_dict[c] = np.interp(ctimes, tab['ctime'], y)
    if 'RA' in cols:
        ephem_dict['RA'] = np.mod(ephem_dict['RA'], 360.0)
    return ephem_dict

def get_jpl_planet_omega(planet: str, ctime:float, eph_path:str=EPH_PATH):
    """
    Get omega_planet using JPL Horizons Ephemerides:

    Parameters
    ----------
    eph_path : str
        Path to astropy Table fits file with database for the JPL Horizons ephemerides. 
        Ephemerides database consists of samples every 20 minutes,
        so interpolation is needed to access the data at the exact ctime argument provided
    planet : str
        Planet to get solid angle for.
    ctime : float
        Unix timestamp for evaluation of the ephemerides data. 

    Returns
    -------
    omega_planet : float
        Planet solid angle in [sr]

    Raises
    ------
    ValueError 
        If the planet argument is not one of jupiter, saturn, mars, uranus, or neptune.
    """
    #jpl ephemeris, cache it so we don't keep loading it.
    global _cached_eph_table
    if _cached_eph_table is None:
        eph = Table.read(eph_path)
        eph['ctime'] = Time(eph['datetime_jd'], format='jd', scale='utc').unix
        _cached_eph_table = eph
    else:
        eph = _cached_eph_table

    if planet == 'jupiter':
        r_pol = 66854 #km, from Weiland et al. 2011
        r_eq = 71492 #km
    elif planet == 'saturn':
        r_pol = 54364
        r_eq = 60268
    elif planet == 'mars':
        r_pol = 3376.2 #From nssdc fact sheet:
        r_eq = 3396.2 # https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
    elif planet == 'uranus':
        r_pol = 24973 #From nssdc fact sheet:
        r_eq = 25559 # https://nssdc.gsfc.nasa.gov/planetary/factsheet/uranusfact.html
    elif planet == 'neptune':
        r_pol = 24341 #From nssdc fact sheet:
        r_eq = 24764 # https://nssdc.gsfc.nasa.gov/planetary/factsheet/neptunefact.html

    else:
        raise ValueError(f"Planet {planet} not recognized. Must be one of jupiter, saturn, mars, uranus, or Neptune")

    jpl_ephem = interp_JPL_Horizons(eph=eph, planet=planet, ctimes=ctime, cols=('ang_width', 'delta','PDObsLat'))
    Dw = np.deg2rad(jpl_ephem['PDObsLat']) #planetodetic lattiude, where center of observed disk is relative to the equatorial plane of the planet
    planet_distance = jpl_ephem['delta'] #Earth--> planet distance

    def get_rproj(r_pol, r_eq, Dw):
        return r_pol * np.sqrt(np.cos(Dw)**2 + (r_eq/r_pol)**2 * np.sin(Dw)**2) #from planck LII intermediate results, WMAP Weiland 2011 also has this formula but accidentally flips r_eq and r_pol in the ratio in the sqrt()

    rproj = get_rproj(r_pol=r_pol,r_eq=r_eq,Dw=Dw) #projected polar radius along line of site to planet (correcting for time-dependent changes to planet oblateness)
    omega_planet = (np.pi*r_eq*rproj/(planet_distance*1.495978707e8)**2) #planck LII formula, planet distance naturally in AU, converted to km
    return omega_planet

def angular_diameter_to_solid_angle(angular_diameter):
    """Angular diameter is in ARCSEC"""
    return 2 * np.pi * (1 - np.cos(np.deg2rad(angular_diameter / 3600) / 2))

def make_results_dict(
    cal_dict: dict,
) -> dict:
    """
    Function which makes the result dict from the cal dict.
    Mostly this just reorganizes data; cal_dict has primary keys [obs_id],
    this function reorganizes them to [ufm][freq], which is more useful.

    Parameters
    ----------
    cal_dict : dict
        Dictionary of results by obs_id

    Returns
    -------
    result_dict : dict
        Dictionary of results by [ufm][freq]
    """
    result_dict = {}

    for key in cal_dict:
        ufm = key.split("_")[0]
        freq = key.split("_")[1]
        if ufm in result_dict:
            continue
        if "030" in freq or "040" in freq:
            result_dict[ufm] = {
                "030": {
                    "cal": [],
                    "chi": [],
                    "obs": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "fwhm": [],
                    "raw_opt": [],
                    "cal_opt": [],
                    "omega_data": [],
                    "source": [],
                    "time": [],
                    "relcal": [],
                },
                "040": {
                    "cal": [],
                    "chi": [],
                    "obs": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "fwhm": [],
                    "raw_opt": [],
                    "cal_opt": [],
                    "omega_data": [],
                    "source": [],
                    "time": [],
                    "relcal": [],
                },
            }
        elif "090" in freq or "150" in freq:
            result_dict[ufm] = {
                "090": {
                    "cal": [],
                    "chi": [],
                    "obs": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "fwhm": [],
                    "raw_opt": [],
                    "cal_opt": [],
                    "omega_data": [],
                    "source": [],
                    "time": [],
                    "relcal": [],
                },
                "150": {
                    "cal": [],
                    "chi": [],
                    "obs": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "fwhm": [],
                    "raw_opt": [],
                    "cal_opt": [],
                    "omega_data": [],
                    "source": [],
                    "time": [],
                    "relcal": [],
                },
            }
        else:
            result_dict[ufm] = {
                "220": {
                    "cal": [],
                    "chi": [],
                    "obs": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "fwhm": [],
                    "raw_opt": [],
                    "cal_opt": [],
                    "omega_data": [],
                    "source": [],
                    "time": [],
                    "relcal": [],
                },
                "280": {
                    "cal": [],
                    "chi": [],
                    "obs": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "fwhm": [],
                    "raw_opt": [],
                    "cal_opt": [],
                    "omega_data": [],
                    "source": [],
                    "time": [],
                    "relcal": [],
                },
            }
    for key in cal_dict:
        ufm = key.split("_")[0]
        freq = key.split("_")[1]
        result_dict[ufm][freq]["cal"].append(cal_dict[key]["adj_cal"])
        result_dict[ufm][freq]["raw_cal"].append(cal_dict[key]["raw_cal"])
        result_dict[ufm][freq]["el"].append(cal_dict[key]["el"])
        result_dict[ufm][freq]["pwv"].append(cal_dict[key]["pwv"])
        result_dict[ufm][freq]["obs"].append(key)
        result_dict[ufm][freq]["fwhm"].append(cal_dict[key]["fwhm"])
        result_dict[ufm][freq]["raw_opt"].append(cal_dict[key]["raw_opt"])
        result_dict[ufm][freq]["cal_opt"].append(cal_dict[key]["cal_opt"])
        result_dict[ufm][freq]["omega_data"].append(cal_dict[key]["omega_data"])
        result_dict[ufm][freq]["source"].append(cal_dict[key]["source"])
        result_dict[ufm][freq]["time"].append(cal_dict[key]["time"])
        result_dict[ufm][freq]["relcal"].append(cal_dict[key]["relcal"])
    return result_dict


def make_db(result_dict: dict) -> core.metadata.ManifestDb:
    """
    Make the ManifestDB from the result_dict.

    Parameters
    ----------
    result_dict : dict
        Dictionary of results sorted by [ufm][freq]

    Returns
    db : core.metadata.ManifestDb
        Manifest db with abscal results for each LAT epoch.

    """
    # Load important times in LAT history i.e. slipage/alighnment
    lat_times = {
        "alignment0": {"start": 1744848000, "stop": 1745150000},
        "cr_slip0": {"start": 1745150000, "stop": 1749355200},
        "alignment1": {"start": 1749600000, "stop": 1755576000},
        "alignment2": {"start": 1756699200, "stop": 20000000000},
    }
    cals = []
    raw_cals = []
    data_freqs = []
    data_ufms = []
    cals_cmb = []
    raw_cals_cmb = []
    omegas = []
    obs = []
    relcals = []

    freqs = ["030", "040", "090", "150", "220", "280"]
    ufms = sorted(result_dict.keys())

    for freq in freqs:
        temp_conv = mu.temp_conv(
            T_B=2.725 * u.Kelvin,
            flavor=flavor_dict[freq].split("_")[0],
            ch=flavor_dict[freq],
            kind="baseline",
        )  # Temperature for rj->cmb
        for ufm in ufms:
            for key, sub_dict in result_dict.items():
                if ufm not in key:
                    continue
                for sub_key in sub_dict:
                    if freq not in sub_key:
                        continue
                    cur_cals = np.array(sub_dict[sub_key]["cal"])
                    cur_raw_cals = np.array(sub_dict[sub_key]["raw_cal"])
                    omega_data = np.array(sub_dict[sub_key]["omega_data"])
                    cur_obs = np.array(sub_dict[sub_key]["obs"])
                    for j in range(len(cur_cals)):
                        cals.append(cur_cals[j])
                        raw_cals.append(cur_raw_cals[j])
                        cals_cmb.append(cur_cals[j] * temp_conv)
                        raw_cals_cmb.append(cur_cals[j] * temp_conv)
                        data_freqs.append(freq)
                        data_ufms.append(ufm)
                        omegas.append(omega_data[j])
                        obs.append(cur_obs[j][9:])
                        relcals.append(sub_dict[sub_key]["relcal"][j])

    data_freqs = np.array(data_freqs)
    data_ufms = np.array(data_ufms)
    cals = np.array(cals)
    raw_cals = np.array(raw_cals)
    relcals = np.array(relcals)
    obs = np.array(obs, dtype=float)

    df = pd.DataFrame(
        {
            "freqs": data_freqs,
            "ufms": data_ufms,
            "cals": cals,
            "raw_cals": raw_cals,
            "cals_cmb": cals_cmb,
            "raw_cals_cmb": raw_cals_cmb,
            "omegas": omegas,
            "obs": obs,
            "relcals": relcals,
        }
    )
    lfs = ["030", "040"]
    mfs = ["090", "150"]
    ufs = ["220", "280"]
    for key in lat_times:
        data = []

        # For each period, we're going to compute the average abscal for each ufm and freq

        for ufm in ufms:
            for freq in freqs:
                if freq in lfs and "ln" not in ufm:
                    continue
                if freq in mfs and "mv" not in ufm:
                    continue
                if freq in ufs and "uv" not in ufm:
                    continue
                if (
                    len(np.where((df.freqs == str(freq)) & (df.ufms == str(ufm)))[0])
                    == 0
                ):
                    print(
                        freq, ufm
                    )  # Let me know if there are no obs with this array/freq

                if (
                    len(
                        np.where(
                            (df.freqs == str(freq))
                            & (df.ufms == str(ufm))
                            & (df.obs >= lat_times[key]["start"])
                            & (df.obs <= lat_times[key]["stop"])
                        )[0]
                    )
                    == 0
                ):
                    # If there are no obs in this particular time range, just use the all time average for that array
                    cur_df = df.where((df.freqs == str(freq)) & (df.ufms == str(ufm)))
                else:
                    cur_df = df.where(
                        (df.freqs == str(freq))
                        & (df.ufms == str(ufm))
                        & (df.obs >= lat_times[key]["start"])
                        & (df.obs <= lat_times[key]["stop"])
                    )
                cal, raw_cal, cal_cmb, raw_cal_cmb = compute_abscals(
                    ufm_df=cur_df, full_df=df
                )
                data.append(
                    (
                        "ufm_" + str(ufm),
                        "f" + str(freq),
                        cal,
                        raw_cal,
                        cal_cmb,
                        raw_cal_cmb,
                        float(np.nanmean(cur_df.omegas)),
                    )
                )

            data.append(
                ("ufm_" + str(ufm), "NC", np.nan, np.nan, np.nan, np.nan, np.nan)
            )

        # Write to HDF5
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
        io_meta.write_dataset(
            rs,
            "/global/u2/j/jorlo/dev/LAT_Loading/abscals/abscals.h5",
            f"{key}",
            overwrite=True,
        )

    # Record in ManifestDb.
    scheme = core.metadata.ManifestScheme()
    scheme.add_range_match("obs:timestamp")
    scheme.add_data_field("dataset")

    db = core.metadata.ManifestDb(scheme=scheme)
    db.add_entry(
        {
            "obs:timestamp": (
                lat_times["alignment0"]["start"],
                lat_times["alignment0"]["stop"],
            ),
            "dataset": "alignment0",
        },
        filename="abscals.h5",
    )
    db.add_entry(
        {
            "obs:timestamp": (
                lat_times["cr_slip0"]["start"],
                lat_times["cr_slip0"]["stop"],
            ),
            "dataset": "cr_slip0",
        },
        filename="abscals.h5",
    )
    db.add_entry(
        {
            "obs:timestamp": (
                lat_times["alignment1"]["start"],
                lat_times["alignment1"]["stop"],
            ),
            "dataset": "alignment1",
        },
        filename="abscals.h5",
    )
    db.add_entry(
        {
            "obs:timestamp": (
                lat_times["alignment2"]["start"],
                lat_times["alignment2"]["stop"],
            ),
            "dataset": "alignment2",
        },
        filename="abscals.h5",
    )

    return db


def get_aso_abscal(ot: str, full_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Get the abscals for a given ASO tube.

    Parameters
    ----------
    ot : str
        The ASO tube to get abscals for.
    full_df : pd.DataFrame
        Dataframe of observations for the full dataset.

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
    ufm_list = ol.ufm_dict[ot]
    ot_df = full_df.where(full_df.ufms.isin(ufm_list))
    cal = np.nanmean(ot_df.cals * ot_df.relcals)
    raw_cal = np.nanmean(ot_df.raw_cals * ot_df.relcals)
    cal_cmb = np.nanmean(ot_df.cals_cmb * ot_df.relcals)
    raw_cal_cmb = np.nanmean(ot_df.raw_cals_cmb * ot_df.relcals)
    return cal, raw_cal, cal_cmb, raw_cal_cmb


def compute_abscals(
    ufm_df: pd.DataFrame, full_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the abscals for a given dataframe of observations.

    Parameters
    ----------
    ufm_df : pd.DataFrame
        Dataframe of observations for a given ufm and freq.
    full_df : pd.DataFrame
        Dataframe of observations for the full dataset.

    Returns
    -------
    cals : np.ndarray
        Array of abscals in K/pW.
    raw_cals : np.ndarray
        Array of raw abscals in K/pW.
    cals_cmb : np.ndarray
        Array of abscals in K_CMB/pW.
    raw_cals_cmb : np.ndarray
        Array of raw abscals in K_CMB/pW.
    """
    if ufm_df.ufms[0] in ol.aso_tubes:
        ot = ol.ot_from_ufm(ufm_df.ufms[0])
        ot_cal, ot_raw_cal, ot_cal_cmb, ot_raw_cal_cmb = get_aso_abscal(ot, full_df)
        cal = ot_cal / np.nanmean(ufm_df.relcals)
        raw_cal = ot_raw_cal / np.nanmean(ufm_df.relcals)
        cal_cmb = ot_cal_cmb / np.nanmean(ufm_df.relcals)
        raw_cal_cmb = ot_raw_cal_cmb / np.nanmean(ufm_df.relcals)

    else:
        # For the nominal SO tubes we have enough per ufm
        # data that we can just use the mean
        cal = float(np.nanmean(ufm_df.cals))
        raw_cal = float(np.nanmean(ufm_df.raw_cals))
        cal_cmb = float(np.nanmean(ufm_df.cals_cmb))
        raw_cal_cmb = float(np.nanmean(ufm_df.raw_cals_cmb))

    return cal, raw_cal, cal_cmb, raw_cal_cmb


def load_amans(
    f: h5py.File,
) -> tuple[
    np.array,
    np.array,
    np.array,
    np.array,
]:
    obs_ids = []
    times = []
    stream_ids = []
    bands = []
    paths = []
    for o in f:
        for s in f[o]:
            for b in f[o][s]:
                obs_ids += [o]
                times += [float(o.split("_")[1])]
                stream_ids += [s]
                bands += [b]
                paths += ["full" if "full" in f[o][s][b] else ""]

    limit_bands = ["f030", "f040", "f090", "f150", "f220", "f280"]
    msk = np.isin(bands, limit_bands)
    obs_ids = np.array(obs_ids)[msk]
    times = np.array(times)[msk]
    stream_ids = np.array(stream_ids)[msk]
    bands = np.array(bands)[msk]
    paths = np.array(paths)[msk]

    amans = []
    flags = np.ones_like(obs_ids, bool)
    for i, (o, s, b, p) in enumerate(zip(obs_ids, stream_ids, bands, paths)):
        try:
            amans.append(core.AxisManager.load(f[os.path.join(o, s, b, p)]))
        except KeyError:
            flags[i] = False
            continue

    amans = np.array(amans)

    return amans, obs_ids[flags], stream_ids[flags], bands[flags]


def bootstrap(cals: np.ndarray, samps: int = 1000) -> np.ndarray:
    samples = np.zeros(samps)
    rng = np.random.default_rng()
    for i in range(len(samples)):
        sample = rng.choice(cals, size=len(cals))
        samples[i] = np.mean(sample)
    return samples


def bootstrap_err(cals: np.ndarray, q=None, samps: int = 1000) -> np.ndarray:
    if q is None:
        q = [15.9, 84.1]
    samples = bootstrap(cals, samps)
    return np.percentile(samples, q)


def get_ufm_band(band: str) -> int:
    """
    Get the ufm band from the band string.

    Parameters
    ----------
    band : str
        The band string, e.g. "f030", "f040", etc.

    Returns
    -------
    ufm_band : int
        The ufm band, 1 for LF, 2 for MF, 3 for UHF.

    Raises
    ------
    ValueError
        If the band string is not recognized.
    """
    if band in ["030", "090", "220"]:
        return 1
    elif band in ["040", "150", "280"]:
        return 2
    else:
        raise ValueError(
            f"Band {band} not recognized. Must be one of 030, 040, 090, 150, 220, 280."
        )


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
    wafer_flag = np.array([ufm in ufms for ufms in meta.det_info.stream_id])

    if len(wafer_flag) == 0:
        print(f"No relcal info for obs {meta.obs_info.obs_id}, ufm {ufm}, band {band}")
        return np.nan

    ufm_band = get_ufm_band(band)

    bp = (meta.det_cal.bg % 4) // 2

    if ufm_band == 1:
        net_flag = wafer_flag * (bp == 0)
    elif ufm_band == 2:
        net_flag = wafer_flag * (bp == 1)
    net_flag = net_flag * (meta.relcal.rel_factor_error < 0.1)

    relcal = meta.relcal.rel_factor[net_flag]
    return np.nanmean(relcal)


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
) -> tuple[float, float, float, float] | None:
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
    """
    fiducial_pwv = 1  # mm
    planet_temp = get_planet_temp(planet=planet, obs_id=timestamp, band=band, ufm=ufm)
    if planet_temp is None:
        return None, None, None, None
    # Get pwv/el adjustment
    pwv_idx = np.where(
        np.array([np.abs(pwv - fiducial_pwv) < 0.1 for pwv in atmosphere_eff["pwv"]])
    )[0]

    if el_obs > 90:
        el_obs = 180 - el_obs

    obs_idx_pwv = np.where(
        np.isclose(
            np.abs(atmosphere_eff["pwv"] - pwv_obs),
            np.min(np.abs(atmosphere_eff["pwv"] - pwv_obs)),
        )
    )[0][0]
    try:
        obs_key_el = [
            el for el in atmosphere_eff["LF"]["LF_1"] if np.abs(int(el) - el_obs) < 2.5
        ][0]

    except IndexError:
        return None, None, None, None
    # Adjust the amplitude for the pwv
    el_key = "50"  # hardcoded fiducial observation idx. TODO: Double check this is up-to-date

    ufm_type, ufm_band = keys_from_wafer(ufm, band)
    t_atm_obs = atmosphere_eff[ufm_type][ufm_band][obs_key_el][obs_idx_pwv]
    t_atm_fiducial = atmosphere_eff[ufm_type][ufm_band][el_key][pwv_idx]
    pwv_adjust = t_atm_fiducial / t_atm_obs

    adjusted_amplitude = amp * pwv_adjust[0]

    omega_planet = get_jpl_planet_omega(
        planet=planet.lower(), ctime=int(timestamp), 
    )  # arcsec, we are using exact temperatures

    # By default we use the pwv/sin(el) adjusted abscals, but I want to save/keep the
    # unadjusted ones as well
    abscal, opt_eff = data_to_cal_factor(
        p_meas=adjusted_amplitude,
        omega_beam=omega_beam,
        omega_planet=omega_planet,
        bandwidth=bandwidths[ufm][band],
        planet_temp=planet_temp,
    )
    raw_abscal, raw_opt_eff = data_to_cal_factor(
        p_meas=amp,
        omega_beam=omega_beam,
        omega_planet=omega_planet,
        bandwidth=bandwidths[ufm][band],
        planet_temp=planet_temp,
    )

    return abscal, opt_eff, raw_abscal, raw_opt_eff

def get_abscals_from_perobs(abscal_perobs: core.metadata.ResultSet) -> tuple[float, float, float, float]:
    """
    Get the abscals from a ResultSet of per observation abscals.

    Parameters
    ----------
    abscal_perobs : core.metadata.ResultSet
        ResultSet containing per observation abscals.

    Returns
    -------
    abscal_rj : float
        The per observation abscal.
    raw_abscal_rj : float
        The per observation raw abscal.
    abscal_cmb : float
        The per observation abscal to Kcmb.
    raw_abscal_cmb : float
        The per observation raw abscal to Kcmb.
    """
    abscal_perobs = abscal_perobs.to_pandas()
    abscal_rj = abscal_perobs["abscal"]
    raw_abscal_rj = abscal_perobs["raw_abscal_rj"]
    abscal_cmb = np.zeros_like(abscal_rj)
    raw_abscal_cmb = np.zeros_like(abscal_rj)
    for i in range(len(abscal_perobs)):
        band = abscal_perobs["band"][i]
        temp_conv = mu.temp_conv(
            T_B=2.725 * u.Kelvin,
            flavor=flavor_dict[band].split("_")[0],
            ch=flavor_dict[band],
            kind="baseline",
        ) 
        abscal_cmb[i] = abscal_rj * temp_conv
        raw_abscal_cmb[i] = raw_abscal_rj * temp_conv
 
    return abscal_rj, raw_abscal_rj, abscal_cmb, raw_abscal_cmb