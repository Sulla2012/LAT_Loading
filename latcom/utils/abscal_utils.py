"""
Abscal helpers. LAT_beams/make_abscal.py uses these to compute per obs abscals
and to collate them into the abscal ManifestDb.
"""

import warnings
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import astropy.units as u
import dill as pk
import numpy as np
import pandas as pd
import sotodlib.io.metadata as io_meta
from astropy import constants as consts
from astropy.table import Table
from astropy.time import Time
from sotodlib import core
from sotodlib.core.metadata.loader import LoaderError

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

# result_dict field -> abscal_perobs column written by make_abscal
PEROBS_FIELDS = {
    "obs": "obs_id",
    "source": "source",
    "amp": "amp",
    "cal": "abscal",
    "raw_cal": "raw_abscal",
    "cal_cmb": "abscal_cmb",
    "raw_cal_cmb": "raw_abscal_cmb",
    "cal_opt": "opt_eff",
    "raw_opt": "raw_opt_eff",
    "omega_data": "solid_angle",
    "relcal": "relcal",
    "el": "el",
    "pwv": "pwv",
}
RESULT_FIELDS = (*PEROBS_FIELDS, "time")
# abscal_perobs columns filled with NaN when make_abscal didn't save them
OPTIONAL_PEROBS = ("solid_angle", "el", "pwv")
# Abscal columns of the per obs results dataframe
CAL_COLS = ("cals", "raw_cals", "cals_cmb", "raw_cals_cmb")


################################################################
# Relcal                                                       #
################################################################


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


def get_obs_relcal(
    ctx: core.Context, obs_id: str, stream_id: str, ufm: str, band: str
) -> float:
    """
    Load the metadata of one UFM/band of an observation and get its mean relcal.

    Parameters
    ----------
    ctx : core.Context
        Context with relcal metadata.
    obs_id : str
        Observation to load.
    stream_id : str
        Stream id of the UFM, e.g. "ufm_mv11".
    ufm : str
        The ufm, e.g. "mv11".
    band : str
        The band, e.g. "f090".

    Returns
    -------
    relcal : float
        The mean relcal, or NaN if the metadata can't be loaded.
    """
    try:
        meta = ctx.get_meta(
            obs_id,
            dets={"dets:stream_id": stream_id, "dets:wafer.bandpass": band},
        )
        return get_relcal(meta, ufm, band)
    except (LoaderError, KeyError, AttributeError) as e:
        print(f"No relcal for obs {obs_id}, ufm {ufm}, band {band}: {e!r}")
        return np.nan


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
        # Horizons marks unitless columns with "---", which astropy warns about
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", u.UnitsWarning)
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


@cache
def load_atmosphere_eff(path: str | Path = DATA_DIR / "atmosphere_eff.pk") -> dict:
    """
    Load the atmospheric efficiency vs pwv/el table.

    Parameters
    ----------
    path : str | Path
        Path to the pickled table.

    Returns
    -------
    atmosphere_eff : dict
        Atmospheric efficiency keyed by "pwv" and [flavor][channel][el].
    """
    with open(path, "rb") as f:
        return pk.load(f)


def get_single_abscal(
    amp: float,
    planet: str,
    timestamp: str,
    band: str,
    ufm: str,
    el_obs: float,
    solid_angle: float,
    pwv_obs: float,
    atmosphere_eff: dict | None = None,
    fiducial_pwv: float = 1.0,
    fiducial_el: str = "50",
) -> tuple[float, float, float, float, float, float] | tuple[None, ...]:
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
        The band, e.g. "f090".
    ufm : str
        The ufm associated with the abscal.
    el_obs : float
        Elevation of observation in degrees.
    solid_angle : float
        Beam solid angle in sr, e.g. of the stacked beam.
    pwv_obs : float
        PWV of obs.
    atmosphere_eff : dict | None, default None
        Dictionary which gives atm efficiency modification for pwv/el.
        Loaded with load_atmosphere_eff if None.
    fiducial_pwv : float, default 1.0
        PWV in mm to adjust the amplitude to.
    fiducial_el : str, default "50"
        Elevation key of atmosphere_eff to adjust the amplitude to.

    Returns
    -------
    abscal : float
        PWV/elevation adjusted abscal in K_RJ/pW.
    abscal_cmb : float
        PWV/elevation adjusted abscal in K_CMB/pW.
    opt_eff : float
        PWV/elevation adjusted optical efficiency.
    raw_abscal : float
        Unadjusted abscal in K_RJ/pW.
    raw_abscal_cmb : float
        Unadjusted abscal in K_CMB/pW.
    raw_opt_eff : float
        Unadjusted optical efficiency.
        All six are None if the planet temperature or elevation is unavailable.
    """
    planet_temp = get_planet_temp(planet=planet, obs_id=timestamp, band=band, ufm=ufm)
    if planet_temp is None:
        return (None,) * 6
    if atmosphere_eff is None:
        atmosphere_eff = load_atmosphere_eff()

    if el_obs > 90:
        el_obs = 180 - el_obs
    try:
        obs_key_el = next(
            el for el in atmosphere_eff["LF"]["LF_1"] if np.abs(int(el) - el_obs) < 2.5
        )
    except StopIteration:
        return (None,) * 6

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
            omega_beam=solid_angle,
            omega_planet=omega_planet,
            bandwidth=bandwidths[ufm][band],
            planet_temp=planet_temp,
        )

    # By default we use the pwv/sin(el) adjusted abscals, but keep the unadjusted ones
    abscal, opt_eff = _cal(amp * pwv_adjust[0])
    raw_abscal, raw_opt_eff = _cal(amp)
    cmb = cmb_conv(band)
    return abscal, abscal * cmb, opt_eff, raw_abscal, raw_abscal * cmb, raw_opt_eff


################################################################
# Collating results                                            #
################################################################


def collate_perobs(
    abscal_perobs: Iterable[core.metadata.ResultSet], ctx: core.Context | None = None
) -> pd.DataFrame:
    """
    Concatenate the abscal_perobs ResultSets written by make_abscal (one per
    tube/band/epoch job) into one table with every column in PEROBS_FIELDS.

    Parameters
    ----------
    abscal_perobs : Iterable[core.metadata.ResultSet]
        The per map abscals, one row per obs/UFM/band.
    ctx : core.Context | None, default None
        Context to load each row's relcal from, used only if the
        ResultSets have no relcal column.

    Returns
    -------
    perobs : pd.DataFrame
        All rows. Missing OPTIONAL_PEROBS columns are NaN.

    Raises
    ------
    ValueError
        If there are no rows, or no relcal column and no ctx to load it with.
    """
    tables = [pd.DataFrame(rs.asarray()) for rs in abscal_perobs]
    if sum(len(t) for t in tables) == 0:
        raise ValueError("No abscal_perobs rows to collate")
    perobs = pd.concat(tables, ignore_index=True)
    for col in OPTIONAL_PEROBS:
        if col not in perobs:
            perobs[col] = np.nan
    if "relcal" not in perobs:
        if ctx is None:
            raise ValueError("abscal_perobs has no relcal and no ctx was given")
        perobs["relcal"] = [
            get_obs_relcal(ctx, row.obs_id, row.stream_id, row.array, row.band)
            for row in perobs.itertuples(index=False)
        ]
    return perobs


def make_results_dict(
    abscal_perobs: Iterable[core.metadata.ResultSet], ctx: core.Context | None = None
) -> dict:
    """
    Collate the per map abscals written by make_abscal and organize them
    by [ufm][freq].

    Parameters
    ----------
    abscal_perobs : Iterable[core.metadata.ResultSet]
        The abscal_perobs ResultSets of make_abscal, see collate_perobs.
    ctx : core.Context | None, default None
        Context to load relcals from if abscal_perobs has none.

    Returns
    -------
    result_dict : dict
        Dictionary of per map results by [ufm][freq][field], see RESULT_FIELDS.
    """
    result_dict = {}
    for row in collate_perobs(abscal_perobs, ctx).itertuples(index=False):
        ufm, freq = row.array, str(row.band)
        if ufm not in result_dict:
            result_dict[ufm] = {
                band: {field: [] for field in RESULT_FIELDS}
                for band in bands_for_ufm(ufm)
            }
        entry = result_dict[ufm][freq]
        for field, col in PEROBS_FIELDS.items():
            entry[field].append(getattr(row, col))
        entry["time"].append(float(row.obs_id.split("_")[1]))
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
    abscal divided by the UFM's mean relcal. UFMs without observations, or
    without any finite relcal, get the average of their tube-mates.

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
        relcals = ufm_df.relcals[np.isfinite(ufm_df.relcals)]
        if len(relcals):
            abscals[ufm] = (
                *(float(c / relcals.mean()) for c in tube_cals),
                float(np.nanmean(ufm_df.omegas)),
            )

    mates = np.array(list(abscals.values())).reshape(-1, 5)
    for ufm in ufm_dict[ot]:
        if ufm not in abscals:
            reason = "relcal" if (tube_df.ufms == ufm).any() else "obs"
            print(f"No {reason} for {ufm}, using the average of its tube-mates")
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
    cols = {
        "cals": "cal",
        "raw_cals": "raw_cal",
        "cals_cmb": "cal_cmb",
        "raw_cals_cmb": "raw_cal_cmb",
        "omegas": "omega_data",
        "obs": "time",
        "relcals": "relcal",
    }
    dfs = [
        pd.DataFrame({col: sub[field] for col, field in cols.items()}).assign(
            freqs=freq, ufms=ufm
        )
        for freq in BANDS
        for ufm in sorted(result_dict)
        if (sub := result_dict[ufm].get(freq))
    ]
    return pd.concat(dfs, ignore_index=True)


def make_db(
    result_dict: dict,
    h5_path: str | Path,
    epochs: dict[str, tuple[float, float]],
    verbose: bool = True,
) -> core.metadata.ManifestDb:
    """
    Write per-epoch abscals to h5_path and make the ManifestDB indexing them.

    Parameters
    ----------
    result_dict : dict
        Dictionary of results sorted by [ufm][freq], see make_results_dict.
    h5_path : str | Path
        Path of the h5 file to write the abscals to.
    epochs : dict[str, tuple[float, float]]
        Epoch name (used as the h5 dataset name) -> [start, stop) unix times.
        Each epoch gets its own abscals.
    verbose : bool, default True
        Print tube/freq combinations with no observations in an epoch,
        whose abscals are set to NaN.

    Returns
    -------
    db : core.metadata.ManifestDb
        Manifest db with abscal results for each epoch.
    """
    df = _results_df(result_dict)
    tubes = sorted({ot_from_ufm(ufm) for ufm in result_dict})
    ufms = sorted(ufm for ot in tubes for ufm in ufm_dict[ot])

    for epoch, (start, stop) in epochs.items():
        in_epoch = (df.obs >= start) & (df.obs < stop)
        abscals = {}
        for freq in BANDS:
            for ot in tubes:
                if freq not in bands_for_ufm(ufm_dict[ot][0]):
                    continue
                tube_df = df[(df.freqs == freq) & df.ufms.isin(ufm_dict[ot]) & in_epoch]
                if tube_df.empty:
                    # No obs of the tube in this epoch, so its abscals are NaN
                    if verbose:
                        print(f"No {ot} f{freq} obs in {epoch}, setting to NaN")
                    continue
                for ufm, cals in compute_tube_abscals(ot, tube_df).items():
                    abscals[ufm, freq] = cals

        data = []
        for ufm in ufms:
            for freq in bands_for_ufm(ufm):
                cals = abscals.get((ufm, freq), (np.nan,) * 5)
                data.append((f"ufm_{ufm}", freq, *cals))
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
    for epoch, (start, stop) in epochs.items():
        db.add_entry(
            {"obs:timestamp": (start, stop), "dataset": epoch},
            filename=Path(h5_path).name,
        )
    return db
