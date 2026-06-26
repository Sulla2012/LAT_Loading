import os

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import sotodlib.io.metadata as io_meta
from astropy import constants as consts
from sotodlib import core

from latcom.utils import map_utils as mu


def data_to_cal_factor(
    p_meas, beam_solid_angle, planet_diameter, bandwidth, planet_temp
):
    fiducial_solid_angle = mu.angular_diameter_to_solid_angle(planet_diameter)

    fill_factor = fiducial_solid_angle / (beam_solid_angle)
    t_eff_planet = planet_temp * fill_factor
    cal_factor = t_eff_planet / p_meas  # K -> pW

    opt_eff = ((1 / (cal_factor) * u.pW / u.K) / (consts.k_B * bandwidth * u.GHz)).to(1)

    return cal_factor, opt_eff


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
    "030": [6.0, 10.8],
    "040": [4.1, 8.1],
    "090": [1.8, 3.0],
    "150": [1.1, 2.0],
    "220": [0.8, 1.8],
    "280": [0.7, 1.6],
}

# These are roughly 20% - 500% the expected beam value
beam_volume_cuts = {
    "090": [8e-8, 2e-6],
    "150": [4e-8, 1e-6],
    "220": [1e-8, 2e-7],
    "280": [1e-8, 2e-7],
}


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

    freqs = ["030", "040", "090", "150", "220", "280"]
    ufms = sorted(result_dict.keys())

    flavor_dict = {
        "030": "LF_1",
        "040": "LF_2",
        "090": "MF_1",
        "150": "MF_2",
        "220": "UHF_1",
        "280": "UHF_2",
    }

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

    data_freqs = np.array(data_freqs)
    data_ufms = np.array(data_ufms)
    cals = np.array(cals)
    raw_cals = np.array(raw_cals)
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
                data.append(
                    (
                        "ufm_" + str(ufm),
                        "f" + str(freq),
                        float(np.nanmean(cur_df.cals)),
                        float(np.nanmean(cur_df.raw_cals)),
                        float(np.nanmean(cur_df.cals_cmb)),
                        float(np.nanmean(cur_df.raw_cals_cmb)),
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
        io_meta.write_dataset(rs, "../abscals/abscals.h5", f"{key}", overwrite=True)

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
