import argparse as ap
import datetime as dt
import os
from zoneinfo import ZoneInfo

import dill as pk
import h5py
import numpy as np
import pandas as pd
import sotodlib.io.metadata as io_meta
from astropy import units as u
from sotodlib import core
from sotodlib.core.metadata.loader import LoaderError

from latcom.bands.bands import bandwidths
from latcom.planet_models.core import get_planet_temp
from latcom.utils import abscal_utils as au
from latcom.utils import map_utils as mu
from latcom.utils.optical_loading import keys_from_wafer, pwv_interp


def _make_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(
        description="Compute abscal factors for Saturn/Mars observations"
    )
    parser.add_argument(
        "--datadir",
        "-dd",
        type=str,
        default="/global/cfs/cdirs/sobs/users/skh/data/beams/lat/pointing_model_atm_relcal/",
        help="Path to h5 file containing beam fits",
    )

    parser.add_argument(
        "--no-results",
        "-nr",
        action="store_true",
        help="Whether to save the abscal results as a pickle file",
    )
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    #TODO: set lacom path
    with open("../data/atmosphere_eff.pk", "rb") as f:
        atmosphere_eff = pk.load(f)

    fiducial_elevation = 50
    fiducial_pwv = 1  # mm
    el_key = "50"  # hardcoded :(
    pwv = pwv_interp()

    save_results = not args.no_results  # note inversion

    # This is the fpath used for nominal SO Commissioning. Keeping for reproducibility
    # fpath = "/so/home/saianeesh/data/beams/lat_old/source_maps/pointing_model/fits/beam_pars.h5"

    # ASO path
    data_dir = args.datadir
    f = h5py.File(data_dir + "beam_pars.h5", mode="r")
    obs_ids = []
    times = []
    stream_ids = []
    bands = []
    for o in f:
        for s in f[o]:
            for b in f[o][s]:
                obs_ids += [o]
                times += [float(o.split("_")[1])]
                stream_ids += [s]
                bands += [b]
    limit_bands = ["f030", "f040", "f090", "f150", "f220", "f280"]
    msk = np.isin(bands, limit_bands)
    obs_ids = np.array(obs_ids)[msk]
    times = np.array(times)[msk]
    stream_ids = np.array(stream_ids)[msk]
    bands = np.array(bands)[msk]

    amans = []
    for o, s, b in zip(obs_ids, stream_ids, bands):
        try:
            amans.append(core.AxisManager.load(f[os.path.join(o, s, b, "")]))
        except KeyError:
            continue

    amans = np.array(amans)

    cal_dict = {}

    ctx = core.Context(
        "/global/cfs/cdirs/sobs/metadata/lat/contexts/smurf_detsets_local.yaml"
    )


    for i, aman in enumerate(amans):
        obs_id = obs_ids[i].split("_")[1]
        ufm = stream_ids[i].split("_")[1]
        band = bands[i][1:]
        ufm_type, ufm_band = keys_from_wafer(ufm, band)

        fitted_fwhm = aman.data_fwhm.to(u.arcmin).value
        data_solid_angle = aman.data_solid_angle_corr.value

        if "amp_outer" in aman:
            amp = aman.amp.value + aman.amp_outer.value
        else:
            amp = aman.amp.value

        # First cut is on FWHM
        if au.fwhm_cuts[band][1] < fitted_fwhm or fitted_fwhm < au.fwhm_cuts[band][0]:
            print(au.fwhm_cuts[band][0], fitted_fwhm, au.fwhm_cuts[band][1], band, ufm)
            continue

        # Second cut is on beam volume
        if (
            au.beam_volume_cuts[band][1] < data_solid_angle
            or data_solid_angle < au.beam_volume_cuts[band][0]
        ):
            print(
                au.beam_volume_cuts[band][0],
                data_solid_angle,
                au.beam_volume_cuts[band][1],
                band,
                ufm,
            )
            continue

        # Get planet temperature
        try:
            tags = ctx.obsdb.get(obs_ids[i], tags=True)["tags"]
        except LoaderError:
            continue

        if "mars" in tags:
            planet = "mars"
        elif "saturn" in tags:
            planet = "saturn"
        elif "uranus" in tags:
            planet = "uranus"
        else:
            print(f"Error: no planet in tags: {tags[-1]}")
            continue

        planet_temp = get_planet_temp(planet=planet, obs_id=obs_id, band=band, ufm=ufm)
        if planet_temp is None:
            continue

        subdir = obs_ids[i]
        resid_name = subdir + "_ufm_" + ufm + "_f" + band + "_resid.fits"
        resid_path = os.path.join(data_dir, planet, obs_id[:5], subdir, resid_name)
        rmse = mu.get_resid_rmse(resid_path, band)

        # Third cut is on RMSE
        if rmse > 0.05:
            print(f"RMSE = {rmse} > 0.05")
            continue

        # Get pwv/el adjustment
        start_date = dt.datetime.fromtimestamp(int(obs_id), dt.UTC) - dt.timedelta(
            days=1
        )
        end_date = dt.datetime.fromtimestamp(int(obs_id), dt.UTC) + dt.timedelta(days=1)

        pwv_idx = np.where(
            np.array(
                [np.abs(pwv - fiducial_pwv) < 0.1 for pwv in atmosphere_eff["pwv"]]
            )
        )[0]

        try:
            pwv_obs = pwv(obs_id)
        except ValueError:
            print(f"obs {obs_id} outside of pwv range")
            continue
        if np.isnan(pwv_obs):
            continue
        if pwv_obs > 2.5:
            continue

        meta = ctx.get_meta(obs_ids[i])

        el_obs = meta.obs_info.el_center

        if el_obs > 90:
            el_obs = 180 - el_obs

        obs_idx_pwv = np.where(
            np.isclose(
                np.abs(atmosphere_eff["pwv"] - pwv_obs),
                np.min(np.abs(atmosphere_eff["pwv"] - pwv_obs)),
            )
        )[0][0]
        try:
            obs_key_el = next([
                el
                for el in atmosphere_eff["LF"]["LF_1"]
                if np.abs(int(el) - el_obs) < 2.5
            ])
        except:
            print(f"El {el_obs} for obs {obs_ids[i]} out of range")
            continue

        t_atm_obs = atmosphere_eff[ufm_type][ufm_band][obs_key_el][obs_idx_pwv]
        t_atm_fiducial = atmosphere_eff[ufm_type][ufm_band][el_key][pwv_idx]
        pwv_adjust = t_atm_fiducial / t_atm_obs

        adjusted_amplitude = amp * pwv_adjust[0]

        planet_diameter = mu.get_planet_diameter(
            int(obs_id), planet.capitalize()
        )  # arcsec, we are using exact temperatures

        cal_factor, cal_opt_efc = au.data_to_cal_factor(
            p_meas=adjusted_amplitude,
            beam_solid_angle=data_solid_angle,
            planet_diameter=planet_diameter,
            bandwidth=bandwidths[ufm][band],
            planet_temp=planet_temp,
        )
        raw_factor, raw_opt_efc = au.data_to_cal_factor(
            p_meas=amp,
            beam_solid_angle=data_solid_angle,
            planet_diameter=planet_diameter,
            bandwidth=bandwidths[ufm][band],
            planet_temp=planet_temp,
        )

        if raw_factor >= 40 and planet == "saturn":
            continue  # Some of the saturn observations are accidentally of Neptune, leading to very high abscals (when using Saturn temp)
            # Matt is working on a real fix but for now since the Neptune amp is >10x lower, a cut on the abscal is safe
        cal_dict[str(ufm) + "_" + str(band) + "_" + str(obs_id)] = {
            "adj_cal": cal_factor,
            "raw_cal": raw_factor,
            "pwv": pwv_obs,
            "el": el_obs,
            "omega_data": data_solid_angle,
            "fwhm": fitted_fwhm,
            "raw_opt": raw_opt_efc,
            "cal_opt": cal_opt_efc,
            "source": planet,
            "time": obs_id,
        }


    if save_results:
        with open("abscals.pk", "wb") as f:
            pk.dump(cal_dict, f)

        result_dict = {}

        for key in cal_dict:
            ufm = key.split("_")[0]
            freq = key.split("_")[1]
            if ufm in result_dict:
                continue
            if "090" in freq or "150" in freq:
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

        today = dt.datetime.now(tz=ZoneInfo("America/New_York")).date()
        date_str = str(today.month).zfill(2) + str(today.day).zfill(2) + str(today.year)

        with open(f"results_{date_str}.pk", "wb") as f:
            pk.dump(result_dict, f)

        # Now to write the manifest db

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
        data_solid_angles = []
        omegas = []
        obs = []

        pwv = pwv_interp()

        freqs = ["090", "150", "220", "280"]
        ufms = sorted(result_dict.keys())

        flavor_dict = {"090": "MF_1", "150": "MF_2", "220": "UHF_1", "280": "UHF_2"}

        for freq in freqs:
            temp_conv = mu.temp_conv(
                T_B=2.725 * u.Kelvin,
                flavor=flavor_dict[freq].split("_")[0],
                ch=flavor_dict[freq],
                kind="baseline",
            )  # Temperature for rj->cmb
            for ufm in ufms:
                for key in result_dict:
                    if ufm not in key:
                        continue
                    for sub_key in result_dict[key]:
                        if freq not in sub_key:
                            continue
                        cur_cals = np.array(result_dict[key][sub_key]["cal"])
                        cur_raw_cals = np.array(result_dict[key][sub_key]["raw_cal"])
                        omega_data = np.array(result_dict[key][sub_key]["omega_data"])
                        cur_obs = np.array(result_dict[key][sub_key]["obs"])
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

        for key in lat_times:
            data = []

            # For each period, we're going to compute the average abscal for each ufm and freq
            mfs = ["090", "150"]
            ufs = ["220", "280"]
            for ufm in ufms:
                for freq in freqs:
                    if freq in mfs and "uv" in ufm:
                        continue
                    if freq in ufs and "mv" in ufm:
                        continue
                    if (
                        len(
                            np.where((df.freqs == str(freq)) & (df.ufms == str(ufm)))[0]
                        )
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
                        cur_df = df.where(
                            (df.freqs == str(freq)) & (df.ufms == str(ufm))
                        )
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
            io_meta.write_dataset(rs, "abscals.h5", f"{key}", overwrite=True)

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

        # db.add_entry({'dataset': 'abscal'}, filename='abscals.h5')
        db.to_file("db.sqlite")
