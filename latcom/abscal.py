import argparse as ap
import datetime as dt
import os
from zoneinfo import ZoneInfo

import dill as pk
import h5py
import numpy as np
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
        nargs="+",
        default=[
            "/global/cfs/cdirs/sobs/users/skh/data/beams/lat/pointing_model_atm_relcal/",
            "/global/cfs/cdirs/sobs/users/skh/data/beams/lat/pointing_model_template_bs_relcal/",
        ],
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
    # TODO: set lacom path
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
    data_dirs = args.datadir
    if isinstance(data_dirs, list):
        for i, data_dir in enumerate(data_dirs):
            f = h5py.File(data_dir + "beam_pars.h5", mode="r")
            if i == 0:
                amans, obs_ids, stream_ids, bands = au.load_amans(f)
            else:
                cur_amans, cur_obs_ids, cur_stream_ids, cur_bands = au.load_amans(f)

                amans = np.append(amans, cur_amans)
                obs_ids = np.append(obs_ids, cur_obs_ids)
                stream_ids = np.append(stream_ids, cur_stream_ids)
                bands = np.append(bands, cur_bands)
    else:
        f = h5py.File(data_dirs + "beam_pars.h5", mode="r")
        amans, obs_ids, stream_ids, bands = au.load_amans(f)

    cal_dict = {}

    ctx = core.Context(
        "/global/cfs/cdirs/sobs/metadata/lat/contexts/smurf_detsets_local.yaml"
    )

    for i, aman in enumerate(amans):
        obs_id = obs_ids[i].split("_")[1]
        ufm = stream_ids[i].split("_")[1]
        band = bands[i][1:]
        ufm_type, ufm_band = keys_from_wafer(ufm, band)

        # Get beam pars
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
            print(f"Error: no planet in tags: {tags}")
            continue

        planet_temp = get_planet_temp(planet=planet, obs_id=obs_id, band=band, ufm=ufm)
        if planet_temp is None:
            continue

        subdir = obs_ids[i]
        resid_name = subdir + "_ufm_" + ufm + "_f" + band + "_resid.fits"
        for data_dir in data_dirs:
            try:
                resid_path = os.path.join(
                    data_dir, planet, obs_id[:5], subdir, resid_name
                )
                rmse = mu.get_resid_rmse(resid_path, band)
            except FileNotFoundError:
                continue

        # Third cut is on RMSE
        if rmse > 0.05:
            print(f"RMSE = {rmse} > 0.05")
            continue

        # Get pwv/el adjustment
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
        if pwv_obs > 3.0:
            continue

        # now load the metadata after cuts
        meta = ctx.get_meta(obs_ids[i])
        el_obs = meta.obs_info.el_center
        if el_obs > 90:
            el_obs = 180 - el_obs

        # Convert from actual observed el to the nominal el of observation used in the effective atmosphere dict
        obs_idx_pwv = np.where(
            np.isclose(
                np.abs(atmosphere_eff["pwv"] - pwv_obs),
                np.min(np.abs(atmosphere_eff["pwv"] - pwv_obs)),
            )
        )[0][0]
        try:
            obs_key_el = [
                el
                for el in atmosphere_eff["LF"]["LF_1"]
                if np.abs(int(el) - el_obs) < 2.5
            ][0]

        except IndexError:
            print(f"El {el_obs} for obs {obs_ids[i]} out of range")
            continue

        # Adjust the amplitude for the pwv
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
        today = dt.datetime.now(tz=ZoneInfo("America/New_York")).date()
        date_str = str(today.month).zfill(2) + str(today.day).zfill(2) + str(today.year)

        result_dict = au.make_results_dict(
            cal_dict=cal_dict,
        )

        with open(f"../results_{date_str}.pk", "wb") as f:
            pk.dump(result_dict, f)

        with open(f"../abscals_{date_str}.pk", "wb") as f:
            pk.dump(cal_dict, f)

        # Now to write the manifest db
        db = au.make_db(result_dict=result_dict)
        db.to_file(f"../db_{date_str}.sqlite")
