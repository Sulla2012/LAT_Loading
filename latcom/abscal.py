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
        default="/global/cfs/cdirs/sobs/users/skh/data/beams/lat/pointing_model_atm_relcal/",  # ASO
        help="Path to h5 file containing beam fits",
    )

    parser.add_argument(
        "--no_results",
        "-nr",
        action="store_true",
        help="Whether to save the abscal results as a pickle file",
    )
    parser.add_argument(
        "--skip_planets",
        "-sp",
        default=["saturn", "neptune"],
        help="List of planets to skip during abscal calculation.",
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
    data_dir = args.datadir
    f = h5py.File(data_dir + "beam_pars.h5", mode="r")
    amans, obs_ids, stream_ids, bands = au.load_amans(f)

    cal_dict = {}

    ctx = core.Context("../ctxs/abscal_ctx_09242026.yaml")

    for i, aman in enumerate(amans):
        obs_id = obs_ids[i].split("_")[1]
        if "ufm" in stream_ids[i]:
            ufm = stream_ids[i].split("_")[1]
        else:
            ufm = stream_ids[i]
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
        elif "neptune" in tags:
            planet = "neptune"
        else:
            print(f"Error: no planet in tags: {tags}")
            continue
        if planet in args.skip_planets:
            continue

        subdir = obs_ids[i]
        resid_name = subdir + "_" + ufm + "_f" + band + "_full_resid.fits"
        try:
            resid_path = os.path.join(data_dir, planet, obs_id[:5], subdir, resid_name)
            rmse = mu.get_resid_rmse(resid_path, band)
        except FileNotFoundError:
            continue

        # Third cut is on RMSE
        if rmse > 0.05:
            print(f"RMSE = {rmse} > 0.05")
            continue

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

        abscal, opt_eff, raw_abscal, raw_opt_eff = au.get_single_abscal(
            amp=amp,
            planet=planet,
            timestamp=obs_id,
            band=band,
            ufm=ufm,
            el_obs=el_obs,
            solid_angle=data_solid_angle,
            pwv_obs=pwv_obs,
        )
        if abscal is None:
            continue

        if raw_abscal >= 40 and planet == "saturn":
            continue  # Some of the saturn observations are accidentally of Neptune, leading to very high abscals (when using Saturn temp)
            # Matt is working on a real fix but for now since the Neptune amp is >10x lower, a cut on the abscal is safe

        relcal = au.get_relcal(meta=meta, ufm=ufm, band=band)

        cal_dict[str(ufm) + "_" + str(band) + "_" + str(obs_id)] = {
            "adj_cal": abscal,
            "raw_cal": raw_abscal,
            "pwv": pwv_obs,
            "el": el_obs,
            "omega_data": data_solid_angle,
            "fwhm": fitted_fwhm,
            "raw_opt": raw_opt_eff,
            "cal_opt": opt_eff,
            "source": planet,
            "time": obs_id,
            "relcal": relcal,
        }

    if save_results:
        today = dt.datetime.now(tz=ZoneInfo("America/New_York")).date()
        date_str = str(today.month).zfill(2) + str(today.day).zfill(2) + str(today.year)

        result_dict = au.make_results_dict(
            cal_dict=cal_dict,
        )

        with open(f"../abscals/results_{date_str}.pk", "wb") as f:
            pk.dump(result_dict, f)

        with open(f"../abscals/abscals_{date_str}.pk", "wb") as f:
            pk.dump(cal_dict, f)

        # Now to write the manifest db
        db = au.make_db(result_dict=result_dict)
        db.to_file(f"../abscals/db_{date_str}.sqlite")
