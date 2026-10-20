"""Compute abscal factors from planet observations and write them to a ManifestDb."""

import argparse as ap
import os

import dill as pk
import h5py
import numpy as np
from astropy import units as u
from sotodlib import core
from sotodlib.core.metadata.loader import LoaderError

from latcom.utils import abscal_utils as au
from latcom.utils.common import (
    CTX_DIR,
    DATA_DIR,
    REPO_DIR,
    date_str,
    pwv_interp,
    save_pickle,
)

PLANETS = ("mars", "saturn", "uranus", "neptune")


def _make_parser() -> ap.ArgumentParser:
    """
    Make the argument parser.

    Returns
    -------
    parser : ap.ArgumentParser
        The parser.
    """
    parser = ap.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datadir",
        "-dd",
        # Nominal SO commissioning used
        # /so/home/saianeesh/data/beams/lat_old/source_maps/pointing_model/fits/
        default="/global/cfs/cdirs/sobs/users/skh/data/beams/lat/pointing_model_atm_relcal/",
        help="Directory containing beam_pars.h5 and the per planet residual maps",
    )
    parser.add_argument(
        "--ctx",
        "-c",
        default=str(CTX_DIR / "abscal_ctx_09242026.yaml"),
        help="Path to context file",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=str(REPO_DIR / "abscals"),
        help="Directory to save results to",
    )
    parser.add_argument(
        "--no_results",
        "-nr",
        action="store_true",
        help="Don't save the abscal results",
    )
    parser.add_argument(
        "--skip_planets",
        "-sp",
        nargs="*",
        default=["saturn", "neptune"],
        help="Planets to skip during abscal calculation",
    )
    return parser


def _passes_beam_cuts(fwhm: float, omega_beam: float, band: str, ufm: str) -> bool:
    """
    Check the fitted FWHM and beam volume are within the expected range.

    Parameters
    ----------
    fwhm : float
        Fitted FWHM in arcmin.
    omega_beam : float
        Beam solid angle in sr.
    band : str
        Band of the observation.
    ufm : str
        UFM of the observation.

    Returns
    -------
    passes : bool
        True if both cuts pass.
    """
    for val, (lo, hi) in (
        (fwhm, au.fwhm_cuts[band]),
        (omega_beam, au.beam_volume_cuts[band]),
    ):
        if not lo <= val <= hi:
            print(lo, val, hi, band, ufm)
            return False
    return True


def main() -> None:
    """Compute abscals for every beam fit passing cuts and save the results."""
    args = _make_parser().parse_args()

    with open(DATA_DIR / "atmosphere_eff.pk", "rb") as f:
        atmosphere_eff = pk.load(f)
    pwv = pwv_interp()
    ctx = core.Context(args.ctx)

    with h5py.File(os.path.join(args.datadir, "beam_pars.h5"), mode="r") as f:
        amans, obs_ids, stream_ids, bands = au.load_amans(f)

    cal_dict = {}
    for aman, full_obs_id, stream_id, band in zip(amans, obs_ids, stream_ids, bands):
        obs_id = full_obs_id.split("_")[1]
        ufm = stream_id.split("_")[1] if "ufm" in stream_id else stream_id
        band = band[1:]

        # Get beam pars
        fitted_fwhm = aman.data_fwhm.to(u.arcmin).value
        omega_beam = aman.data_solid_angle_corr.value
        amp = aman.amp.value + (aman.amp_outer.value if "amp_outer" in aman else 0)

        # First cuts are on FWHM and beam volume
        if not _passes_beam_cuts(fitted_fwhm, omega_beam, band, ufm):
            continue

        # Get planet
        try:
            tags = ctx.obsdb.get(full_obs_id, tags=True)["tags"]
        except LoaderError:
            continue
        planet = next((p for p in PLANETS if p in tags), None)
        if planet is None:
            print(f"Error: no planet in tags: {tags}")
            continue
        if planet in args.skip_planets:
            continue

        # Third cut is on RMSE
        resid_name = f"{full_obs_id}_{ufm}_f{band}_full_resid.fits"
        resid_path = os.path.join(
            args.datadir, planet, obs_id[:5], full_obs_id, resid_name
        )
        try:
            rmse = au.get_resid_rmse(resid_path, band)
        except FileNotFoundError:
            continue
        if rmse > 0.05:
            print(f"RMSE = {rmse} > 0.05")
            continue

        try:
            pwv_obs = pwv(obs_id)
        except ValueError:
            print(f"obs {obs_id} outside of pwv range")
            continue
        if np.isnan(pwv_obs) or pwv_obs > 3.0:
            continue

        # Now load the metadata after cuts
        meta = ctx.get_meta(full_obs_id)
        el_obs = meta.obs_info.el_center

        abscal, opt_eff, raw_abscal, raw_opt_eff = au.get_single_abscal(
            amp=amp,
            planet=planet,
            timestamp=obs_id,
            band=band,
            ufm=ufm,
            el_obs=el_obs,
            omega_beam=omega_beam,
            pwv_obs=pwv_obs,
            atmosphere_eff=atmosphere_eff,
        )
        if abscal is None:
            continue
        if raw_abscal >= 40 and planet == "saturn":
            # Some of the saturn observations are accidentally of Neptune, leading to
            # very high abscals (when using Saturn temp). Since the Neptune amp is
            # >10x lower, a cut on the abscal is safe until there is a real fix.
            continue

        cal_dict[f"{ufm}_{band}_{obs_id}"] = {
            "adj_cal": abscal,
            "raw_cal": raw_abscal,
            "pwv": pwv_obs,
            "el": el_obs,
            "omega_data": omega_beam,
            "fwhm": fitted_fwhm,
            "raw_opt": raw_opt_eff,
            "cal_opt": opt_eff,
            "source": planet,
            "time": obs_id,
            "relcal": au.get_relcal(meta=meta, ufm=ufm, band=band),
        }

    if args.no_results:
        return

    date = date_str()
    result_dict = au.make_results_dict(cal_dict=cal_dict)
    save_pickle(result_dict, os.path.join(args.outdir, f"results_{date}.pk"))
    save_pickle(cal_dict, os.path.join(args.outdir, f"abscals_{date}.pk"))

    db = au.make_db(
        result_dict=result_dict, h5_path=os.path.join(args.outdir, "abscals.h5")
    )
    db.to_file(os.path.join(args.outdir, f"db_{date}.sqlite"))


if __name__ == "__main__":
    main()
