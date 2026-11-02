"""
Read-only sweep of the abscal profile fit radius. Reproduces LAT_beams make_abscal's
map selection, profile fit and cuts for a grid of radii (in units of the band's
nominal FWHM), and reports how many maps pass and how consistent their abscals are.
Needs LAT_beams (the lat_beams package and its configs); nothing is written to its
jobdb or data directories.
"""

import argparse as ap
import datetime as dt
import os
import sys
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sotodlib.core import AxisManager, Context
from sotodlib.site_pipeline import jobdb

import latcom.utils.abscal_utils as au
from latcom.utils.common import UFM_BANDS

LF_BANDS = UFM_BANDS["ln"]
MAD_TO_SIGMA = 1.4826


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
        "--beams_dir",
        default=os.path.expanduser("~/dev/LAT_beams"),
        help="LAT_beams checkout (configs and contexts are relative to it)",
    )
    parser.add_argument(
        "--cfg",
        default="configs/make_source_map/abscal.yaml",
        help="make_abscal config, relative to --beams_dir",
    )
    parser.add_argument(
        "--r_min", type=float, default=0.4, help="Smallest radius [FWHM]"
    )
    parser.add_argument(
        "--r_max", type=float, default=1.2, help="Largest radius [FWHM]"
    )
    parser.add_argument("--r_step", type=float, default=0.1, help="Radius step [FWHM]")
    parser.add_argument(
        "--ref_radius", type=float, default=1.0, help="Radius to compare abscals to"
    )
    parser.add_argument(
        "--core_fracs",
        type=float,
        nargs="+",
        default=[0.2, 0.5],
        help="Ratio cuts only use radii where the stack is above this fraction of "
        "peak; each value is a separate cut variant (make_abscal uses 0.2)",
    )
    parser.add_argument(
        "--drop_edge",
        choices=("no", "yes", "both"),
        default="both",
        help="Whether to leave the outermost fit bin out of the ratio cuts "
        "(make_abscal: no); 'both' runs each variant with and without",
    )
    parser.add_argument(
        "--include_off",
        choices=("no", "yes", "both"),
        default="yes",
        help="Whether the template fit has a free offset term (make_abscal: yes); "
        "'both' fits each map both ways",
    )
    parser.add_argument(
        "--rms_drop_edge",
        choices=("no", "yes", "both"),
        default="both",
        help="Whether to leave the outermost fit bin out of the profile diff rms cut "
        "(make_abscal: no); 'both' runs each variant with and without",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only jobs whose split_str contains one of these, e.g. i1+f090 o6",
    )
    parser.add_argument(
        "--out",
        default=os.path.expanduser(
            f"~/dev/LAT_Loading/abscals/radius_sweep_{dt.date.today():%m%d%Y}.csv"
        ),
        help="CSV of per map, per radius results",
    )
    return parser


def load_beams_cfg(beams_dir: str, cfg_path: str) -> tuple[ap.Namespace, Context, str]:
    """
    Load the make_abscal config the same way make_abscal does.

    Parameters
    ----------
    beams_dir : str
        LAT_beams checkout; becomes the working directory.
    cfg_path : str
        Config path, relative to beams_dir.

    Returns
    -------
    cfg : ap.Namespace
        The configuration.
    ctx : Context
        The context make_abscal uses.
    data_dir : str
        The LAT_beams data directory.
    """
    from lat_beams.utils import get_args_cfg, setup_cfg, setup_paths

    os.chdir(beams_dir)
    sys.argv = ["make_abscal.py", cfg_path]
    args, cfg_dict = get_args_cfg()
    cfg, _ = setup_cfg(
        args,
        cfg_dict,
        {"map_mask_size": "mask_size", "abscal_source_list": "source_list"},
    )
    _, data_dir = setup_paths(
        cfg.root_dir,
        "beams",
        cfg.tel,
        f"{cfg.pointing_type}{(cfg.append != '') * '_'}{cfg.append}"
        f"{cfg.single_det * '_single_det'}",
    )
    return cfg, Context(cfg.ctx_path), data_dir


def db_split(split_str: str) -> str:
    """
    The split the abscal db is built from for a band: LF from lf_only, else mainsrc.

    Parameters
    ----------
    split_str : str
        Job split string, ending in the band.

    Returns
    -------
    split : str
        Split used for the db.
    """
    band = split_str.rsplit("+", maxsplit=1)[-1]
    return "tube_slot+lf_only+band" if band in LF_BANDS else "mainsrc+tube_slot+band"


def tube_from_split_str(split_str: str) -> str:
    """
    Get the tube from a tube_slot split string.

    Parameters
    ----------
    split_str : str
        e.g. "mainsrc+i1+f090" or "o6+lf_only+f030".

    Returns
    -------
    tube : str
        e.g. "i1" or "o6".
    """
    parts = split_str.split("+")
    return parts[0] if parts[1] == "lf_only" else parts[1]


def select_jobs(
    jdb: jobdb.JobManager, cfg: ap.Namespace, only: list[str] | None
) -> list[jobdb.Job]:
    """
    Get the make_abscal jobs whose results go into the abscal db.

    Parameters
    ----------
    jdb : jobdb.JobManager
        The LAT_beams jobdb.
    cfg : ap.Namespace
        The configuration, for the epochs.
    only : list[str] | None
        If given, only jobs whose split_str contains one of these.

    Returns
    -------
    jobs : list[jobdb.Job]
        The jobs.
    """
    epochs = {(float(a), float(b)) for a, b in cfg.epochs}
    return [
        job
        for job in jdb.get_jobs(jclass="make_abscal")
        if job.tags["det_split"] == "full"
        and job.tags["split"] == db_split(job.tags["split_str"])
        and (float(job.tags["epoch_start"]), float(job.tags["epoch_end"])) in epochs
        and (not only or any(s in job.tags["split_str"] for s in only))
    ]


def load_fits(
    jdb: jobdb.JobManager,
    cfg: ap.Namespace,
    ctx: Context,
    data_dir: str,
    jobs: list[jobdb.Job],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the map fits and apply make_abscal's global cuts. Fit jobs are first
    restricted to the tubes and bands of the selected jobs, to save loading time.

    Parameters
    ----------
    jdb : jobdb.JobManager
        The LAT_beams jobdb.
    cfg : ap.Namespace
        The configuration.
    ctx : Context
        Context for obsdb lookups.
    data_dir : str
        The LAT_beams data directory.
    jobs : list[jobdb.Job]
        The selected abscal jobs.

    Returns
    -------
    fits : np.ndarray
        Structured array of fits passing the global cuts.
    fjobs : np.ndarray
        The matching fit_map jobs.
    """
    from lat_beams import beam_utils as bu

    tubes = {tube_from_split_str(j.tags["split_str"]) for j in jobs}
    bands = {j.tags["split_str"].split("+")[-1] for j in jobs}
    fjobs = [
        job
        for job in jdb.get_jobs(jclass="fit_map", jstate="done")
        if job.tags["source"] in cfg.source_list
        and job.tags["split"] == "full"
        and job.tags["band"] in bands
        and any(f"lat{t}_" in job.tags["obs_id"] for t in tubes)
    ]
    fits = bu.load_beam_fits_from_jobs(os.path.join(data_dir, "beam_pars.h5"), fjobs)

    # make_abscal's global cuts
    snr = bu.get_fit_vec(fits, "amp") / bu.get_fit_vec(fits, "noise")
    solid_angle = bu.get_fit_vec(fits, "gauss.data_solid_angle_corr")
    fwhm_exp = np.array([cfg.nominal_fwhm[b] for b in fits["band"]]) * u.arcmin
    data_fwhm = bu.get_fit_vec(fits, "data_fwhm")
    msk = snr > cfg.min_stack_snr
    msk *= data_fwhm < 1.5 * fwhm_exp
    msk *= data_fwhm > 0.5 * fwhm_exp
    msk *= solid_angle > 0
    pwv = bu.get_split_vec(fits, "pwv_mean", ctx)
    pwv[pwv == "None"] = "1"
    el = np.deg2rad(np.array(bu.get_split_vec(fits, "el_center", ctx), float))
    msk *= np.array(pwv, float) / np.sin(el) <= cfg.max_pwv
    return fits[msk], np.array(fjobs)[msk]


def cut_reason(
    prof_norm: np.ndarray,
    stack_prof: np.ndarray,
    cfg: ap.Namespace,
    core_frac: float,
    drop_edge: bool,
    rms_drop_edge: bool,
) -> str | None:
    """
    Apply make_abscal's profile cuts.

    Parameters
    ----------
    prof_norm : np.ndarray
        The map's normalized profile over the fit radii.
    stack_prof : np.ndarray
        The fitted stack profile over the same radii.
    cfg : ap.Namespace
        The configuration, for the cut thresholds.
    core_frac : float
        Ratio cuts only use radii where the stack is above this fraction of peak.
    drop_edge : bool
        Leave the outermost fit bin out of the ratio cuts.
    rms_drop_edge : bool
        Leave the outermost fit bin out of the profile diff rms cut.

    Returns
    -------
    reason : str | None
        The first cut the map fails, or None if it passes.
    """
    core = stack_prof > core_frac * np.max(stack_prof)
    if drop_edge:
        core[-1] = False
    if not core.any():
        return "no core bins"
    prof_ratio = prof_norm / stack_prof
    prof_diff = prof_norm - stack_prof
    if np.abs(np.mean(prof_ratio[core])) > cfg.abscal_max_avg_prat:
        return "avg prof ratio"
    if np.max(np.abs(prof_ratio[core] - 1)) > cfg.abscal_max_pdiff:
        return "max prof ratio"
    if abs(prof_diff[0]) > cfg.abscal_max_pdiff:
        return "first prof diff"
    rms_diff = prof_diff[:-1] if rms_drop_edge and len(prof_diff) > 1 else prof_diff
    if np.std(rms_diff) > cfg.abscal_max_perr:
        return "prof diff rms"
    return None


def sweep_job(
    job: jobdb.Job,
    fits: np.ndarray,
    fjobs: np.ndarray,
    split_vec: np.ndarray,
    cfg: ap.Namespace,
    ctx: Context,
    data_dir: str,
    radii: np.ndarray,
    variants: list[tuple[float, bool, bool]],
    include_offs: list[bool],
) -> list[dict]:
    """
    Fit every map of one abscal job at each radius and apply each cut variant.

    Parameters
    ----------
    job : jobdb.Job
        The abscal job.
    fits : np.ndarray
        Fits passing the global cuts.
    fjobs : np.ndarray
        The matching fit_map jobs.
    split_vec : np.ndarray
        The job's split label of every fit.
    cfg : ap.Namespace
        The configuration.
    ctx : Context
        Context for obsdb lookups.
    data_dir : str
        The LAT_beams data directory.
    radii : np.ndarray
        Fit radii in units of the nominal FWHM.
    variants : list[tuple[float, bool, bool]]
        Cut variants, (core_frac, drop_edge, rms_drop_edge); see cut_reason.
    include_offs : list[bool]
        Template fit settings: whether the fit has a free offset term.

    Returns
    -------
    rows : list[dict]
        One row per map, radius and cut variant.
    """
    from lat_beams import beam_utils as bu

    tags = job.tags
    smsk = (
        (split_vec == tags["split_str"])
        * (fits["split"] == tags["det_split"])
        * (fits["time"] >= float(tags["epoch_start"]))
        * (fits["time"] < float(tags["epoch_end"]))
    )
    sfits, sfjobs = fits[smsk], fjobs[smsk]
    if len(sfits) == 0:
        return []
    band = tags["split_str"].split("+")[-1]
    epoch = f"{float(tags['epoch_start']):.0f}_{float(tags['epoch_end']):.0f}"
    pwvs = [
        None if p == "None" else float(p)
        for p in bu.get_split_vec(sfits, "pwv_mean", ctx)
    ]
    els = np.asarray(bu.get_split_vec(sfits, "el_center", ctx, round_to=6), dtype=float)

    stack_name = f"{tags['det_split']}_{tags['epoch_start']}_{tags['epoch_end']}"
    stack_fit = AxisManager.load(
        os.path.join(data_dir, "stacks", "beam_pars.h5"),
        os.path.join(tags["split"], tags["split_str"], stack_name),
    )
    solid_angle = u.Quantity(stack_fit.bessel.model_solid_angle_true, u.sr).value
    prof = AxisManager.load(
        os.path.join(
            data_dir,
            "stack_profiles",
            tags["split"],
            tags["split_str"],
            f"beam_profiles_{tags['split_str']}_{stack_name}.h5",
        )
    ).prof_cov
    prof_interp = PchipInterpolator(np.deg2rad(np.asarray(prof.r) / 3600), prof.profile)

    rows = []
    for fit, fjob, pwv, el in zip(sfits, sfjobs, pwvs, els):
        base = {
            "split_str": tags["split_str"],
            "epoch": epoch,
            "obs_id": fjob.tags["obs_id"],
            "array": fjob.tags["array"],
            "band": band,
            "source": fjob.tags["source"],
        }
        # abscal is proportional to 1/amp, so get the rest of it once per map
        unit_abscal = np.nan
        status = None
        if pwv is None:
            status = "no pwv"
        else:
            try:
                res = au.get_single_abscal(
                    amp=1.0,
                    planet=fjob.tags["source"],
                    timestamp=fit["time"],
                    band=fit["band"],
                    ufm=fjob.tags["array"],
                    el_obs=el,
                    solid_angle=solid_angle,
                    pwv_obs=pwv,
                )
                unit_abscal = res[0] if res[0] is not None else np.nan
                if res[0] is None:
                    status = "no abscal"
            except Exception as e:  # noqa: BLE001 - record and keep sweeping
                status = f"abscal error: {type(e).__name__}"

        aman = fit["aman"]
        r = aman.r.to(u.rad).value
        mprof = aman.rprof.to(u.pW).value - aman[aman.final_model].off.to(u.pW).value
        init_norm = (
            aman.gauss.amp.to(u.pW).value
            + aman.gauss.off.to(u.pW).value
            - aman[aman.final_model].off.to(u.pW).value
        )
        res_r = np.mean(np.diff(r))
        for radius, include_off in ((r_, o_) for r_ in radii for o_ in include_offs):
            row = {
                **base,
                "radius": round(float(radius), 3),
                "include_off": include_off,
                "amp": np.nan,
                "abscal": np.nan,
            }
            if status is None:
                ext_rad = np.deg2rad(radius * cfg.nominal_fwhm[band] / 60)
                r_msk = (r < ext_rad) * (r >= res_r)
                try:
                    amp, off, _, _, _, _, stack_prof = bu.norm_prof_to_ref(
                        r, mprof / init_norm, prof_interp, ext_rad, include_off
                    )
                    amp *= init_norm
                    off *= init_norm
                    prof_norm = (mprof[r_msk] - off) / amp
                    row.update(
                        n_bins=int(r_msk.sum()), amp=amp, abscal=unit_abscal / amp
                    )
                except (ValueError, TypeError, np.linalg.LinAlgError) as e:
                    fit_status = f"fit failed: {type(e).__name__}"
            for core_frac, drop_edge, rms_drop_edge in variants:
                if status is not None:
                    variant_status = status
                elif "n_bins" not in row:
                    variant_status = fit_status
                else:
                    reason = cut_reason(
                        prof_norm, stack_prof, cfg, core_frac, drop_edge, rms_drop_edge
                    )
                    variant_status = reason or "pass"
                rows.append(
                    {
                        **row,
                        "core_frac": core_frac,
                        "drop_edge": drop_edge,
                        "rms_drop_edge": rms_drop_edge,
                        "status": variant_status,
                    }
                )
    return rows


def rel_scatter(x: np.ndarray) -> float:
    """
    Robust relative scatter, 1.4826 * MAD / median.

    Parameters
    ----------
    x : np.ndarray
        Values.

    Returns
    -------
    scatter : float
        Relative scatter.
    """
    med = np.median(x)
    return MAD_TO_SIGMA * np.median(np.abs(x - med)) / abs(med)


def summarize(df: pd.DataFrame, ref_radius: float) -> pd.DataFrame:
    """
    Summarize the sweep per cut variant, band and radius.

    Parameters
    ----------
    df : pd.DataFrame
        Per map, per radius results from sweep_job.
    ref_radius : float
        Radius to compare abscals to.

    Returns
    -------
    summary : pd.DataFrame
        One row per (include_off, core_frac, drop_edge, rms_drop_edge, band, radius):
        maps passing,
        UFM/epoch groups covered,
        median within-group scatter and uncertainty on the group mean,
        and the median abscal ratio to the reference radius on common maps.
    """
    keys = ["obs_id", "array", "band"]
    rows = []
    variant_cols = ["include_off", "core_frac", "drop_edge", "rms_drop_edge"]
    for (*variant_key, band, radius), sub in df.groupby(
        [*variant_cols, "band", "radius"]
    ):
        in_variant = np.all(
            [df[c] == v for c, v in zip(variant_cols, variant_key)], axis=0
        )
        variant = df[in_variant & (df.status == "pass")]
        ref = variant[np.isclose(variant.radius, ref_radius)].set_index(keys).abscal
        ok = sub[sub.status == "pass"]
        groups = [g.abscal.to_numpy() for _, g in ok.groupby(["array", "epoch"])]
        big = [g for g in groups if len(g) >= 3]
        scatter = [rel_scatter(g) for g in big]
        sem = [s / np.sqrt(len(g)) for s, g in zip(scatter, big)]
        common = ok.set_index(keys).abscal
        ratio = (common / ref).dropna()
        rows.append(
            {
                **dict(zip(variant_cols, variant_key)),
                "band": band,
                "radius": radius,
                "maps": len(sub),
                "pass": len(ok),
                "groups_covered": len(groups),
                "scatter_med": np.median(scatter) if scatter else np.nan,
                "sem_med": np.median(sem) if sem else np.nan,
                "ratio_to_ref": np.median(ratio) if len(ratio) else np.nan,
                "ratio_16_84": (
                    f"{np.percentile(ratio, 16):.3f}-{np.percentile(ratio, 84):.3f}"
                    if len(ratio)
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the sweep, save the per map results and print the summaries."""
    args = _make_parser().parse_args()
    # Small radii leave few bins, which makes polyfit warn on every map
    warnings.simplefilter("ignore", np.exceptions.RankWarning)
    out = Path(args.out).expanduser().resolve()
    cfg, ctx, data_dir = load_beams_cfg(args.beams_dir, args.cfg)
    radii = np.round(
        np.arange(args.r_min, args.r_max + args.r_step / 2, args.r_step), 3
    )
    choice = {"no": [False], "yes": [True], "both": [False, True]}
    variants = [
        (c, e, re)
        for c in args.core_fracs
        for e in choice[args.drop_edge]
        for re in choice[args.rms_drop_edge]
    ]

    from lat_beams import beam_utils as bu

    jdb = jobdb.JobManager(sqlite_file=os.path.join(data_dir, "jobdb.db"))
    jobs = select_jobs(jdb, cfg, args.only)
    print(f"{len(jobs)} abscal jobs, radii {radii.tolist()} FWHM", flush=True)
    fits, fjobs = load_fits(jdb, cfg, ctx, data_dir, jobs)
    print(f"{len(fits)} fits after the global cuts", flush=True)
    split_vecs = {
        split: bu.get_split_vec(fits, split, ctx, metasplits=cfg.metasplits)
        for split in {j.tags["split"] for j in jobs}
    }

    rows = []
    for i, job in enumerate(jobs):
        try:
            rows += sweep_job(
                job,
                fits,
                fjobs,
                split_vecs[job.tags["split"]],
                cfg,
                ctx,
                data_dir,
                radii,
                variants,
                choice[args.include_off],
            )
        except (OSError, KeyError) as e:
            print(
                f"  skipping {job.tags['split_str']} {job.tags['epoch_start']}: {e!r}"
            )
        print(
            f"[{i + 1}/{len(jobs)}] {job.tags['split_str']} {job.tags['epoch_start']}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nPer map results: {out}")

    pd.set_option("display.width", 200)
    print("\nStatus counts by cut variant and radius (all bands):")
    print(
        df.pivot_table(
            index=["include_off", "core_frac", "drop_edge", "rms_drop_edge", "radius"],
            columns="status",
            aggfunc="size",
            fill_value=0,
        )
    )
    print(
        f"\nPer band summary (scatter/sem are relative; ratio_to_ref vs {args.ref_radius} FWHM):"
    )
    summary = summarize(df, args.ref_radius)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
