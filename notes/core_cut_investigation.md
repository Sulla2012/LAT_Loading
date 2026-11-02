# Abscal core-cut investigation log

Investigation of how the profile-shape cuts in `LAT_beams/make_abscal.py` affect which maps
enter the abscal and the resulting per-UFM abscals. Started 2026-10-30.

Tools: `latcom/tools/abscal_radius_sweep.py` (read-only re-run of make_abscal's per-map fit and
cuts for a grid of radii and cut settings). Plots and per-map tables in `plts/`, sweep outputs in
`abscals/radius_sweep*_10302026*`.

## Definitions

- **Fit radius** `abscal_r_fwhm`: rings with r < abscal_r_fwhm × nominal FWHM are used to fit
  map ≈ amp × stack(s·r + Δr) + off (`bu.norm_prof_to_ref`). Rings are ~10″ (one pixel) apart.
- **Ratio cuts** (avg and max |map/template − 1| < 5%) use only rings where the template is above
  `abscal_ratio_core_frac` × its peak ("core").
- **rms cut**: std(map − template) < 2% of peak over the fit rings (`abscal_rms_drop_edge` drops the
  outermost ring).
- **Wing excess**: median(map − fitted template) at 1.3–2.0 FWHM, in % of peak.

## Timeline of findings

1. **0.8 FWHM dip (i1 f090).** With core 0.2, 38/99 maps failed only at 0.8 FWHM: the outermost
   fit ring (91″, template at 22% of peak) was both the least constrained ring and inside the
   ratio-check core. Fix: core 0.5 or drop the edge ring from the ratio check.
2. **Abscal depends on fit radius.** Per-map abscal drifts with radius (f090 −13.5% at 0.4 FWHM,
   +1.5% at 1.2 FWHM vs 1.0); larger at f150/UHF (−17 to −21% at 0.5 FWHM). Small radii also leave
   UHF with 1–3 rings for a 4-parameter fit (UHF has 4–5 rings at 1.0 FWHM).
3. **rms cut has the same edge problem** at UHF (jumpy counts vs radius). Dropping the edge ring
   from the rms helps f150/f280 at 1.0 FWHM; f220's dip at 0.9–1.0 FWHM is the few-rings problem.
4. **Production change (2026-10-30, job 60564032):** core 0.5 + rms edge dropped at 1.0 FWHM.
   Maps with an abscal 663 → 849 (f090 325→338, f150 208→316, f220 46→68, f280 80→122).
5. **But the db abscals shifted at UHF.** vs the old cuts: f090 +0.03%, f150 +0.4%, f220 +2.3%,
   f280 +9.4% median (o1 f280 +25–36%). Decomposition: core 0.2→0.5 drives it (f280 +6.1%),
   rms edge +1.5%, relcal method ~0. Newly admitted maps have higher per-map abscals than kept ones
   in the same UFM/band/epoch (f150 +2.5%, f220 +4.8%, f280 +12.4%).
6. **Core sweep 0.2→0.5** (rms edge dropped): f150 gains at 0.25 with no shift and lower spread;
   f280 shifts +6–9% at 0.30–0.35. o1: 6→17 f220 maps at 0.25; 2→15 f280 maps at 0.35.
7. **o1 beams** (`plts/o1_core_cut_beams.png`): maps only admitted by a looser core show a flat
   +3–10% wing excess beyond ~60″; lower SNR, higher pwv/sin(el), larger fitted FWHM.
8. **Not temporal** (`plts/pedestal_vs_time.png`): high- and low-wing-excess maps coexist on the
   same days in 2025 and 2026. Not source (Mars 58% vs Uranus 59% high at UHF). Partly tube (o1 worst).
9. **Not weather / time of day** (`plts/wing_excess_vs_weather.png`): Spearman |ρ| ≤ 0.15 with pwv,
   el, pwv/sin(el), SNR, solar hour. f150/UHF wing excess is bimodal at every pwv.
10. **Common residual shape in every tube** (`plts/wing_resid_by_tube.png`): +2% center, −1 to −2%
    at 0.4–0.6 FWHM, flat plateau beyond ~1 FWHM (f090 +1–3%, f150 +3–5%, f220 +5–6%, f280 1–7%).
11. **Not the map offset subtraction**: final model is always Bessel; subtracted offsets ~1e-4 of peak.
12. **Cause: the template fit's free offset.** Wing excess vs −(fitted off/amp): Spearman ρ = 0.96–0.998
    in every band. Without the fitted offset the median wing residual is −0.1% of peak in every band.
    The offset is not at its bounds (1–2% of MF/UHF maps). The template is still ~6% of peak at
    1 FWHM, so offset and amplitude are partly degenerate inside the fit radius; the fit uses the
    offset to improve the core match, and it appears as a flat "pedestal" outside. The 0.2 ratio and
    rms cuts were rejecting maps for this fit artifact.

13. **Fixing the fit offset at 0 (`include_off=False`)** (sweep `abscals/radius_sweep3_offset*`,
    1.0 FWHM, rms edge dropped; all numbers from one sweep, same templates):
    - Core 0.2 passes: fixed 331/314/141/135 (f090/f150/f220/f280) vs free 325/213/53/80.
    - Within-group spread (core 0.2): fixed f090 2.8%, f150 2.6%, f280 6.0% vs free 3.5/3.0/6.7%.
    - Radius drift vs 1.0 FWHM over 0.6–2.0: f090 −0.9..+0.4% (free −3.6..+1.7%), f150 −2.8..+0.6%
      (free −9.7..+2.9%). UHF still drifts (f220 ±5%, f280 ~+10% at 0.6 and 2.0 FWHM): few rings,
      stretch/shift still trade with the amplitude.
    - Maps passing core 0.5 but not 0.2: f150 21 maps, now 0.0% vs kept (free: +2.5%); f280 12 maps
      still +12.6%.
    - Per-map abscal fixed/free: f090 −0.15%, f150 −0.8%, f220 −4.3%, f280 −7.5% (median).
    - Candidate production settings: `include_off=False`, core 0.2 (or per band), rms edge dropped.

14. **Production rerun (job 60569472, `include_off=False`, core 0.2, rms edge dropped).** Maps
    331/314/141/135 + 3 f030, identical per map to the sweep. Per-UFM median vs 10/29:
    f090 +0.2%, f150 −1.0%, f220 −4.3%, f280 −5.8%. Latest epoch all finite except LF f040
    (ln2/3/4 NaN): all 14 f040 maps fail "max prof ratio".
15. **LF f040 cause: ring count, not the offset.** Rings are 10″, so 1 FWHM has 30 (f040) / 44 (f030)
    rings vs ~12 (f090) — the per-ring 5% max cut is a noise test at LF. Averaging rings to ~12 per
    FWHM (k = round(n_rings/12); k=1 for f090 and up) passes f040 4/14 (ln3 Jup ×2 24.0/26.0, ln3 Mars
    26.0, ln4 Jup ×1) and f030 5/7. Diagnostic: scratchpad `lf_diag2.py` (session-only).
16. **ln4 f040 maps are ~40× too faint**: per-map abscal 1300–2600 vs ln3 24–32; relcal only 2.13 vs
    1.02. ln2 f040 inconsistent (29/46/630). f030 consistent across ln2/3/4 (130–180). Binning alone
    would let ln4's map poison the tube average (cal×relcal ~4800 vs ~25).
    Proposed (not applied): binned cuts (`abscal_rings_per_fwhm`, default 12) + per tube/band/epoch
    outlier guard on cal×relcal (~3× from median), then rerun. Flag ln4 f040 to LF det-mapping owners.

## Data caveat (2026-10-30)

skh's stack job 60559516 (started 10:39) recomputes the stack covariances and rewrote 10
db-relevant stack profile files (o1–o5, latest epoch) between sweeps 2 and 3. The profiles are
unchanged to numerical precision (median per-map abscal change exactly 1.0000 in every job), so the
production db (job 60564032) and the sweep comparisons are effectively unaffected.
BUT a few maps in those jobs moved by up to ±5–7% (o4/o5 f150 up to +6.9%, o2/o4 f150 down to −6.1%,
o5 f090 +5.3%, o1 f280 +4.0%): the free-offset 4-parameter fit has near-degenerate solutions, so a
negligible template change can flip a map's amplitude. Worth testing fit stability with a template
jitter, offset free vs fixed.
A latcom pwv change (uncommitted `common.py`) also moved 157 maps' (May–Sep 2026) non-fit abscal
factor by −0.5..+0.7% between sweeps 2 and 3.

## Other findings along the way

- 51 maps at elevation > 67.5° had NaN abscals but were kept as rows (`abscal_res is None` bug);
  fixed in make_abscal (`abscal_res[0] is None`).
- 71 maps were taken during the mirror alignment (2025-08-19 → 2025-09-02) and are inside abscal
  epochs: 23 f090 in epoch 1, 48 in epoch 2 (Sep 1–2). Epoch boundaries could move to
  1755561600 / 1756857600 (needs restacking).
- No UHF maps in epoch 1. 41 June-2025 rows have NaN relcal (missing bias groups; tube-mate fallback).
- `make_abscal` used to crash with 0 jobs in `distribute_jobs` (hung 4 h); guarded now.

## Open questions / next steps

- Refit with `include_off=False` (make_abscal originally used this; maps' baselines are already
  subtracted) and redo the sweep: counts, within-group spread, db shifts, and radius dependence.
- If the offset term is needed, constrain it from rings outside the core (or fit it beyond the
  fit radius) instead of letting it trade off with the amplitude.
- Flux-ratio abscal (integrate map and stack to the same radius) as a shape-independent check.
- Decide UHF cut settings only after the offset question is settled.
