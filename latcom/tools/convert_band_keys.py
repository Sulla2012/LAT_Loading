"""Rename bare band keys ("090") to f-prefixed ones ("f090") in pickled result dicts."""

import argparse as ap
import os
import shutil
from pathlib import Path

import dill as pk

from latcom.utils.common import BANDS

OLD_BANDS = {band.removeprefix("f"): band for band in BANDS}


def _make_parser() -> ap.ArgumentParser:
    """
    Make the argument parser.

    Returns
    -------
    parser : ap.ArgumentParser
        The parser.
    """
    parser = ap.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Pickles to convert in place")
    parser.add_argument(
        "--backup_dir",
        "-b",
        required=True,
        help="Directory to copy the originals of converted pickles to",
    )
    return parser


def rename_band_keys(obj: object) -> int:
    """
    Rename bare band keys to f-prefixed ones in place, in all nested dicts
    (including dicts inside lists and tuples). Values are not changed.

    Parameters
    ----------
    obj : object
        Object to convert.

    Returns
    -------
    n_renamed : int
        Number of keys renamed.

    Raises
    ------
    ValueError
        If a dict has both a bare and an f-prefixed key for the same band.
    """
    if isinstance(obj, (list, tuple)):
        return sum(
            rename_band_keys(x) for x in obj if isinstance(x, (dict, list, tuple))
        )
    if not isinstance(obj, dict):
        return 0

    n_renamed = sum(key in OLD_BANDS for key in obj)
    if n_renamed:
        if any(OLD_BANDS.get(key) in obj for key in obj):
            raise ValueError(
                f"Dict has both bare and f-prefixed band keys: {list(obj)}"
            )
        # Rebuild in place to keep the key order and the dict type
        items = list(obj.items())
        obj.clear()
        obj.update((OLD_BANDS.get(key, key), val) for key, val in items)
    return n_renamed + sum(rename_band_keys(val) for val in obj.values())


def convert_pickle(path: Path, backup_dir: Path) -> int:
    """
    Convert one pickle in place, backing up the original first.
    Pickles without bare band keys are left untouched.

    Parameters
    ----------
    path : Path
        Pickle to convert.
    backup_dir : Path
        Directory to copy the original to, mirroring its path.

    Returns
    -------
    n_renamed : int
        Number of keys renamed.
    """
    with open(path, "rb") as f:
        obj = pk.load(f)
    n_renamed = rename_band_keys(obj)
    if n_renamed == 0:
        return 0

    backup = backup_dir / path.resolve().relative_to("/")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)

    # Write to a temporary file first so a failure can't leave a partial pickle
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pk.dump(obj, f)
    os.replace(tmp, path)
    return n_renamed


def main() -> None:
    """Convert each pickle and report how many keys were renamed."""
    args = _make_parser().parse_args()
    backup_dir = Path(args.backup_dir)
    for path in map(Path, args.paths):
        try:
            n_renamed = convert_pickle(path, backup_dir)
        except Exception as e:  # noqa: BLE001 - report and move on to the next file
            print(f"FAILED   {path}: {e!r}")
            continue
        status = "renamed " if n_renamed else "no bands"
        print(f"{status} {path}: {n_renamed} keys", flush=True)


if __name__ == "__main__":
    main()
