"""Shared SIMBAD coordinate resolution for stream catalogs."""

import pandas as pd
from astropy.coordinates import SkyCoord


def _resolve_name(name: str) -> tuple[float | None, float | None]:
    """Resolve a single galaxy name to (ra, dec) in degrees via SIMBAD."""
    # Fix double-dash naming convention (from 2104_06071)
    name_fixed = name.replace("--", "-")

    try:
        coord = SkyCoord.from_name(name_fixed)
        return coord.ra.deg, coord.dec.deg
    except Exception as e:
        print(f"    Warning: Could not resolve '{name}' (tried '{name_fixed}'): {e}")
        return None, None


def resolve_coordinates(
    df: pd.DataFrame,
    name_column: str,
    output_ra_col: str = "ra",
    output_dec_col: str = "dec",
) -> pd.DataFrame:
    """Add RA/Dec columns to a DataFrame by resolving names via SIMBAD.

    Caches resolved names to avoid duplicate queries.
    Inserts ra/dec columns right after the name column.
    """
    coord_cache: dict[str, tuple[float | None, float | None]] = {}

    unique_names = df[name_column].unique()
    print(f"  Resolving {len(unique_names)} unique names ({len(df)} total rows)...")

    for i, name in enumerate(sorted(unique_names)):
        print(f"    [{i + 1}/{len(unique_names)}] Resolving {name}...")
        coord_cache[name] = _resolve_name(name)

    ra_vals = []
    dec_vals = []
    for name in df[name_column]:
        ra, dec = coord_cache[name]
        ra_vals.append(ra)
        dec_vals.append(dec)

    # Insert after name column
    name_idx = df.columns.get_loc(name_column)
    result = df.copy()
    result.insert(name_idx + 1, output_ra_col, ra_vals)
    result.insert(name_idx + 2, output_dec_col, dec_vals)

    resolved = sum(1 for v in ra_vals if v is not None)
    print(f"  Resolved {resolved}/{len(df)} rows")

    return result
