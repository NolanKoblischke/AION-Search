"""Analysis scripts for stream catalogs.

- check_mystreams: 30 arcsec matching of mystreams against all reference catalogs
- analyze_duplicates: 30 arcsec cross-paper duplicate finding
"""

import csv
from pathlib import Path

from astropy.coordinates import SkyCoord
import astropy.units as u


def _load_csv_with_coords(
    csv_path: Path, ra_col: str, dec_col: str, name_col: str | None = None
) -> list[tuple[str, float, float, dict]]:
    """Load CSV and return list of (name, ra, dec, row) tuples."""
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ra = float(row[ra_col])
                dec = float(row[dec_col])
                name = row.get(name_col, "Unknown") if name_col else "Unknown"
                entries.append((name, ra, dec, row))
            except (ValueError, KeyError):
                pass
    return entries


# Column mappings for each catalog
_CATALOGS = {
    "1301_4275": {"ra": "ra", "dec": "dec", "name": "CFHTLS"},
    "1803_05447": {"ra": "ra", "dec": "dec", "name": "RESOLVE"},
    "1807_07195": {"ra": "ra", "dec": "dec", "name": "SimbadName"},
    "2104_06071": {"ra": "ra", "dec": "dec", "name": "Name"},
    "2209_08636": {"ra": "ra", "dec": "dec", "name": "Host"},
    "2406_13496": {"ra": "RA_deg", "dec": "Dec_deg", "name": "Name"},
    "2407_20483": {"ra": "RA_deg", "dec": "Dec_deg", "name": "Host"},
    "2502_14531": {"ra": "RA_deg", "dec": "Dec_deg", "name": None},
    "2503_18480": {"ra": "ra", "dec": "dec", "name": "Galaxy"},
    "2508_02154": {"ra": "RA", "dec": "DEC", "name": "Name"},
    "2509_18274": {"ra": "ra", "dec": "dec", "name": None},
}


def check_mystreams(data_root: Path) -> None:
    """Check if mystreams.csv entries match any catalog within 30 arcsec."""
    streams_dir = data_root / "streams"
    max_sep = 30  # arcsec

    mystreams = _load_csv_with_coords(
        streams_dir / "mystreams.csv", ra_col="ra", dec_col="dec"
    )
    print(f"mystreams.csv: {len(mystreams)} entries\n")

    # Load all catalogs
    loaded_catalogs = {}
    for cat_name, config in _CATALOGS.items():
        cat_file = streams_dir / cat_name / "streams.csv"

        if cat_file.exists():
            entries = _load_csv_with_coords(
                cat_file,
                ra_col=config["ra"],
                dec_col=config["dec"],
                name_col=config["name"],
            )
            loaded_catalogs[cat_name] = entries
            print(f"  {cat_name}: {len(entries)} entries")

    my_coords = SkyCoord(
        ra=[e[1] for e in mystreams] * u.deg,
        dec=[e[2] for e in mystreams] * u.deg,
    )

    print(f"\n{'=' * 80}")
    print(f"Checking mystreams.csv against all catalogs (within {max_sep} arcsec)")
    print("=" * 80)

    matched_count = 0
    unmatched = []

    for i, (my_name, my_ra, my_dec, my_row) in enumerate(mystreams):
        matches = []

        for cat_name, cat_entries in loaded_catalogs.items():
            if not cat_entries:
                continue

            cat_coords = SkyCoord(
                ra=[e[1] for e in cat_entries] * u.deg,
                dec=[e[2] for e in cat_entries] * u.deg,
            )

            seps = my_coords[i].separation(cat_coords)
            min_idx = seps.argmin()
            min_sep = seps[min_idx].arcsec

            if min_sep <= max_sep:
                match_entry = cat_entries[min_idx]
                matches.append((cat_name, match_entry[0], min_sep, match_entry[3]))

        print(f"\n[{i + 1}] RA={my_ra:.4f}, Dec={my_dec:.4f}")

        if matches:
            matched_count += 1
            for cat_name, match_name, sep, row in sorted(matches, key=lambda x: x[2]):
                extra = ""
                if "Majority feature" in row:
                    extra = f" [{row['Majority feature']}]"
                elif "Type" in row:
                    extra = f" [{row['Type']}]"
                elif "Morphology" in row:
                    extra = f" [{row['Morphology']}]"
                print(
                    f"    {cat_name:25s}: {match_name:25s} ({sep:6.1f}\"){extra}"
                )
        else:
            unmatched.append((i + 1, my_ra, my_dec))
            print(f"    No matches within {max_sep} arcsec")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"\nMatched: {matched_count}/{len(mystreams)}")
    print(f"Unmatched: {len(unmatched)}/{len(mystreams)}")

    if unmatched:
        print("\nUnmatched entries (potential new discoveries):")
        for idx, ra, dec in unmatched:
            print(f"  [{idx}] RA={ra:.4f}, Dec={dec:.4f}")


def analyze_duplicates(data_root: Path) -> None:
    """Find cross-paper duplicates within 30 arcsec."""
    streams_dir = data_root / "streams"

    all_entries = []  # (paper_id, name, ra, dec)

    for paper_id, config in _CATALOGS.items():
        csv_file = streams_dir / paper_id / "streams.csv"
        if not csv_file.exists():
            print(f"  {paper_id}: No streams.csv found, skipping")
            continue

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            ra_col = config["ra"]
            dec_col = config["dec"]
            if ra_col not in fieldnames or dec_col not in fieldnames:
                print(f"  {paper_id}: No {ra_col}/{dec_col} columns, skipping")
                continue

            name_col = config["name"]
            rows = list(reader)
            count = 0
            for row in rows:
                try:
                    ra = float(row[ra_col])
                    dec = float(row[dec_col])
                    name = row.get(name_col, "Unknown") if name_col else "Unknown"
                    all_entries.append((paper_id, name, ra, dec))
                    count += 1
                except (ValueError, KeyError):
                    pass
            print(f"  {paper_id}: {count} entries")

    print(f"\nTotal entries across all papers: {len(all_entries)}")

    coords = SkyCoord(
        ra=[e[2] for e in all_entries] * u.deg,
        dec=[e[3] for e in all_entries] * u.deg,
    )

    print("Searching for cross-paper duplicates within 30 arcsec...")

    duplicates = []
    seen_pairs = set()

    for i in range(len(all_entries)):
        for j in range(i + 1, len(all_entries)):
            if all_entries[i][0] == all_entries[j][0]:
                continue

            sep = coords[i].separation(coords[j])
            if sep.arcsec <= 30:
                pair_key = tuple(sorted([i, j]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    duplicates.append((all_entries[i], all_entries[j], sep.arcsec))

    print(f"\nFound {len(duplicates)} cross-paper duplicates (within 30 arcsec):")
    print("-" * 80)

    for e1, e2, sep in sorted(duplicates, key=lambda x: x[2]):
        paper1, name1, ra1, dec1 = e1
        paper2, name2, ra2, dec2 = e2
        print(f"  {name1} ({paper1}) <-> {name2} ({paper2})")
        print(
            f"    Sep: {sep:.2f} arcsec | "
            f"RA/Dec: ({ra1:.4f}, {dec1:.4f}) vs ({ra2:.4f}, {dec2:.4f})"
        )

    # Group duplicates by approximate coordinates
    coord_groups = []
    for e1, e2, sep in duplicates:
        found_group = None
        for group in coord_groups:
            for entry in group:
                if entry[0] == e1[0] and entry[1] == e1[1]:
                    found_group = group
                    break
                if entry[0] == e2[0] and entry[1] == e2[1]:
                    found_group = group
                    break
            if found_group:
                break

        if found_group:
            if (e1[0], e1[1]) not in [(e[0], e[1]) for e in found_group]:
                found_group.append(e1)
            if (e2[0], e2[1]) not in [(e[0], e[1]) for e in found_group]:
                found_group.append(e2)
        else:
            coord_groups.append([e1, e2])

    print(f"\n{'-' * 80}")
    print("Summary:")
    print(f"  Total entries: {len(all_entries)}")
    print(f"  Cross-paper duplicate pairs: {len(duplicates)}")
    print(f"  Unique objects appearing in multiple papers: {len(coord_groups)}")

    paper_overlap = {}
    for e1, e2, sep in duplicates:
        key = tuple(sorted([e1[0], e2[0]]))
        paper_overlap[key] = paper_overlap.get(key, 0) + 1

    if paper_overlap:
        print("\nPaper overlap counts:")
        for (p1, p2), count in sorted(paper_overlap.items(), key=lambda x: -x[1]):
            print(f"  {p1} <-> {p2}: {count} shared objects")
