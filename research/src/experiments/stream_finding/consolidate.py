"""Consolidate all stream catalogs into a single CSV with (ra, dec, source)."""

import csv
from pathlib import Path

import pandas as pd


# Column mappings for each source's streams.csv
SOURCES = {
    "1301_4275": {"ra": "ra", "dec": "dec"},
    "1803_05447": {"ra": "ra", "dec": "dec"},
    "1807_07195": {"ra": "ra", "dec": "dec"},
    "2104_06071": {"ra": "ra", "dec": "dec"},
    "2209_08636": {"ra": "ra", "dec": "dec"},
    "2406_13496": {"ra": "RA_deg", "dec": "Dec_deg"},
    "2407_20483": {"ra": "RA_deg", "dec": "Dec_deg"},
    "2502_14531": {"ra": "RA_deg", "dec": "Dec_deg"},
    "2503_18480": {"ra": "ra", "dec": "dec"},
    "2508_02154": {"ra": "RA", "dec": "DEC"},
    "2509_18274": {"ra": "ra", "dec": "dec"},
}


def consolidate(data_root: Path) -> pd.DataFrame:
    streams_dir = data_root / "streams"
    output_file = streams_dir / "consolidated.csv"

    all_entries = []

    print("Consolidating streams from:")

    # Paper catalogs
    for source_name, col_map in SOURCES.items():
        filepath = streams_dir / source_name / "streams.csv"
        print(f"  {filepath}")

        if not filepath.exists():
            print(f"    WARNING: File not found, skipping")
            continue

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                try:
                    ra = float(row[col_map["ra"]])
                    dec = float(row[col_map["dec"]])
                    all_entries.append({
                        "ra": ra,
                        "dec": dec,
                        "source": source_name,
                    })
                    count += 1
                except (ValueError, KeyError):
                    pass
            print(f"    -> {count} entries")

    # mystreams.csv
    mystreams_path = streams_dir / "mystreams.csv"
    if mystreams_path.exists():
        print(f"  {mystreams_path}")
        with open(mystreams_path, "r") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                try:
                    ra = float(row["ra"])
                    dec = float(row["dec"])
                    all_entries.append({
                        "ra": ra,
                        "dec": dec,
                        "source": "mystreams",
                    })
                    count += 1
                except (ValueError, KeyError):
                    pass
            print(f"    -> {count} entries")

    # Remove exact duplicates by (ra, dec)
    seen = set()
    unique_entries = []
    for entry in all_entries:
        coord_key = (entry["ra"], entry["dec"])
        if coord_key not in seen:
            seen.add(coord_key)
            unique_entries.append(entry)

    duplicates_removed = len(all_entries) - len(unique_entries)
    print(f"\nRemoved {duplicates_removed} exact duplicates")

    # Write consolidated CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ra", "dec", "source"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(unique_entries)

    print(f"Wrote {len(unique_entries)} total entries to {output_file}")

    return pd.DataFrame(unique_entries)
