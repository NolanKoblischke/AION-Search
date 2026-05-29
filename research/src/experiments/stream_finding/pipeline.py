"""Stream finding pipeline.

Usage:
    uv run python src/experiments/stream_finding/pipeline.py                    # run all
    uv run python src/experiments/stream_finding/pipeline.py --paper 2104_06071 # run one
    uv run python src/experiments/stream_finding/pipeline.py --skip-resolve     # skip SIMBAD
    uv run python src/experiments/stream_finding/pipeline.py --analyze          # include analysis
"""

import argparse
from pathlib import Path

from src.experiments.stream_finding.extract import get_extractor, get_all_extractors
from src.experiments.stream_finding.extract.extract_2508_02154 import Extract2508_02154
from src.experiments.stream_finding.resolve import resolve_coordinates
from src.experiments.stream_finding.consolidate import consolidate
from src.experiments.stream_finding.analyze import check_mystreams, analyze_duplicates


DATA_ROOT = Path("data")


def run_extractor(extractor, data_root: Path, skip_resolve: bool = False) -> None:
    """Run a single extractor: extract, optionally resolve, save output."""
    arxiv_id = extractor.arxiv_id
    source = extractor.source_dir(data_root)
    output = extractor.output_dir(data_root)
    output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing {arxiv_id}")
    print(f"{'=' * 60}")

    # 2508_02154 (STRRINGS) has a custom run method
    if isinstance(extractor, Extract2508_02154):
        extractor.run(data_root)
        return

    # Extract
    df = extractor.extract(source)
    print(f"  Extracted {len(df)} rows")

    if extractor.needs_resolution:
        # Save raw extraction
        raw_path = output / "streams_raw.csv"
        df.to_csv(raw_path, index=False)
        print(f"  Wrote {raw_path}")

        if not skip_resolve:
            # Resolve coordinates via SIMBAD
            df = resolve_coordinates(
                df,
                name_column=extractor.name_column,
                output_ra_col="ra",
                output_dec_col="dec",
            )

        # Save resolved
        streams_path = output / "streams.csv"
        df.to_csv(streams_path, index=False)
        print(f"  Wrote {streams_path}")
    else:
        streams_path = output / "streams.csv"
        df.to_csv(streams_path, index=False)
        print(f"  Wrote {streams_path}")


def main():
    parser = argparse.ArgumentParser(description="Stream finding pipeline")
    parser.add_argument(
        "--paper", type=str, default=None,
        help="Run only for a specific paper (e.g. 2104_06071)",
    )
    parser.add_argument(
        "--skip-resolve", action="store_true",
        help="Skip SIMBAD coordinate resolution",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run analysis (mystreams check + duplicate finding)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=DATA_ROOT,
        help="Path to data root directory",
    )
    args = parser.parse_args()

    # Step 1: Extract
    if args.paper:
        extractor = get_extractor(args.paper)
        run_extractor(extractor, args.data_root, skip_resolve=args.skip_resolve)
    else:
        for extractor in get_all_extractors():
            run_extractor(extractor, args.data_root, skip_resolve=args.skip_resolve)

    # Step 2: Consolidate
    if not args.paper:
        print(f"\n{'=' * 60}")
        print("Consolidating")
        print(f"{'=' * 60}")
        consolidate(args.data_root)

    # Step 3: Analyze (optional)
    if args.analyze and not args.paper:
        print(f"\n{'=' * 60}")
        print("Analyzing")
        print(f"{'=' * 60}")
        check_mystreams(args.data_root)
        print()
        analyze_duplicates(args.data_root)


if __name__ == "__main__":
    main()
