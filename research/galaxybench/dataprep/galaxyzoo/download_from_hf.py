#!/usr/bin/env python3
"""Materialize the frozen GalaxyBench HF dataset as the legacy HDF5 input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from datasets import load_dataset


HF_REPO = "astronolan/galaxy-description-benchmark"
HF_REVISION = "ebb13986d04b6b5e47529fb1fc68761839bffd75"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT = Path("galaxybench/dataprep/galaxyzoo/gz5_selected_galaxies.hdf5")
DEFAULT_SUMMARY = Path("galaxybench/dataprep/galaxyzoo/selected_galaxies_summary.json")
SKIP_COLUMNS = {"image_rgb"}


def _flatten_summary_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()

    data = json.loads(summary_path.read_text())
    ids: set[str] = set()
    for path_info in data.get("paths", {}).values():
        ids.update(str(object_id) for object_id in path_info.get("galaxy_ids", []))
    return ids


def _string_dtype(values: list[Any]) -> np.dtype:
    max_len = max((len(str(value).encode("utf-8")) for value in values), default=1)
    return np.dtype(f"S{max(1, max_len)}")


def _column_dtype(name: str, values: list[Any]) -> tuple[str, Any, tuple[int, ...] | None]:
    first_non_null = next((value for value in values if value is not None), None)

    if name == "image_array":
        array = np.asarray(first_non_null, dtype=np.float32)
        return name, np.float32, tuple(array.shape)

    if isinstance(first_non_null, (str, bytes)):
        return name, _string_dtype(values), None

    if isinstance(first_non_null, (bool, np.bool_)):
        return name, np.bool_, None

    if isinstance(first_non_null, (int, np.integer)):
        return name, np.int64, None

    if isinstance(first_non_null, (float, np.floating)):
        return name, np.float64, None

    raise TypeError(f"Unsupported column {name!r} with example value {type(first_non_null)!r}")


def _build_dtype(dataset) -> np.dtype:
    dtype_fields = []
    for name in dataset.column_names:
        if name in SKIP_COLUMNS:
            continue
        values = dataset[name]
        field_name, dtype, shape = _column_dtype(name, values)
        if shape is None:
            dtype_fields.append((field_name, dtype))
        else:
            dtype_fields.append((field_name, dtype, shape))
    return np.dtype(dtype_fields)


def _assign_value(row, name: str, value: Any) -> None:
    if name == "image_array":
        row[name] = np.asarray(value, dtype=np.float32)
        return

    if row.dtype[name].kind == "S":
        row[name] = str(value).encode("utf-8")
        return

    row[name] = value


def materialize(output_path: Path, summary_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    dataset = load_dataset(HF_REPO, split=DEFAULT_SPLIT, revision=HF_REVISION)
    dataset_ids = {str(object_id) for object_id in dataset["object_id"]}
    summary_ids = _flatten_summary_ids(summary_path)
    if summary_ids and dataset_ids != summary_ids:
        missing = sorted(summary_ids - dataset_ids)
        extra = sorted(dataset_ids - summary_ids)
        raise ValueError(
            "HF dataset object IDs do not match selected_galaxies_summary.json. "
            f"Missing={missing[:5]} Extra={extra[:5]}"
        )

    dtype = _build_dtype(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        table = handle.create_dataset("__astropy_table__", shape=(len(dataset),), dtype=dtype)
        for i, item in enumerate(dataset):
            row = np.zeros((), dtype=dtype)
            for name in dtype.names or ():
                _assign_value(row, name, item[name])
            table[i] = row

        provenance_attrs = {
            "source_repo": HF_REPO,
            "source_revision": HF_REVISION,
            "source_split": DEFAULT_SPLIT,
            "rows": len(dataset),
        }
        handle.attrs.update(provenance_attrs)
        table.attrs.update(provenance_attrs)

    print(f"Wrote {len(dataset)} rows to {output_path}")
    print(f"Source: {HF_REPO}@{HF_REVISION}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    materialize(args.output, args.summary, args.overwrite)


if __name__ == "__main__":
    main()
