#!/usr/bin/env python3
"""Build ABIDE-I TR.json, a seed-0 6:2:2 split, and a raw-file manifest.

Site keys match phenotype SITE_ID, except when one SITE_ID contains scans in
more than one Stage 0 TR bucket. CMU is that case: CMU_a is 2.0 s and CMU_b
is 1.5 s, so those subjects are keyed by BIDS folder.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

RAW_DIR = Path("data/abide_i/raw")
OUT_DIR = Path("data/abide_i/processed")
PHENOTYPE = Path("data/Phenotypic V1.0b.csv")

# Stage 0 buckets in conf_embed_downstream.py. A subject outside all of them
# is dropped by the loader.
BUCKETS = (
    (1.50, 1.55),
    (1.60, 1.70),
    (2.00, 2.05),
    (2.20, 2.25),
    (2.50, 2.55),
    (3.00, 3.05),
)

FOLDER_TO_SITE = {
    "Caltech": "CALTECH",
    "CMU_a": "CMU",
    "CMU_b": "CMU",
    "KKI": "KKI",
    "Leuven_1": "LEUVEN_1",
    "Leuven_2": "LEUVEN_2",
    "MaxMun_a": "MAX_MUN",
    "MaxMun_b": "MAX_MUN",
    "MaxMun_c": "MAX_MUN",
    "MaxMun_d": "MAX_MUN",
    "NYU": "NYU",
    "OHSU": "OHSU",
    "Olin": "OLIN",
    "Pitt": "PITT",
    "SBL": "SBL",
    "SDSU": "SDSU",
    "Stanford": "STANFORD",
    "Trinity": "TRINITY",
    "UCLA_1": "UCLA_1",
    "UCLA_2": "UCLA_2",
    "UM_1": "UM_1",
    "UM_2": "UM_2",
    "USM": "USM",
    "Yale": "YALE",
}


def bucket(tr: float) -> tuple[float, float] | None:
    for low, high in BUCKETS:
        if low <= tr < high:
            return (low, high)
    return None


def bold_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"run-(\d+)", path.name)
    run = int(match.group(1)) if match else 1
    # Prefer run-1, then the lowest remaining run number.
    return (0 if run == 1 else 1, run, path.name)


def header_tr(path: Path) -> float:
    zooms = nib.load(str(path)).header.get_zooms()
    if len(zooms) < 4:
        raise RuntimeError(f"no TR in {path}")
    return float(zooms[3])


def discover() -> list[dict]:
    with PHENOTYPE.open(newline="") as handle:
        phenotype = {row["SUB_ID"]: row for row in csv.DictReader(handle)}

    found: list[dict] = []
    missing_label: list[str] = []
    for folder, site in FOLDER_TO_SITE.items():
        folder_dir = RAW_DIR / folder
        if not folder_dir.is_dir():
            continue
        for subject_dir in sorted(p for p in folder_dir.iterdir() if p.is_dir()):
            sid = subject_dir.name.split("-")[-1]
            phenotype_id = str(int(sid))
            t1s = sorted(subject_dir.glob("anat/*T1w.nii.gz"))
            bolds = sorted(subject_dir.glob("func/*task-rest*bold.nii.gz"), key=bold_sort_key)
            if not t1s or not bolds:
                continue
            row = phenotype.get(phenotype_id)
            if row is None:
                missing_label.append(f"{folder}/{subject_dir.name}")
                continue
            tr = header_tr(bolds[0])
            found.append(
                {
                    "id": f"{site}/sub-{sid}",
                    "label": int(row["DX_GROUP"]),
                    "site": site,
                    "bids_folder": folder,
                    "sub_id": sid,
                    "tr": round(tr, 4),
                    "t1": str(t1s[0]),
                    "bold": str(bolds[0]),
                    "n_bold_runs": len(bolds),
                }
            )
    if missing_label:
        print(f"{len(missing_label)} paired scans have no phenotype row")
    return found


def assign_site_keys(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a phenotype site when its scanners fall in different TR buckets."""
    by_site: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_site[row["site"]].append(row)

    kept: list[dict] = []
    dropped: list[dict] = []
    for site, items in by_site.items():
        buckets = {bucket(item["tr"]) for item in items}
        if None in buckets:
            for item in items:
                if bucket(item["tr"]) is None:
                    dropped.append(item)
            items = [item for item in items if bucket(item["tr"]) is not None]
            buckets.discard(None)
        if len(buckets) <= 1:
            kept.extend(items)
            continue
        print(f"{site} spans TR buckets {sorted(buckets)}; keying by BIDS folder")
        for item in items:
            item["site"] = item["bids_folder"]
            item["id"] = f"{item['bids_folder']}/sub-{item['sub_id']}"
            kept.append(item)
    return kept, dropped


def stratified_split(rows: list[dict], seed: int = 0) -> dict[str, list[dict]]:
    """6:2:2 within each site and diagnosis. n//5 goes to val and to test."""
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["site"], row["label"])].append(row)

    rng = np.random.RandomState(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        items = groups[key]
        order = rng.permutation(len(items))
        n_val = len(items) // 5
        n_test = len(items) // 5
        for rank, index in enumerate(order):
            record = {"id": items[index]["id"], "label": items[index]["label"]}
            if rank < n_val:
                splits["val"].append(record)
            elif rank < n_val + n_test:
                splits["test"].append(record)
            else:
                splits["train"].append(record)
    for name in splits:
        splits[name].sort(key=lambda record: record["id"])
    return splits


def main() -> None:
    rows, dropped = assign_site_keys(discover())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tr_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        tr_values[row["site"]].append(row["tr"])
    tr_json = {site: round(float(np.median(values)), 4) for site, values in sorted(tr_values.items())}

    splits = stratified_split(rows, seed=0)
    manifest = {
        "seed": 0,
        "n_paired": len(rows),
        "subjects": rows,
    }

    (OUT_DIR / "TR.json").write_text(json.dumps(tr_json, indent=2) + "\n")
    (OUT_DIR / "data_splits_seed0.json").write_text(json.dumps(splits, indent=2) + "\n")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    labels = [row["label"] for row in rows]
    print(f"paired subjects: {len(rows)}")
    print(f"  ASD (1): {labels.count(1)}  control (2): {labels.count(2)}")
    print(
        "splits seed 0:"
        f" train {len(splits['train'])},"
        f" val {len(splits['val'])},"
        f" test {len(splits['test'])}"
    )
    print("TR.json:")
    for site, tr in tr_json.items():
        n = len(tr_values[site])
        print(f"  {site:12} {tr:.4f}s  n={n}  bucket={bucket(tr)}")
    if dropped:
        print(f"dropped (TR outside Stage 0 buckets): {len(dropped)}")
        for item in dropped:
            print(f"  {item['id']} tr={item['tr']}")
    multi = [row for row in rows if row["n_bold_runs"] > 1]
    print(f"subjects with extra rest runs (using run-1, else lowest run): {len(multi)}")


if __name__ == "__main__":
    main()
