#!/usr/bin/env python3
"""Download paired ABIDE-I T1 and resting-state fMRI from the public FCP-INDI bucket.

Only anatomical T1w and task-rest BOLD NIfTIs are fetched, for subjects listed in
the phenotypic CSV. CMU and MaxMun are split across several BIDS site folders.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://fcp-indi.s3.amazonaws.com"
LIST_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Phenotype SITE_ID -> BIDS folders under RawDataBIDS/.
SITE_FOLDERS = {
    "CALTECH": ["Caltech"],
    "CMU": ["CMU_a", "CMU_b"],
    "KKI": ["KKI"],
    "LEUVEN_1": ["Leuven_1"],
    "LEUVEN_2": ["Leuven_2"],
    "MAX_MUN": ["MaxMun_a", "MaxMun_b", "MaxMun_c", "MaxMun_d"],
    "NYU": ["NYU"],
    "OHSU": ["OHSU"],
    "OLIN": ["Olin"],
    "PITT": ["Pitt"],
    "SBL": ["SBL"],
    "SDSU": ["SDSU"],
    "STANFORD": ["Stanford"],
    "TRINITY": ["Trinity"],
    "UCLA_1": ["UCLA_1"],
    "UCLA_2": ["UCLA_2"],
    "UM_1": ["UM_1"],
    "UM_2": ["UM_2"],
    "USM": ["USM"],
    "YALE": ["Yale"],
}


def list_keys(prefix: str) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    token = None
    while True:
        query = f"list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            query += f"&continuation-token={urllib.parse.quote(token)}"
        url = f"{BUCKET}/?{query}"
        with urllib.request.urlopen(url, timeout=120) as response:
            root = ET.fromstring(response.read())
        for content in root.findall("s:Contents", LIST_NS):
            keys.append(
                (
                    content.find("s:Key", LIST_NS).text,
                    int(content.find("s:Size", LIST_NS).text),
                )
            )
        if root.find("s:IsTruncated", LIST_NS).text != "true":
            return keys
        token = root.find("s:NextContinuationToken", LIST_NS).text


def is_paired_scan(key: str) -> bool:
    name = key.rsplit("/", 1)[-1]
    if "/anat/" in key and name.endswith("_T1w.nii.gz"):
        return True
    return "/func/" in key and "task-rest" in name and name.endswith("_bold.nii.gz")


def subject_id_from_key(key: str) -> str:
    name = key.rsplit("/", 1)[-1]
    token = name.split("_", 1)[0]
    return str(int(token.split("-", 1)[1]))


def load_subjects(phenotype: Path, sites: set[str] | None) -> list[dict[str, str]]:
    with phenotype.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    unknown = sorted({row["SITE_ID"] for row in rows} - set(SITE_FOLDERS))
    if unknown:
        raise SystemExit(f"Phenotype has unmapped SITE_ID values: {unknown}")
    if sites:
        missing = sites - set(SITE_FOLDERS)
        if missing:
            raise SystemExit(
                f"Unknown SITE {sorted(missing)}. Choose from: {', '.join(sorted(SITE_FOLDERS))}"
            )
        rows = [row for row in rows if row["SITE_ID"] in sites]
    return rows


def collect_downloads(
    rows: list[dict[str, str]],
) -> tuple[list[tuple[str, int]], list[str]]:
    folders = []
    for site in sorted({row["SITE_ID"] for row in rows}):
        folders.extend(SITE_FOLDERS[site])

    by_subject: dict[str, list[tuple[str, int]]] = {}
    for folder in folders:
        print(f"Listing s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/{folder}/", flush=True)
        for key, size in list_keys(f"data/Projects/ABIDE/RawDataBIDS/{folder}/"):
            if is_paired_scan(key):
                by_subject.setdefault(subject_id_from_key(key), []).append((key, size))

    downloads: list[tuple[str, int]] = []
    missing: list[str] = []
    for row in rows:
        files = by_subject.get(row["SUB_ID"], [])
        kinds = {("t1" if "/anat/" in key else "bold") for key, _ in files}
        if kinds != {"t1", "bold"}:
            missing.append(f"{row['SITE_ID']}/{row['SUB_ID']} ({', '.join(sorted(kinds)) or 'none'})")
        downloads.extend(files)
    # Same scan can only belong to one subject, but keep the list unique.
    unique = list(dict.fromkeys(downloads))
    return unique, missing


def download_one(key: str, size: int, out_dir: Path) -> str:
    relative = key.split("RawDataBIDS/", 1)[1]
    dest = out_dir / relative
    if dest.is_file() and dest.stat().st_size == size:
        return f"skip {relative}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(dest),
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size != size:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"failed {relative}: {detail}")
    return f"got  {relative}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotype",
        type=Path,
        default=Path("data/Phenotypic V1.0b.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/abide_i/raw"))
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        help="Phenotype SITE_ID to download (repeatable). Default: every site.",
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    rows = load_subjects(args.phenotype, {site for site in args.site if site} or None)
    downloads, missing = collect_downloads(rows)
    total = sum(size for _, size in downloads)
    print(
        f"{len(rows)} phenotype subjects, {len(downloads)} files, {total / 1e9:.2f} GB"
        f" -> {args.out}",
        flush=True,
    )
    if missing:
        print(f"{len(missing)} subjects are missing a T1 or a rest scan:", flush=True)
        for item in missing:
            print(f"  {item}", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(download_one, key, size, args.out) for key, size in downloads
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                message = future.result()
            except Exception as exc:
                failures += 1
                print(f"[{done}/{len(futures)}] {exc}", file=sys.stderr, flush=True)
            else:
                print(f"[{done}/{len(futures)}] {message}", flush=True)
    if failures:
        raise SystemExit(f"{failures} downloads failed. Re-run to resume.")


if __name__ == "__main__":
    main()
