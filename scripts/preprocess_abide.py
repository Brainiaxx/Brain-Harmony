#!/usr/bin/env python3
"""Preprocess ABIDE-I into the tensors datasets.py can read.

This is the paper's Appendix A pipeline adapted to run without FreeSurfer or FSL:

- T1: morphological skull-strip, affine registration to MNI152NLin6Asym 1 mm
  (the FSL 182×218×182 template), crop to 167×212×160, scale to [0, 1].
- fMRI: drop the first 4 volumes, rigid motion correction, nuisance regression
  (global, WM, CSF, 6 motion parameters and their derivatives), despike,
  band-pass 0.009–0.08 Hz, then average inside Schaefer 400 / 17-network ROIs.

The paper skull-strips with FreeSurfer, motion-corrects with mcflirt, and
coregisters with bbregister. Those binaries are not installed here. ROI order
matches Schaefer labels 1..400, which is the order of the position-embedding file.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from nibabel.processing import resample_from_to
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.signal import clean
from scipy import ndimage

ROOT = Path("data/abide_i")
MODEL_DIR = Path("data/models")
TEMPLATE_URL = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/"
    "tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz"
)
TEMPLATE_PATH = MODEL_DIR / "mni152nlin6asym_res-01_desc-brain_T1w.nii.gz"
ATLAS_PATH = MODEL_DIR / "schaefer400_17networks_on_mni_brain.nii.gz"
CROP = (slice(6, 173), slice(2, 214), slice(0, 160))  # 182×218×182 -> 167×212×160


def bucket_hi(tr: float) -> float:
    for low, high in (
        (1.50, 1.55),
        (1.60, 1.70),
        (2.00, 2.05),
        (2.20, 2.25),
        (2.50, 2.55),
        (3.00, 3.05),
    ):
        if low <= tr < high:
            return high
    raise RuntimeError(f"TR {tr} is outside every Stage 0 bucket")


def max_timepoints(tr: float) -> int:
    # Use the top of the bucket so a slightly larger site TR in the same
    # loader pass still satisfies ceil(T / patch_size) <= 18.
    patch = round((48 * 0.735) / (bucket_hi(tr) - 1e-4))
    return 18 * patch


def download_template() -> None:
    if TEMPLATE_PATH.is_file() and TEMPLATE_PATH.stat().st_size > 1_000_000:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"Downloading MNI152 template -> {TEMPLATE_PATH}", flush=True)
    urllib.request.urlretrieve(TEMPLATE_URL, TEMPLATE_PATH)


def ensure_atlas(template: nib.Nifti1Image) -> None:
    if ATLAS_PATH.is_file():
        return
    print("Downloading Schaefer 400 / 17-network atlas", flush=True)
    fetched = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=1)
    atlas = nib.load(fetched.maps)
    on_template = resample_from_to(atlas, template, order=0)
    labels = np.unique(np.asanyarray(on_template.dataobj))
    missing = [i for i in range(1, 401) if i not in set(labels.tolist())]
    if missing:
        raise RuntimeError(f"Schaefer atlas is missing labels {missing[:8]}")
    nib.save(on_template, str(ATLAS_PATH))


def canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def sitk_to_array(image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3:
        return np.transpose(array, (2, 1, 0))
    return np.transpose(array, (3, 2, 1, 0))


def _largest(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        raise RuntimeError("skull strip found no brain")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def brain_mask(data: np.ndarray, zooms: tuple[float, float, float]) -> np.ndarray:
    """Keep the brain and drop the skull.

    A low threshold selects the whole head. Eroding by about 10 mm breaks the
    skull off, and dilating the remaining core by about 6 mm restores cortex
    without bringing the skull back.
    """
    finite = np.isfinite(data)
    if not np.any(finite):
        raise RuntimeError("T1 has no finite voxels")
    step = float(min(zooms))
    robust_max = float(np.percentile(data[finite], 99))
    head = _largest(ndimage.binary_opening(finite & (data > 0.10 * robust_max), iterations=1))
    head = ndimage.binary_fill_holes(head)
    core = ndimage.binary_erosion(head, iterations=max(1, int(round(10.0 / step))))
    core = _largest(core)
    brain = ndimage.binary_dilation(core, iterations=max(1, int(round(6.0 / step))))
    brain &= head
    return ndimage.binary_fill_holes(brain)


def recenter_on(moving: sitk.Image, fixed: sitk.Image) -> None:
    """Put the moving volume's center on the fixed volume's center.

    CMU (and Yale) store the rest scan hundreds of millimetres away from the
    T1. Rigid registration then never sees the brain, and every ROI is empty.
    """
    fixed_center = np.array(
        fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(fixed.GetSize(), dtype=float) - 1) / 2))
    )
    moving_center = np.array(
        moving.TransformContinuousIndexToPhysicalPoint(tuple((np.array(moving.GetSize(), dtype=float) - 1) / 2))
    )
    origin = np.array(moving.GetOrigin()) + (fixed_center - moving_center)
    moving.SetOrigin(tuple(float(value) for value in origin))


def register(fixed: sitk.Image, moving: sitk.Image, dof: str) -> sitk.Transform:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.1, 1)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=60,
        relaxationFactor=0.5,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2])
    method.SetSmoothingSigmasPerLevel([2, 1])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if dof == "affine":
        initial = sitk.AffineTransform(3)
    elif dof == "rigid":
        initial = sitk.Euler3DTransform()
    else:
        raise ValueError(dof)
    initializer = sitk.CenteredTransformInitializerFilter.MOMENTS
    method.SetInitialTransform(
        sitk.CenteredTransformInitializer(fixed, moving, initial, initializer),
        inPlace=False,
    )
    return method.Execute(fixed, moving)


def rigid_params(fixed: sitk.Image, moving: sitk.Image) -> tuple[sitk.Transform, np.ndarray]:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.05, 1)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-3,
        numberOfIterations=25,
        relaxationFactor=0.5,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4])
    method.SetSmoothingSigmasPerLevel([2])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(sitk.Euler3DTransform(), inPlace=False)
    transform = method.Execute(fixed, moving)
    return transform, np.array(transform.GetParameters(), dtype=np.float64)


def tissue_classes(t1: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from skimage.filters import threshold_multiotsu

    values = t1[mask]
    if values.size < 100:
        raise RuntimeError("brain mask is too small to segment")
    low, high = threshold_multiotsu(values, classes=3)
    csf = mask & (t1 <= low)
    wm = mask & (t1 > high)
    csf = ndimage.binary_erosion(csf, iterations=1)
    wm = ndimage.binary_erosion(wm, iterations=1)
    return wm, csf


def mean_signal(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(data.shape[-1], dtype=np.float32)
    return data[mask].mean(axis=0).astype(np.float32)


def despike(series: np.ndarray) -> np.ndarray:
    median = ndimage.median_filter(series, size=(1, 5), mode="nearest")
    residual = series - median
    mad = np.median(np.abs(residual), axis=1, keepdims=True) + 1e-6
    spikes = np.abs(residual) > (4.4478 * mad)
    cleaned = series.copy()
    cleaned[spikes] = median[spikes]
    return cleaned


def extract_rois(bold: np.ndarray, labels: np.ndarray) -> np.ndarray:
    flat_bold = bold.reshape(-1, bold.shape[-1])
    flat_labels = labels.reshape(-1)
    series = np.zeros((400, bold.shape[-1]), dtype=np.float32)
    for roi in range(1, 401):
        voxels = flat_bold[flat_labels == roi]
        if voxels.shape[0] == 0:
            continue
        series[roi - 1] = voxels.mean(axis=0)
    empty = int(np.sum(np.all(series == 0, axis=1)))
    # Short ABIDE slabs leave a lot of cortex outside the field of view.
    # Caltech covers about 119 mm and typically hits only 80–150 of 400 ROIs.
    # Fewer than 60 means the atlas missed the brain.
    if 400 - empty < 60:
        raise RuntimeError(f"{empty} Schaefer ROIs were empty after registration")
    return series


def save_t1(data: np.ndarray, template: nib.Nifti1Image, dest: Path) -> None:
    cropped = np.clip(data[CROP], 0, None).astype(np.float32)
    if cropped.shape != (167, 212, 160):
        raise RuntimeError(f"cropped T1 shape is {cropped.shape}")
    positive = cropped[cropped > 0]
    if positive.size == 0:
        raise RuntimeError("cropped T1 is empty")
    cropped = np.clip(cropped / np.percentile(positive, 99.5), 0, 1)
    affine = template.affine.copy()
    # Shift the origin by the crop start so the voxels keep their MNI coordinates.
    starts = np.array([CROP[0].start, CROP[1].start, CROP[2].start, 1.0])
    affine[:3, 3] = (template.affine @ starts)[:3]
    dest.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(cropped, affine), str(dest))


def save_series(series: np.ndarray, dest: Path) -> None:
    frame = pd.DataFrame(series, index=[f"ROI_{i}" for i in range(1, 401)])
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, compression="gzip")


def process_subject(record: dict, work_root: Path) -> str:
    sid = record["sub_id"]
    t1_out = ROOT / "processed" / "t1" / f"{sid}_T1.nii.gz"
    fmri_out = ROOT / "processed" / "fmri" / sid / f"{sid}_Schaefer17n400p.csv.gz"
    if t1_out.is_file() and fmri_out.is_file():
        return f"skip {record['id']}"

    work = work_root / sid
    work.mkdir(parents=True, exist_ok=True)
    template = nib.load(str(TEMPLATE_PATH))
    atlas_img = nib.load(str(ATLAS_PATH))

    t1_img = canonical(Path(record["t1"]))
    t1 = np.asanyarray(t1_img.dataobj, dtype=np.float32)
    mask = brain_mask(t1, tuple(float(z) for z in t1_img.header.get_zooms()[:3]))
    voxel_ml = float(np.prod(t1_img.header.get_zooms()[:3])) / 1000.0
    brain_ml = float(mask.sum()) * voxel_ml
    if not 700 <= brain_ml <= 2200:
        raise RuntimeError(f"skull strip volume {brain_ml:.0f} ml is outside 700-2200")
    stripped = t1 * mask
    stripped_path = work / "t1_stripped.nii.gz"
    nib.save(nib.Nifti1Image(stripped, t1_img.affine), str(stripped_path))

    fixed = sitk.ReadImage(str(TEMPLATE_PATH), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(stripped_path), sitk.sitkFloat32)
    t1_to_mni = register(fixed, moving, "affine")
    warped = sitk.Resample(moving, fixed, t1_to_mni, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    mask_sitk = sitk.GetImageFromArray(np.transpose(mask.astype(np.uint8), (2, 1, 0)))
    mask_sitk.CopyInformation(moving)
    mask_mni = sitk.Resample(mask_sitk, fixed, t1_to_mni, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    warped_array = sitk_to_array(warped).astype(np.float32) * sitk_to_array(mask_mni)
    if warped_array.shape != template.shape:
        raise RuntimeError(f"warped T1 shape {warped_array.shape} != {template.shape}")
    bright = np.argwhere(warped_array > np.percentile(warped_array, 90))
    center = bright.mean(axis=0)
    if np.any(np.abs(center - np.array(warped_array.shape) / 2) > np.array(warped_array.shape) * 0.2):
        raise RuntimeError(f"warped brain center {center.round(1)} is outside the MNI volume")
    save_t1(warped_array, template, t1_out)

    bold_img = canonical(Path(record["bold"]))
    bold = np.asanyarray(bold_img.dataobj, dtype=np.float32)
    if bold.ndim != 4:
        raise RuntimeError(f"bold rank is {bold.ndim}")
    bold = bold[..., 4:]
    limit = max_timepoints(float(record["tr"]))
    if bold.shape[-1] > limit:
        bold = bold[..., :limit]
    if bold.shape[-1] < 20:
        raise RuntimeError(f"only {bold.shape[-1]} volumes after trimming")
    bold_img = nib.Nifti1Image(bold, bold_img.affine)
    bold_path = work / "bold.nii.gz"
    nib.save(bold_img, str(bold_path))

    bold_sitk = sitk.ReadImage(str(bold_path), sitk.sitkFloat32)
    size = list(bold_sitk.GetSize())
    n_volumes = size[3]
    size[3] = 0
    extract = sitk.ExtractImageFilter()
    extract.SetSize(size)
    extract.SetIndex([0, 0, 0, n_volumes // 2])
    reference = extract.Execute(bold_sitk)

    corrected = np.empty_like(bold)
    motion = np.zeros((n_volumes, 6), dtype=np.float64)
    for frame in range(n_volumes):
        extract.SetIndex([0, 0, 0, frame])
        volume = extract.Execute(bold_sitk)
        transform, params = rigid_params(reference, volume)
        motion[frame] = params
        moved = sitk.Resample(volume, reference, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        corrected[..., frame] = sitk_to_array(moved)

    wm, csf = tissue_classes(stripped, mask)
    t1_sitk = sitk.ReadImage(str(stripped_path), sitk.sitkFloat32)
    mean_bold = sitk.GetImageFromArray(np.transpose(corrected.mean(axis=3), (2, 1, 0)))
    mean_bold.CopyInformation(reference)
    recenter_on(reference, t1_sitk)
    mean_bold.SetOrigin(reference.GetOrigin())
    bold_to_t1 = register(t1_sitk, mean_bold, "rigid")

    def resample_mask(mask_array: np.ndarray) -> np.ndarray:
        mask_img = sitk.GetImageFromArray(np.transpose(mask_array.astype(np.uint8), (2, 1, 0)))
        mask_img.CopyInformation(t1_sitk)
        resampled = sitk.Resample(mask_img, reference, bold_to_t1, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk_to_array(resampled).astype(bool)

    atlas_on_t1 = sitk.Resample(
        sitk.ReadImage(str(ATLAS_PATH), sitk.sitkUInt16),
        t1_sitk,
        t1_to_mni,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )
    # t1_to_mni maps a T1 point into MNI, which is what Resample expects when
    # the atlas (input) is in MNI and the reference is the T1 grid.
    labels_on_t1 = sitk.GetImageFromArray(sitk.GetArrayFromImage(atlas_on_t1))
    labels_on_t1.CopyInformation(t1_sitk)
    labels = sitk_to_array(
        sitk.Resample(labels_on_t1, reference, bold_to_t1, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    ).astype(np.int16)

    derivative = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])
    confounds = [mean_signal(corrected, resample_mask(mask))]
    wm_bold = resample_mask(wm)
    csf_bold = resample_mask(csf)
    if np.any(wm_bold):
        confounds.append(mean_signal(corrected, wm_bold))
    if np.any(csf_bold):
        confounds.append(mean_signal(corrected, csf_bold))
    confounds.append(motion.astype(np.float32))
    confounds.append(derivative.astype(np.float32))
    confound_matrix = np.column_stack(confounds)

    series = despike(extract_rois(corrected, labels))
    filtered = clean(
        series.T,
        detrend=True,
        standardize=False,
        confounds=confound_matrix,
        standardize_confounds=True,
        low_pass=0.08,
        high_pass=0.009,
        t_r=float(record["tr"]),
    ).T.astype(np.float32)
    save_series(filtered, fmri_out)
    import shutil

    empty = int(np.sum(np.all(~np.isfinite(filtered) | (filtered == 0), axis=1)))
    shutil.rmtree(work, ignore_errors=True)
    return f"got  {record['id']}  brain={brain_ml:.0f}ml  T={filtered.shape[1]}  empty_rois={empty}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "processed" / "manifest.json")
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument("--subject", action="append", default=[], help="7-digit subject id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    download_template()
    template = canonical(TEMPLATE_PATH)
    # Keep the on-disk template in the orientation SimpleITK and nibabel agree on.
    nib.save(template, str(TEMPLATE_PATH))
    ensure_atlas(nib.load(str(TEMPLATE_PATH)))

    manifest = json.loads(args.manifest.read_text())
    records = manifest["subjects"]
    if args.site:
        wanted = set(args.site)
        records = [row for row in records if row["site"] in wanted or row["bids_folder"] in wanted]
    if args.subject:
        wanted_ids = {sid.zfill(7) for sid in args.subject}
        records = [row for row in records if row["sub_id"] in wanted_ids]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no subjects selected")

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, 4 // args.jobs))
    work_root = ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"preprocessing {len(records)} subjects with {args.jobs} jobs", flush=True)

    failures = 0
    if args.jobs == 1:
        for index, record in enumerate(records, start=1):
            try:
                message = process_subject(record, work_root)
            except Exception as exc:
                failures += 1
                print(f"[{index}/{len(records)}] FAIL {record['id']}: {exc}", flush=True)
                traceback.print_exc()
            else:
                print(f"[{index}/{len(records)}] {message}", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_subject, record, work_root): record for record in records}
            for index, future in enumerate(as_completed(futures), start=1):
                record = futures[future]
                try:
                    message = future.result()
                except Exception as exc:
                    failures += 1
                    print(f"[{index}/{len(records)}] FAIL {record['id']}: {exc}", flush=True)
                else:
                    print(f"[{index}/{len(records)}] {message}", flush=True)
    update_splits(records)
    if failures:
        raise SystemExit(f"{failures} subjects failed")


def has_processed_outputs(subject_id: str) -> bool:
    """True when the loader can open both files for this subject."""
    sid = subject_id.split("-")[-1]
    t1 = ROOT / "processed" / "t1" / f"{sid}_T1.nii.gz"
    fmri = ROOT / "processed" / "fmri" / sid / f"{sid}_Schaefer17n400p.csv.gz"
    return t1.is_file() and fmri.is_file()


def update_splits(records: list[dict]) -> None:
    """Drop attempted subjects that still lack both outputs from the seed-0 split.

    Subjects this run did not attempt stay put, so a one-site run does not empty the split.
    """
    splits_path = ROOT / "processed" / "data_splits_seed0.json"
    if not splits_path.is_file():
        print(f"no split to update at {splits_path}", flush=True)
        return
    splits = json.loads(splits_path.read_text())
    attempted = {row["id"] for row in records}
    updated: dict[str, list[dict]] = {}
    removed: list[tuple[str, dict]] = []
    for name, rows in splits.items():
        updated[name] = []
        for row in rows:
            if row["id"] in attempted and not has_processed_outputs(row["id"]):
                removed.append((name, row))
            else:
                updated[name].append(row)
    if not removed:
        print(
            "seed-0 split already excludes attempted subjects that lack both outputs",
            flush=True,
        )
        return
    splits_path.write_text(json.dumps(updated, indent=2) + "\n")
    print(f"removed {len(removed)} attempted subjects from {splits_path.name}", flush=True)
    for name, row in removed:
        print(f"  {name}  {row['id']}", flush=True)


if __name__ == "__main__":
    main()
