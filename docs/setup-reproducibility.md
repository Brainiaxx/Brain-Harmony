# Setup and reproducibility

Log of what was actually done to reproduce BrainHarmonix downstream results on ABIDE-I. The upstream quick start stays in the root `README.md`. The paper's preprocessing spec is in `docs/abide-preprocess-checklist.md`.

`data/` is gitignored, so the phenotype file and scans stay on this machine.

## Goal

Run Stage 0 embedding and Stage 2 finetuning on ABIDE-I. Stage 0 pretraining (`run_embed_pretrain.sh`) and Stage 1 are the UK Biobank path and are not part of this reproduction.

The loader in `datasets/datasets.py` does not read raw NIfTIs. It expects parcellated fMRI CSVs, cropped T1s, a splits file, and `TR.json`.

## Steps taken

### Before 2026-11-03 — NITRC access and phenotype download

- Registered at [NITRC](https://www.nitrc.org/).
- Joined the [ABIDE / 1000 Functional Connectomes](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) project.
- Downloaded the composite phenotypic file (`Phenotypic_V1_0b.csv`). The columns needed from it are `SITE_ID`, `SUB_ID`, and `DX_GROUP`.
- The paper kept 700 subjects who had both a T1 and a resting-state run (320 control / 380 ASD), not the full 1,112. Which 700 is not in this repo.

### 2026-11-03 — Phenotype file on disk

The downloaded file is at `data/Phenotypic V1.0b.csv`.

- 1,112 subjects, 74 columns.
- `DX_GROUP` is already 1 (ASD) or 2 (control). The loader subtracts 1.
- TR is not in this file. It is a site-level acquisition parameter, read later from the NIfTI headers.

### 2026-11-03 — Raw paired T1 and rest download

Added `scripts/download_abide_raw.py` and `make download-abide`. The script reads the phenotype file and downloads only anatomical T1w and `task-rest` BOLD NIfTIs from the public FCP-INDI bucket. The AWS CLI is not required. Completed files are skipped on a re-run.

Source: `s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/`  
Destination: `data/abide_i/raw/<BIDS site>/sub-00<SUB_ID>/{anat,func}/`

Started at 23:51 UTC+1 on 3 Nov 2026, stopped at 00:10 UTC+1 on 4 Nov 2026 partway through NYU (524 files, 12.3 GB), then finished later the same day.

| | Files | Size |
|---|---:|---:|
| T1 (`*_T1w.nii.gz`) | 1,102 | 7.4 GB |
| Resting-state BOLD | 1,163 | 56.4 GB |
| Total | 2,265 | 63.8 GB |

CMU is stored as `CMU_a` and `CMU_b`. MaxMun is stored as `MaxMun_a` through `MaxMun_d`. Subject ids in the CSV are 5 digits; BIDS folders zero-pad them to 7 (`50002` → `sub-0050002`).

Ten subjects have a rest scan and no T1, so they are not pairs:

- UCLA_1: 51232, 51233, 51242, 51243, 51244, 51245, 51246, 51247, 51270
- UCLA_2: 51310

### Before 2026-11-03 — Checkpoints

Present on disk:

- `checkpoints/harmonix-f/model.pth`
- `checkpoints/harmonix-s/model.pth`
- `checkpoints/harmonizer/model.pth`

### 2026-11-04 — Splits, TR, and config paths

`make prepare-abide` (`scripts/prepare_abide_metadata.py`) read every rest-scan header and wrote:

- `data/abide_i/processed/TR.json`
- `data/abide_i/processed/data_splits_seed0.json`
- `data/abide_i/processed/manifest.json`

1,102 subjects have both a T1 and a rest scan (531 ASD, 571 control). The seed-0 split is stratified by site and diagnosis, `n // 5` each to val and test: 692 / 205 / 205. This is a local split. It is not the paper's unpublished split.

TR is per scanner. CMU does not share one value: `CMU_a` is 2.0 s and `CMU_b` is 1.5 s, so those subjects use the BIDS folder as the site key. Every kept site falls in a Stage 0 bucket. Subjects with more than one rest run (28, all OHSU) are listed in the manifest as run-1, otherwise the lowest run number.

`configs/harmonizer/stage0_embed/conf_embed_downstream.py` now points at `data/abide_i/processed/`. `modules/harmonizer/stage2_finetune/main_finetune.py` reads `data/abide_i/processed/data_splits_seed<seed>.json` instead of the authors' cluster path.

### 2026-11-04 — One-subject preprocessing trial

FreeSurfer and FSL are not installed. `scripts/preprocess_abide.py` was written as a Python stand-in and run on Pitt `0050002` only. Outputs:

- `data/abide_i/processed/t1/0050002_T1.nii.gz` — shape 167 × 212 × 160
- `data/abide_i/processed/fmri/0050002/0050002_Schaefer17n400p.csv.gz` — shape 400 × 196, 73 ROIs empty

The empty ROIs match a short slab: that rest scan covers 116 mm. The cohort has not been preprocessed. About 20 seconds per subject, so all 1,102 would be on the order of 6 hours at one job.

Installing the stand-in also put `nilearn`, `SimpleITK`, `surfa`, and `scikit-image` into the `brainharmonix` environment and upgraded `pandas` (2.2.3 → 2.3.3) and `packaging` (25 → 26.3). Those packages are not in `requirements-macos.txt`.

## Preprocessing choice

Appendix A skull-strips with FreeSurfer, reorients with FSL, affine-registers with FLIRT, motion-corrects with `mcflirt`, and coregisters with `bbregister`, then regresses voxelwise nuisance signals. The stand-in erodes a head mask, registers with SimpleITK to the MNI152NLin6Asym 1 mm brain template, crops the same 167 × 212 × 160 window, drops the first 4 volumes, rigid-aligns each volume, and regresses global, WM, and CSF signals on the ROI time series.

| | Python stand-in | FreeSurfer + FSL |
|---|---|---|
| Runs the repo | Yes. Shapes match the loader. | Yes, after the same crop and CSV layout. |
| Matches Appendix A | No. Skull strip, registration, tissue classes, and regression all differ. No slice-timing. | Yes, for the steps the paper names. |
| Matches published numbers | No. | Closer. The 700-subject list and the paper's split seeds are still unpublished, so the tables can still differ. |
| Install | Already in `brainharmonix`. | FreeSurfer license plus an FSL install. Neither is on this Mac. |
| Time and disk | ~6 hours, small final files. | Days for 1,102 subjects. Intermediate 4D volumes can use more than the ~110 GB free. |
| Reviewer story | A local reimplementation. | The pipeline named in the paper. |

The cohort run is waiting on this choice.

## Not done yet

- [ ] Choose the stand-in or FreeSurfer + FSL, then preprocess the cohort.
- [ ] Run Stage 0: `bash scripts/harmonizer/stage0_embed/run_embed_downstream.sh configs/harmonizer/stage0_embed/conf_embed_downstream.py`
- [ ] Run Stage 2: `make stage-2`

## Disk

Paired raw imaging is 63.8 GB. Cropped T1s and Schaefer CSVs should add well under 10 GB. Delete per-subject intermediates. Keeping every 4D copy will not fit beside the raw data.

## Run this to reproduce

These commands reproduce the local setup above: raw ABIDE-I, the seed-0 split, and the Python stand-in. They do not reproduce the paper's published numbers.

From the repo root, with the phenotype CSV already at `data/Phenotypic V1.0b.csv`:

```bash
make setup-macos
~/.pyenv/versions/brainharmonix/bin/pip install nilearn simpleitk surfa scikit-image

# Place checkpoints in checkpoints/harmonix-f/, checkpoints/harmonix-s/, and checkpoints/harmonizer/.
# https://drive.google.com/drive/folders/12MkUAOcegU60YVlK8u8_Owmgk4eQVheB

make download-abide
make prepare-abide

# One subject. Pitt 0050002 is the subject already on disk.
make preprocess-abide SITE=PITT LIMIT=1

# Cohort, only after choosing the stand-in. About 6 hours at JOBS=1.
# make preprocess-abide JOBS=1

bash scripts/harmonizer/stage0_embed/run_embed_downstream.sh \
  configs/harmonizer/stage0_embed/conf_embed_downstream.py
make stage-2
```

`make stage-2` is `bash scripts/harmonizer/stage2_finetune/run_finetune.sh base 128 AbideI 0`. Stage 0 and Stage 2 need the cohort outputs and a GPU. `make stage-0` and `make stage-1` are the UK Biobank path.
