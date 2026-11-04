# Setup and reproducibility

Log of what was actually done to reproduce BrainHarmonix downstream results on ABIDE-I. The upstream quick start stays in the root `README.md`. Processing rules for the loader live in `docs/abide-preprocess-checklist.md`.

`data/` is gitignored, so the phenotype file and scans stay on this machine.

## Goal

Run Stage 0 embedding and Stage 2 finetuning on ABIDE-I. Stage 0 pretraining (`run_embed_pretrain.sh`) and Stage 1 are the UK Biobank path and are not part of this reproduction.

The loader in `datasets/datasets.py` does not read raw NIfTIs. It expects parcellated fMRI CSVs, cropped T1s, `data_splits.json`, and `TR.json`.

## Steps taken

### Before 2026-11-04 — NITRC access and phenotype download

- Registered at [NITRC](https://www.nitrc.org/).
- Joined the [ABIDE / 1000 Functional Connectomes](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) project.
- Downloaded the composite phenotypic file (`Phenotypic_V1_0b.csv`). The columns needed from it are `SITE_ID`, `SUB_ID`, and `DX_GROUP`.
- The paper kept 700 subjects who had both a T1 and a resting-state run (320 control / 380 ASD), not the full 1,112.

### 2026-11-04 — Phenotype file on disk

The downloaded file is at `data/Phenotypic V1.0b.csv`.

- 1,112 subjects, 74 columns.
- `DX_GROUP` is already 1 (ASD) or 2 (control). The loader subtracts 1.
- TR is not in this file. It is a site-level acquisition parameter.

### 2026-11-04 — Raw paired T1 and rest download

Added `scripts/download_abide_raw.py` and a Makefile target. The script reads the phenotype file and downloads only anatomical T1w and `task-rest` BOLD NIfTIs from the public FCP-INDI bucket. The AWS CLI is not required. Completed files are skipped on a re-run.

```bash
make download-abide
# one phenotype site:
make download-abide SITE=PITT
```

Source: `s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/`  
Destination: `data/abide_i/raw/<BIDS site>/sub-00<SUB_ID>/{anat,func}/`

Started `make download-abide` at 23:51 UTC+1 on 3 Nov 2026. Stopped it at 00:10 UTC+1 on 4 Nov 2026, partway through NYU. On disk then: 524 files, 12.3 GB (Caltech, CMU, KKI, Leuven, MaxMun, and part of NYU). Inventory before the transfer:

| | Files | Size |
|---|---:|---:|
| T1 (`*_T1w.nii.gz`) | 1,102 | 7.4 GB |
| Resting-state BOLD | 1,163 | 56.4 GB |
| Total | 2,265 | 63.8 GB |

CMU is stored as `CMU_a` and `CMU_b`. MaxMun is stored as `MaxMun_a` through `MaxMun_d`. Subject ids in the CSV are 5 digits; BIDS folders zero-pad them to 7 (`50002` → `sub-0050002`).

Ten subjects have a rest scan and no T1, so they are not pairs:

- UCLA_1: 51232, 51233, 51242, 51243, 51244, 51245, 51246, 51247, 51270
- UCLA_2: 51310

Re-run `make download-abide` to resume. Files that already match the remote size are skipped.

## Not done yet

- [x] Finish the raw download (524 / 2,265 files, 12.3 / 63.8 GB on disk after the 4 Nov 2026 stop) with `make download-abide`.
- [ ] Build `data/abide_i/processed/TR.json` from site TRs, not from the phenotype CSV.
- [ ] Build `data/abide_i/processed/data_splits.json`.
- [ ] Preprocess T1 to `{subject}_T1.nii.gz` at 167 × 212 × 160.
- [ ] Preprocess rest fMRI to `{subject}_Schaefer17n400p.csv.gz`.
- [ ] Point `configs/harmonizer/stage0_embed/conf_embed_downstream.py` at those folders. The paths are still `/path/to/...`.
- [x] Download the Google Drive checkpoints into `checkpoints/harmonix-f/`, `checkpoints/harmonix-s/`, and `checkpoints/harmonizer/`.
- [ ] Run `bash scripts/harmonizer/stage0_embed/run_embed_downstream.sh configs/harmonizer/stage0_embed/conf_embed_downstream.py`.
- [ ] Point Stage 2 at a local splits file. `modules/harmonizer/stage2_finetune/main_finetune.py` still uses `/scratch/Projects/project_312_HelenZhou/ABIDE1_fMRI_T1/data_splits_seed${split_seed}.json`.
- [ ] Run `make stage-2` (or `bash scripts/harmonizer/stage2_finetune/run_finetune.sh base 128 AbideI 0`).

## Disk

Paired raw imaging is 63.8 GB. The files the loader keeps (cropped T1s and Schaefer CSVs) should add well under 10 GB. Delete per-subject FSL intermediates; keeping every 4D copy will not fit beside the raw data.
