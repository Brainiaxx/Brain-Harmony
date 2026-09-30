# ABIDE-I download and preprocess checklist

This repo does **not** take raw NIfTIs. `datasets.py` and `configs/harmonizer/stage0_embed/conf_embed_downstream.py` expect already-parcellated fMRI CSVs, cropped T1s, and two JSON files.

Use this for Stage 0 downstream embedding:

```bash
bash scripts/harmonizer/stage0_embed/run_embed_downstream.sh \
  configs/harmonizer/stage0_embed/conf_embed_downstream.py
```

Do **not** use `run_embed_pretrain.sh` for ABIDE. That script is the UK Biobank path.

The default downstream config sets `use_subcortical=False`, so you only need **Schaefer-400**, not Tian subcortical.

## Target layout

```text
data/abide_i/
  raw/                         # what you download
  processed/
    fmri/                      # fmri_data_dir
      0050952/
        0050952_Schaefer17n400p.csv.gz
    t1/                        # T1_data_dir
      0050952_T1.nii.gz
    data_splits.json
    TR.json
```

## 1. Download ABIDE-I

1. Register at [NITRC](https://www.nitrc.org/).
2. Join [ABIDE / 1000 Functional Connectomes](https://fcon_1000.projects.nitrc.org/indi/abide/databases.html).
3. From [NITRC-IR](https://www.nitrc.org/ir/), download for each subject:
   - T1-weighted anatomical
   - resting-state fMRI
4. Also download the phenotypic CSV (`Phenotypic_V1_0b.csv`). You need `SITE_ID`, `SUB_ID`, `DX_GROUP`, and TR.

Expect tens of GB. You need **paired T1 + rest**. The paper used 700 such pairs (320 control / 380 ASD).

## 2. Build `data_splits.json`

Each item must look like this. The `id` format matters because the loader parses it in two places:

```json
{
  "train": [
    {"id": "NYU/sub-0050952", "label": 1}
  ],
  "val": [
    {"id": "PITT/sub-0050003", "label": 2}
  ],
  "test": []
}
```

Rules from `datasets.py`:

| Field | Meaning |
|---|---|
| `id` | `{SITE}/{something}-{SUBJECT}` |
| site | `id.split("/")[0]` → key in `TR.json` (`NYU`, `PITT`, `UCLA_1`, …) |
| fMRI folder | `id.split("/")[-1].split("-")[-1]` → `0050952` |
| T1 stem | same numeric id, file `{id}_T1.nii.gz`, then fallback with the first two chars stripped (`50952_T1.nii.gz`) |
| `label` | **1 or 2** (ASD / control). The loader does `label - 1`. ABIDE `DX_GROUP` is already 1/2 |

Use `split: "all"` first so train+val+test are concatenated. A 6:2:2 split can wait.

Keep `SITE_ID` exactly as in the phenotypic file (`UCLA_1`, not `UCLA`).

## 3. Build `TR.json`

Keys are **site names**, not subject IDs:

```json
{
  "PITT": 1.5,
  "YALE": 1.5,
  "LEUVEN_1": 1.67,
  "NYU": 2.0,
  "SBL": 2.2,
  "KKI": 2.5,
  "UCLA_1": 3.0
}
```

Copy TR from the phenotypic file. Do not guess. Stage 0 buckets subjects by TR:

| Config | TR range |
|---|---|
| `tr_15` | `[1.50, 1.55)` |
| `tr_167` | `[1.60, 1.70)` |
| `tr_20` | `[2.00, 2.05)` |
| `tr_22` | `[2.20, 2.25)` |
| `tr_25` | `[2.50, 2.55)` |
| `tr_30` | `[3.00, 3.05)` |

A site with TR `1.667` goes in `tr_167`. A site with `2.0` goes in `tr_20`. If a site’s TR falls in no bucket, that subject is silently dropped.

## 4. Preprocess T1 → `{subject}_T1.nii.gz`

Paper Appendix A.1, then the extra pad/crop in `ABIDE_T1_Dataset`:

1. Skull-strip with FreeSurfer.
2. `fslreorient2std`.
3. FLIRT to MNI152.
4. Crop to **167 × 212 × 160**.
5. Save `data/abide_i/processed/t1/0050952_T1.nii.gz`.

At load time the code pads `[(6, 9), (2, 4), (0, 22)]` → 182 × 218 × 182, z-scores, then center-crops to **160 × 192 × 160**. If your saved volume is not 167 × 212 × 160, that pad/crop will be wrong.

## 5. Preprocess fMRI → Schaefer-400 CSV

Paper Appendix A.4, then parcellate:

1. De-oblique and reorient.
2. Drop the first few volumes.
3. Slice-timing correction.
4. Motion correction (`mcflirt`).
5. Coregister to that subject’s T1 (`bbregister`).
6. Regress global, WM, CSF, 6 motion params + derivatives.
7. Despike.
8. Band-pass **0.009–0.08 Hz**.
9. Warp to MNI.
10. Average the time series inside each **Schaefer 400, 17-network** ROI (`nilearn` can do this).

Write a gzipped CSV the loader can read:

```python
# 400 rows, first column dropped by df.iloc[:, 1:]
# remaining columns = timepoints
# shape after load: (400, T)
```

Example:

```text
label_name,t0,t1,t2,...
17Networks_LH_Vis_1,0.12,-0.03,...
```

Save as:

```text
data/abide_i/processed/fmri/0050952/0050952_Schaefer17n400p.csv.gz
```

**Length limit.** `target_num_patches=18` and `standard_time=48*0.735` imply:

`patch_size = round(35.28 / TR)`, and `ceil(T / patch_size) <= 18`.

| TR | max T |
|---|---|
| 1.5 s | 432 |
| 1.67 s | 378 |
| 2.0 s | 324 |
| 2.2 s | 288 |
| 2.5 s | 252 |
| 3.0 s | 216 |

If a run is longer, truncate or the `assert` in `pad()` will fire.

## 6. Point the config at your folders

In `configs/harmonizer/stage0_embed/conf_embed_downstream.py`:

```python
fmri_data_dir="data/abide_i/processed/fmri",
T1_data_dir="data/abide_i/processed/t1",
splits_file="data/abide_i/processed/data_splits.json",
tr_file="data/abide_i/processed/TR.json",
```

Also put the Drive checkpoints here before you run Stage 0:

- `checkpoints/harmonix-f/model.pth`
- `checkpoints/harmonix-s/model.pth`

## 7. Sanity-check one subject before a full preprocess

For `{"id": "NYU/sub-0050952", "label": 1}` these must exist:

```text
data/abide_i/processed/fmri/0050952/0050952_Schaefer17n400p.csv.gz
data/abide_i/processed/t1/0050952_T1.nii.gz
```

Then:

```bash
python - <<'PY'
import gzip, json
import nibabel as nib
import pandas as pd

with open("data/abide_i/processed/data_splits.json") as f:
    print(json.load(f)["train"][0])
with open("data/abide_i/processed/TR.json") as f:
    print(json.load(f)["NYU"])

df = pd.read_csv(gzip.open(
    "data/abide_i/processed/fmri/0050952/0050952_Schaefer17n400p.csv.gz", "rt"))
ts = df.iloc[:, 1:].values
print("fmri", ts.shape)          # (400, T)

img = nib.load("data/abide_i/processed/t1/0050952_T1.nii.gz")
print("t1", img.shape)           # (167, 212, 160)
PY
```

Then:

```bash
bash scripts/harmonizer/stage0_embed/run_embed_downstream.sh \
  configs/harmonizer/stage0_embed/conf_embed_downstream.py
```

## Practical order

1. Phenotypic CSV → draft `data_splits.json` + `TR.json` (no imaging yet).
2. Download **one site** (PITT or NYU is small and has a clean TR bucket).
3. Process one subject end-to-end and pass the sanity-check.
4. Scale to the rest of that site, then the other sites.
5. Download checkpoints and run `run_embed_downstream.sh`.

T1/fMRI preprocessing needs **FreeSurfer + FSL** (and usually `nilearn` for Schaefer). That is the slow part; the JSON files are the cheap part.
