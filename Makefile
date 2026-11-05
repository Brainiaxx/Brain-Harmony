setup-macos:
	pyenv prefix brainharmonix >/dev/null 2>&1 || pyenv virtualenv 3.10 brainharmonix
	PYENV_VERSION=brainharmonix pip install -r requirements-macos.txt
	PYENV_VERSION=brainharmonix pip install -e .

# ABIDE-I downstream embeddings. The UK Biobank pretrain script is
# scripts/harmonizer/stage0_embed/run_embed_pretrain.sh.
stage-0:
	bash scripts/harmonizer/stage0_embed/run_embed_downstream.sh configs/harmonizer/stage0_embed/conf_embed_downstream.py

stage-1:
	bash scripts/harmonizer/stage1_pretrain/run_pretrain.sh base 128

stage-2:
	bash scripts/harmonizer/stage2_finetune/run_finetune.sh base 128 AbideI 0

# Paired ABIDE-I T1 and resting-state fMRI (~64 GB) from the public FCP-INDI
# bucket into data/abide_i/raw/. Re-running resumes files that are already
# complete. Optional site filter uses the phenotype SITE_ID, for example:
#   make download-abide SITE=PITT
SITE ?=

PY ?= $(HOME)/.pyenv/versions/brainharmonix/bin/python

.PHONY: download-abide prepare-abide preprocess-abide
download-abide:
	python3 -u scripts/download_abide_raw.py $(if $(SITE),--site $(SITE),)

# TR.json, seed-0 6:2:2 split, and the raw-file manifest.
# preprocess-abide removes subjects that still lack both outputs from this split.
prepare-abide:
	$(PY) -u scripts/prepare_abide_metadata.py

# Cropped T1s and Schaefer-400 time series. Optional: SITE=PITT LIMIT=1 JOBS=1
LIMIT ?=
JOBS ?= 1
preprocess-abide:
	$(PY) -u scripts/preprocess_abide.py $(if $(SITE),--site $(SITE),) $(if $(LIMIT),--limit $(LIMIT),) --jobs $(JOBS)