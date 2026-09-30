setup-macos:
	pyenv prefix brainharmonix >/dev/null 2>&1 || pyenv virtualenv 3.10 brainharmonix
	PYENV_VERSION=brainharmonix pip install -r requirements-macos.txt
	PYENV_VERSION=brainharmonix pip install -e .

stage-0:
	bash scripts/harmonizer/stage0_embed/run_embed_pretrain.sh configs/harmonizer/stage0_embed/conf_embed_pretrain.py

stage-1:
	bash scripts/harmonizer/stage1_pretrain/run_pretrain.sh base 128

stage-2:
	bash scripts/harmonizer/stage2_finetune/run_finetune.sh base 128 AbideI 0