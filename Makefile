setup-macos:
	pyenv prefix brainharmonix >/dev/null 2>&1 || pyenv virtualenv 3.10 brainharmonix
	PYENV_VERSION=brainharmonix pip install -r requirements-macos.txt