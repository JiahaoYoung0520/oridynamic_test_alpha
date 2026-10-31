.PHONY: install test build
install:
	python -m pip install -e .
test:
	PYTHONDONTWRITEBYTECODE=1 LAYERNORM_TYPE=torch PROTENIX_DISABLE_FUSED_LN=1 python -B -m unittest discover -s tests -v
build:
	python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
