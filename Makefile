.PHONY: install test lint type reports validate repro check app

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check .

type:
	python -m mypy src/neurobranch/benchmark src/neurobranch/design

reports:
	python scripts/phase10_reliability.py
	python scripts/phase10_figure.py
	python scripts/phase10_robustness.py
	python scripts/phase10_design.py
	python scripts/phase10_claims.py

validate:
	python scripts/validate_reports.py --strict

repro:
	python scripts/check_reproducibility.py

app:
	python -m streamlit run app/streamlit_app.py

check: lint type test validate repro
