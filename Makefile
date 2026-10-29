.PHONY: install test lint type reports validate audit schemas repro check app phase11

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check .

type:
	python -m mypy src/neurobranch/benchmark src/neurobranch/design src/neurobranch/uncertainty.py

reports:
	python scripts/phase10_reliability.py
	python scripts/phase10_figure.py
	python scripts/phase10_robustness.py
	python scripts/phase10_design.py
	python scripts/phase10_claims.py

phase11:
	python scripts/phase11_sensitivity.py
	python scripts/phase11_benchmark.py
	python scripts/phase11_manifest.py

validate:
	python scripts/validate_reports.py --strict

audit:
	python scripts/audit_claims.py

schemas:
	python scripts/sync_schemas.py --check

repro:
	python scripts/check_reproducibility.py
	python scripts/check_reproducibility.py --reports reports/phase11

app:
	python -m streamlit run app/streamlit_app.py

check: lint type test validate audit schemas repro
