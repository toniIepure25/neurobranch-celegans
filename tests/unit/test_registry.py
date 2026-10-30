from __future__ import annotations

from neurobranch.config import PROJECT_ROOT
from neurobranch.data.registry import load_registry, registry_report

KATO_PATH = PROJECT_ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"


def test_dataset_registry_is_valid_and_reports_availability() -> None:
    specs = load_registry()
    keys = {spec.key for spec in specs}
    assert {"kato2015", "randi2023", "kramer2026", "baskoylu2026", "simeon2024"} <= keys
    report = registry_report()
    assert report["n_datasets"] == len(specs)
    by_key = {row["key"]: row for row in report["datasets"]}
    assert by_key["kato2015"]["available"] is KATO_PATH.exists()
    assert by_key["simeon2024"]["available"] is False
    assert by_key["simeon2024"]["allowed_uses"] == ["external_only"]
