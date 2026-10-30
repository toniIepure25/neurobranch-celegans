"""Read the newest measurement-audit status for the dashboard.

Older phases are retained as historical fallbacks.  Missing gates are returned
as missing, never as passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHASES = ("phase14", "phase13", "phase12r", "phase12")


@dataclass(frozen=True)
class MeasurementStatus:
    phase: str
    path: Path
    decision: str
    model: str
    primary_estimate: float
    reference_estimate: float | None
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_gates: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    checksum: str = "missing"
    historical: bool = False

    @property
    def adopted(self) -> bool:
        return self.decision == "adopted"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalise_phase14(payload: dict[str, Any]) -> tuple[str, str, float, float | None, dict[str, dict[str, Any]]]:
    decision = str(payload.get("decision", "unknown"))
    model = str(payload.get("model", "unknown"))
    real = payload.get("real_fit", {})
    primary = float(real.get("pooled_reliability_median_n22", float("nan")))
    reference_block = payload.get("reference", {})
    reference = reference_block.get("v1_pooled_reliability_at_fold")
    gates = payload.get("gates", {})
    return (
        decision,
        model,
        primary,
        float(reference) if reference is not None else None,
        gates if isinstance(gates, dict) else {},
    )


def _normalise_phase13(payload: dict[str, Any]) -> tuple[str, str, float, float | None, dict[str, dict[str, Any]]]:
    decision = str(payload.get("decision", "unknown"))
    model = str(payload.get("model", "m2_event_variance"))
    primary = float(payload.get("primary_estimate", float("nan")))
    reference = payload.get("reference_estimate")
    gates = payload.get("gates", {})
    return (
        decision,
        model,
        primary,
        float(reference) if reference is not None else None,
        gates if isinstance(gates, dict) else {},
    )


def load_measurement_status(reports_root: Path) -> MeasurementStatus | None:
    """Return the newest available measurement status, or ``None``.

    Phase 14 and Phase 13 reports are authoritative.  Phase 12R/12 are shown
    only as historical fallbacks and are labelled as such.
    """
    for index, phase in enumerate(PHASES):
        directory = reports_root / phase
        summary = _read_json(directory / "summary.json")
        if summary is None:
            continue
        if phase == "phase14":
            decision, model, primary, reference, gates = _normalise_phase14(summary)
        elif phase == "phase13":
            decision, model, primary, reference, gates = _normalise_phase13(summary)
        else:
            decision = str(summary.get("status", summary.get("decision", "unknown")))
            model = str(summary.get("model", "m0_reference"))
            hierarchical = summary.get("hierarchical", {})
            primary = float(hierarchical.get("pooled_reliability_median", float("nan")))
            reference = None
            raw_gates = summary.get("gates", {})
            gates = {
                name: {"passed": bool(value), "value": value, "requirement": "", "note": ""}
                for name, value in raw_gates.items()
                if isinstance(value, bool)
            }
        failed = [name for name, gate in gates.items() if not bool(gate.get("passed", False))]
        manifest = _read_json(directory / "manifest.json") or {}
        checksum = "missing"
        checksum_path = directory / "manifest.sha256"
        if checksum_path.is_file():
            checksum = checksum_path.read_text(encoding="utf-8").split()[0][:12]
        return MeasurementStatus(
            phase=phase,
            path=directory,
            decision=decision,
            model=model,
            primary_estimate=primary,
            reference_estimate=reference,
            gates=gates,
            failed_gates=failed,
            manifest=manifest,
            checksum=checksum,
            historical=index >= 2,
        )
    return None
