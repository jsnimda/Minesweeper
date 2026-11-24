"""Report strict acceptance separately from top-hypothesis geometry recovery."""

from collections import Counter
import json
from time import perf_counter

from .evaluation_metrics import BoardDecoding, _rate, _stats
from .validation import ValidationResult, ValidationSettings, validate_geometry


SUPPORT_METRICS = (
    "center_coverage", "vertical_coverage", "horizontal_coverage", "largest_missing_center_cluster",
    "unsupported_center_rows", "unsupported_center_columns", "vertical_weakest_line_continuity",
    "horizontal_weakest_line_continuity", "vertical_longest_unsupported_gap", "horizontal_longest_unsupported_gap",
    "vertical_unsupported_segments", "horizontal_unsupported_segments", "centers_without_boundary_support",
    "confirmed_outer_edges", "max_exterior_energy_fraction", "max_exterior_ridge_coverage",
)
OUTCOME_METRICS = ("accepted", "correct_acceptance", "incorrect_acceptance", "rejected_correct", "correct_rejection")
VALIDATION_FIELDS = (
    "observation_complete", "validation_status", "validation_reasons", "validation_ms",
    *(f"validation_{field}" for field in (*OUTCOME_METRICS, *SUPPORT_METRICS)),
)


def validate_decoding(probabilities, decoding: BoardDecoding, settings: ValidationSettings) -> tuple[ValidationResult, float]:
    start = perf_counter()
    best = decoding.hypotheses[0] if decoding.hypotheses else None
    result = validate_geometry(probabilities, best, settings, fitting_status=decoding.status)
    return result, (perf_counter() - start) * 1000


def validation_record(
    result: ValidationResult, complete_geometry: bool, observation_complete: bool | None, elapsed_ms: float,
) -> dict:
    measurements = result.measurements.copy()
    if measurements:
        measurements["unsupported_center_rows"] = len(measurements["unsupported_center_rows"])
        measurements["unsupported_center_columns"] = len(measurements["unsupported_center_columns"])
        measurements["max_exterior_energy_fraction"] = max(measurements["exterior_energy_fraction"].values())
        measurements["max_exterior_ridge_coverage"] = max(
            (coverage for coverage in measurements["exterior_ridge_coverage"].values() if coverage is not None),
            default=None,
        )
    return {
        "observation_complete": observation_complete,
        "validation_status": result.status,
        "validation_reasons": result.reasons,
        "validation_ms": elapsed_ms,
        "validation_accepted": result.accepted,
        "validation_correct_acceptance": result.accepted and complete_geometry,
        "validation_incorrect_acceptance": result.accepted and not complete_geometry,
        "validation_rejected_correct": not result.accepted and complete_geometry,
        "validation_correct_rejection": not result.accepted and not complete_geometry,
        **{f"validation_{name}": measurements.get(name) for name in SUPPORT_METRICS},
    }


def validation_csv(record: dict) -> dict:
    return dict(record, validation_reasons=json.dumps(record["validation_reasons"]))


def summarize_validation(records: list[dict]) -> dict:
    total = len(records)
    summary = {name: _rate(sum(row[f"validation_{name}"] for row in records), total) for name in OUTCOME_METRICS}
    accepted = summary["accepted"]["count"]
    reasons = Counter(reason for row in records for reason in row["validation_reasons"])
    summary.update(
        rejected=_rate(total - accepted, total),
        geometry_accuracy_among_accepted=_rate(summary["correct_acceptance"]["count"], accepted),
        rejection_reasons={name: _rate(count, total) for name, count in sorted(reasons.items())},
        support={
            name: _stats([row[f"validation_{name}"] for row in records if row[f"validation_{name}"] is not None])
            for name in SUPPORT_METRICS
        },
    )
    labelled_complete = [row for row in records if row["observation_complete"] is True]
    labelled_incomplete = [row for row in records if row["observation_complete"] is False]
    if labelled_complete or labelled_incomplete:
        summary["labelled_completeness"] = {
            "boards": len(labelled_complete) + len(labelled_incomplete),
            "incomplete_board_acceptance": _rate(
                sum(row["validation_accepted"] for row in labelled_incomplete), len(labelled_incomplete),
            ),
            "complete_board_rejection": _rate(
                sum(not row["validation_accepted"] for row in labelled_complete), len(labelled_complete),
            ),
        }
    return summary
