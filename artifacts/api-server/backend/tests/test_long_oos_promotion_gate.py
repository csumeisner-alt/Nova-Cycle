"""Long-model promotion must require complete purged walk-forward evidence."""

from ml.trainer import _long_oos_gate_failure


def _result(target_type="direction", **overrides):
    result = {
        "target_type": target_type,
        "accuracy_metric": (
            "purged_walk_forward_multiclass"
            if target_type == "three_state"
            else "purged_walk_forward_oos"
        ),
        "calibration": {
            "evaluated": True,
            "oos_accuracy": 0.55,
            "accuracy_lift_vs_majority": 0.05,
            "macro_f1": 0.45,
            "per_class": [
                {"label": "risk_off", "f1": 0.30},
                {"label": "neutral", "f1": 0.40},
                {"label": "risk_on", "f1": 0.50},
            ],
            "precision_lift_vs_base_rate": 2.5,
        },
        "pr_auc_lift_vs_prevalence": 2.5,
    }
    result.update(overrides)
    return result


def test_missing_walk_forward_metric_is_rejected():
    result = _result(accuracy_metric="train")
    reason = _long_oos_gate_failure(result)
    assert reason is not None
    assert "honest purged walk-forward" in reason


def test_incomplete_walk_forward_evaluation_is_rejected():
    result = _result(calibration={"evaluated": False, "reason": "not enough rows"})
    reason = _long_oos_gate_failure(result)
    assert reason is not None
    assert "incomplete" in reason
    assert "not enough rows" in reason


def test_missing_target_specific_metric_is_rejected():
    result = _result(target_type="drawdown_event")
    result["calibration"].pop("precision_lift_vs_base_rate")
    reason = _long_oos_gate_failure(result)
    assert reason is not None
    assert "missing metrics" in reason
    assert "precision_lift_vs_base_rate" in reason


def test_three_state_requires_per_class_walk_forward_metrics():
    result = _result(target_type="three_state")
    result["per_class"] = None
    result["calibration"].pop("per_class")
    reason = _long_oos_gate_failure(result)
    assert reason is not None
    assert "macro-F1 or per-class" in reason


def test_complete_oos_evidence_is_accepted_by_the_boundary_gate():
    assert _long_oos_gate_failure(_result()) is None
    assert _long_oos_gate_failure(_result(target_type="drawdown_event")) is None
    assert _long_oos_gate_failure(_result(target_type="three_state")) is None
