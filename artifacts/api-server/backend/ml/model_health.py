"""
NovaCycle Model Health Checks
==============================
Post-training sanity checks that catch a "quietly broken" model — one that
trains without error but degenerates to a constant predictor (identical
probability for every input, zero feature importances) while still reporting
the base-rate accuracy.

Used by the trainer after each training run; a degenerate model is recorded
as a failed training via ml/training_status.py so the existing health
surface picks it up.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum standard deviation of predict_proba across the training set.
# A healthy model spreads probabilities well beyond this; a constant
# predictor has std == 0 (or numerically indistinguishable from it).
MIN_PROBA_STD = 1e-4

# Sample cap so the check stays cheap on large training sets.
_MAX_SAMPLE = 2000


def check_model_degeneracy(model, X: np.ndarray) -> Tuple[bool, Optional[str]]:
    """Return (degenerate, reason) for a freshly trained classifier.

    Checks:
      1. predict_proba varies across the training set (std > MIN_PROBA_STD)
      2. feature_importances_ (when the model exposes them) are not all zero

    Never raises — a failure to run the check is logged and treated as
    non-degenerate so it cannot break training itself.
    """
    try:
        if model is None or X is None or len(X) < 2:
            return False, None

        X_check = X[-_MAX_SAMPLE:] if len(X) > _MAX_SAMPLE else X

        # ── Prediction variance ───────────────────────────────────────────
        probs = model.predict_proba(X_check)
        pos = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
        proba_std = float(np.std(pos))
        if proba_std < MIN_PROBA_STD:
            return True, (
                f"constant predictions: predict_proba std={proba_std:.2e} "
                f"< {MIN_PROBA_STD:.0e} across {len(X_check)} training rows"
            )

        # ── Feature importances (tree models) ────────────────────────────
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            imp = np.asarray(importances, dtype=float)
            if imp.size > 0 and float(np.abs(imp).sum()) == 0.0:
                return True, "all feature importances are zero"

        return False, None
    except Exception as exc:
        logger.error("check_model_degeneracy error: %s", exc)
        return False, None
