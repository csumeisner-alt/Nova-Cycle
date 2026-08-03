"""Single source of truth for the short-horizon "rally event" definition.

A rally event, observed from a valid 5-minute VOO candle at time t, means:

    max(close[t+1 : t+HORIZON_BARS]) / close[t] - 1  >  RISE_PERCENT / 100

i.e. the price reaches at least +0.3% above the observation close at ANY
point during the next 12 five-minute bars (~1 hour), using closing prices.

Everything that measures or predicts this event MUST import the constants
and label builder from this module so the training label, walk-forward
evaluation, missed-rally reporting, and backtests all describe the same
event.  The historical bug this prevents: the short model was trained on
"close exactly 12 bars later is +0.3% higher" while the missed-rally
detector used "any close within 12 bars is +0.3% higher" — the model and
the dashboard were measuring different things.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The tradable upside move: +0.3% from the observation close.
RALLY_RISE_PERCENT: float = 0.3

# Evaluation horizon: 12 five-minute bars (~1 hour of regular trading).
RALLY_HORIZON_BARS: int = 12

# Fraction form of the rise threshold, for label math.
RALLY_RISE_FRACTION: float = RALLY_RISE_PERCENT / 100.0


def rally_event_labels(close: pd.Series) -> pd.Series:
    """Binary rally-event labels for a chronological series of 5-min closes.

    label[t] = 1 when the maximum close over the NEXT `RALLY_HORIZON_BARS`
    bars exceeds close[t] by more than `RALLY_RISE_FRACTION`.

    The final `RALLY_HORIZON_BARS` rows have incomplete future windows; they
    are returned as NaN so callers can drop them instead of training on
    truncated look-ahead windows.
    """
    if close.empty:
        return pd.Series(dtype=float, index=close.index)

    # Forward-looking max of the next HORIZON bars, excluding the current bar:
    # reverse the series, take a rolling max of the window ending at each row,
    # then shift so the window covers (t, t+H].
    future_max = (
        close[::-1]
        .rolling(RALLY_HORIZON_BARS, min_periods=RALLY_HORIZON_BARS)
        .max()[::-1]
        .shift(-1)
    )
    labels = (future_max / close - 1.0 > RALLY_RISE_FRACTION).astype(float)
    labels[future_max.isna() | (close <= 0)] = np.nan
    return labels
