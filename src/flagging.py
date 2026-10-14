"""
flagging.py -- Deterministic flag logic.

Flags route-weeks where cost is rising vs own history AND is outside
normal range vs own history or peers.
"""

from typing import Optional

import numpy as np
import pandas as pd


def should_flag(
    vs_own_history_pct: Optional[float],
    vs_peer_pct: Optional[float],
    config: dict,
) -> bool:
    """
    Determine whether a route-week should be flagged.

    Flag rule (all must hold):
    1. vs_own_history_pct is not None (we have historical data)
    2. Cost is rising vs own history (pct > rising_threshold, i.e., > 0%)
    3. Cost is outside normal range vs own history OR vs peers:
       - vs_own_history_pct > outlier_threshold (default 15%), OR
       - vs_peer_pct > peer_outlier_threshold (default 15%)
    Returns True if the route-week should be flagged.
    """
    rising_threshold = config.get("flag_own_history_rising_threshold", 0.0)
    own_outlier = config.get("flag_own_history_outlier_threshold", 15.0)
    peer_outlier = config.get("flag_peer_outlier_threshold", 15.0)
    # Cannot flag without historical baseline
    if vs_own_history_pct is None or (
        isinstance(vs_own_history_pct, float) and np.isnan(vs_own_history_pct)
    ):
        return False

    # Must be rising
    if vs_own_history_pct <= rising_threshold:
        return False

    # Must be outside normal range vs own history OR peers
    own_outside = vs_own_history_pct > own_outlier
    peer_outside = (
        vs_peer_pct is not None
        and not (isinstance(vs_peer_pct, float) and np.isnan(vs_peer_pct))
        and vs_peer_pct > peer_outlier
    )

    return own_outside or peer_outside


def apply_flags(weekly: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply flagging logic to the weekly metrics table.

    Adds a 'flagged_bool' column (True/False) to the DataFrame.
    The final 'flagged' string column is set later after note matching.
    """
    result = weekly.copy()
    result["flagged_bool"] = result.apply(
        lambda r: should_flag(
            r.get("vs_own_history_pct"),
            r.get("vs_peer_pct"),
            config,
        ),
        axis=1,
    )
    return result
