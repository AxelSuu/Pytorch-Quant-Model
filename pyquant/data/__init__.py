"""Vendor fetches, feature engineering and panel assembly.

Four external sources (Yahoo Finance, FRED, Finnhub, sector ETFs) are joined into
a single date-indexed daily panel. Two properties matter more than anything else
in this package:

- **No look-ahead.** Every column must be knowable at its own row's timestamp —
  hence FRED's publication-lag handling, sentiment's session-based join, and the
  rule that indicator warm-up rows stay NaN and are dropped rather than filled.
- **Graceful degradation at train time only.** A missing or rate-limited source
  is dropped with a logged notice while training; at predict time a missing
  trained feature is a hard ``FeatureSchemaMismatch``, because a model cannot run
  without the columns it was trained on.
"""
