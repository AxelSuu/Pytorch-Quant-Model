"""A reusable multi-symbol, multi-configuration sweep harness (PYQ-268).

``backlog/README.md``'s ``## Now`` list carried the same item at #1 across
two passes: a multi-symbol repeat of PYQ-247's target comparison and of
investigations.md#pyq-315/#pyq-316's pooling and feature-ablation findings.
``scripts/ablate_features.py`` and ``scripts/compare_pooling.py`` are each
hard-wired one-off investigation scripts; repeating either across fifteen
symbols meant editing a script and reconciling its output by hand. This
package is the missing instrument, not a run -- see :mod:`~pyquant.experiments.sweep`.
"""
