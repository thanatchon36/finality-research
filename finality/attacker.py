"""Rational-choice model of attacker venue selection (security economics).

    E[payoff] = P_breach * V * (1 - P_recovery) * (1 - haircut) - P_attribution * sanction - cost

The retained-value term is a product of four factors, so the log-ratio of two venues'
retained value splits *exactly* into one additive contribution per factor. That
decomposition is the test of H1 (finality dominance): which share of the bank-vs-crypto
payoff gap is explained by finality, (1 - P_recovery), rather than by breach difficulty.

All parameter values are ILLUSTRATIVE. Calibrate against Chainalysis, DeFiLlama/REKT
and FBI IC3 data before citing any number.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

USD_M = 1e6


@dataclass(frozen=True)
class Venue:
    name: str
    p_breach: float        # P(successful compromise | serious attempt)
    value: float           # USD reachable in one successful breach
    p_recovery: float      # P(funds frozen or clawed back): the finality lever
    haircut: float         # share of value lost to laundering / cash-out
    p_attribution: float   # P(attacker identified and punished)
    sanction: float        # USD-equivalent penalty if punished
    cost: float            # cost of mounting the attack

    def retained(self) -> float:
        return self.p_breach * self.value * (1 - self.p_recovery) * (1 - self.haircut)

    def payoff(self) -> float:
        return self.retained() - self.p_attribution * self.sanction - self.cost

    def break_even_recovery(self) -> float:
        """P_recovery at which the attacker's expected payoff is exactly zero."""
        gross = self.p_breach * self.value * (1 - self.haircut)
        return 1.0 - (self.p_attribution * self.sanction + self.cost) / gross


VENUES: tuple[Venue, ...] = (
    Venue("Bank core (SWIFT-connected)", 0.15, 1000 * USD_M, 0.80, 0.60, 0.35, 50 * USD_M, 5 * USD_M),
    Venue("Custodial crypto exchange", 0.20, 500 * USD_M, 0.25, 0.35, 0.20, 20 * USD_M, 2 * USD_M),
    Venue("DeFi cross-chain bridge", 0.30, 600 * USD_M, 0.10, 0.30, 0.15, 15 * USD_M, 1 * USD_M),
    Venue("Hardware-wallet phishing", 0.05, 50 * USD_M, 0.02, 0.25, 0.10, 5 * USD_M, 0.2 * USD_M),
)

FACTORS = ("breach probability", "value at risk", "finality (1 - P_recovery)", "laundering (1 - haircut)")


def log_gap_decomposition(a: Venue, b: Venue) -> dict[str, float]:
    """Exact additive split of ln(retained_b / retained_a) into its four multiplicative factors."""
    parts = {
        FACTORS[0]: float(np.log(b.p_breach / a.p_breach)),
        FACTORS[1]: float(np.log(b.value / a.value)),
        FACTORS[2]: float(np.log((1 - b.p_recovery) / (1 - a.p_recovery))),
        FACTORS[3]: float(np.log((1 - b.haircut) / (1 - a.haircut))),
    }
    parts["total"] = float(np.log(b.retained() / a.retained()))
    return parts


def recovery_curve(v: Venue, grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Attacker payoff as the venue's recovery capability varies."""
    grid = np.linspace(0.0, 0.95, 96) if grid is None else grid
    return grid, np.array([replace(v, p_recovery=float(r)).payoff() for r in grid])
