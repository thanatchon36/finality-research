import numpy as np
import pytest

from finality import attacker, lda
from finality.evt import hill


def test_bank_is_loss_making_and_bridge_is_best():
    pay = [v.payoff() for v in attacker.VENUES]
    assert pay[0] < 0 and int(np.argmax(pay)) == 2


def test_log_gap_decomposition_is_exact():
    d = attacker.log_gap_decomposition(attacker.VENUES[0], attacker.VENUES[2])
    assert sum(v for k, v in d.items() if k != "total") == pytest.approx(d["total"], abs=1e-12)
    assert d["finality (1 - P_recovery)"] / d["total"] > 0.5


def test_break_even_recovery_zeroes_the_payoff():
    from dataclasses import replace
    b = attacker.VENUES[0]
    assert replace(b, p_recovery=b.break_even_recovery()).payoff() == pytest.approx(0, abs=1e-3)


def test_annual_losses_are_sums_of_incidents():
    sim = lda.simulate(2000, lda.LDAParams(), np.random.default_rng(7))
    assert sim.annual.sum() == pytest.approx(sim.incidents.sum(), rel=1e-12)
    inc, ann = sim.window(100, 50)
    assert inc.sum() == pytest.approx(ann.sum(), rel=1e-12)


def test_planted_tail_index_is_recoverable():
    # thin body and no cascades: the top incidents are pure Pareto. With cascades on, the trigger
    # creates a kink, and the local index sits below the asymptotic one at moderate thresholds.
    p = lda.LDAParams(body_sigma=0.5, c_prob=0.0)
    sim = lda.simulate(50_000, p, np.random.default_rng(8))
    assert hill(sim.incidents, 3000)[1499] == pytest.approx(p.tail_alpha, abs=0.15)
