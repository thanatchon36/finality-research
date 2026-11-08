"""Interbank contagion of cyber losses with linear DebtRank (Bardoscia et al. 2015).

    h_i(t+1) = min(1, h_i(t) + sum_j Lambda_ij * [h_j(t) - h_j(t-1)]),   Lambda = lev * K W

h_i is the share of bank i's equity lost, W_ij the share of i's interbank claims that sit on j,
K = diag(k_i) each bank's relative interbank intensity and lev the system-wide scale of
exposure / equity. Shocks stop dying out once the spectral radius rho(Lambda) exceeds 1
(Bardoscia et al. 2017), i.e. above the critical exposure lev* = 1 / rho(K W).

`build_network` reproduces the FINALITY web instrument exactly: same PRNG (mulberry32), same
draw order, same stopping rule, so both implementations can be cross-checked on seed 7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_M32 = 0xFFFFFFFF


def mulberry32(seed: int):
    """Bit-exact port of the JavaScript mulberry32 generator used by the web instrument."""
    a = seed & _M32

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & _M32
        t = ((a ^ (a >> 15)) * (1 | a)) & _M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & _M32)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    return rnd


@dataclass
class Network:
    names: list[str]
    core: np.ndarray     # bool, core bank
    ta: np.ndarray       # total assets, USD
    eq: np.ndarray       # equity, USD
    k: np.ndarray        # relative interbank intensity
    W: np.ndarray        # row-stochastic lending weights, W[i, j] = share of i's claims on j
    perm: np.ndarray     # random order used to pick shared-vendor clients

    @property
    def n(self) -> int:
        return int(self.ta.size)

    def Lambda(self, lev: float) -> np.ndarray:
        return lev * self.k[:, None] * self.W

    @property
    def rho1(self) -> float:
        """Spectral radius of K W, i.e. rho(Lambda) at unit exposure."""
        return float(np.max(np.abs(np.linalg.eigvals(self.k[:, None] * self.W))))

    @property
    def critical_exposure(self) -> float:
        return 1.0 / self.rho1

    def largest(self) -> int:
        return int(np.argmax(self.ta))

    def vendor_clients(self, reach: float) -> np.ndarray:
        return self.perm[:max(1, int(math.floor(reach * self.n + 0.5)))]


def build_network(seed: int = 7, n_core: int = 5, n: int = 28) -> Network:
    """Core-periphery interbank network, identical to the web instrument for the same seed."""
    R = mulberry32(seed)

    def gz() -> float:
        a = R()
        b = R()
        return math.sqrt(-2.0 * math.log(a + 1e-12)) * math.cos(2.0 * math.pi * b)

    core = np.arange(n) < n_core
    ta, eq, k, names = np.empty(n), np.empty(n), np.empty(n), []
    ci = pi = 0
    for i in range(n):
        if core[i]:
            ta[i] = 8e11 * math.exp(R() * math.log(3.75))
            eq[i] = ta[i] * (0.04 + R() * 0.04)
            k[i] = 1.0 + R() * 0.5
            ci += 1
            names.append(f"C{ci}")
        else:
            ta[i] = 1.5e10 * math.exp(R() * math.log(250 / 15))
            eq[i] = ta[i] * (0.06 + R() * 0.08)
            k[i] = 0.2 + R() * 0.8
            pi += 1
            names.append(f"P{pi}")
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = 0.95 if (core[i] and core[j]) else (0.2 if (core[i] or core[j]) else 0.035)
            if R() < p:
                W[i, j] = ta[j] * math.exp(0.9 * gz())
    for i in range(n_core, n):
        lends, borrows = bool((W[i, :n_core] > 0).any()), bool((W[:n_core, i] > 0).any())
        if not lends:
            j = int(math.floor(R() * n_core))
            W[i, j] = ta[j] * math.exp(0.9 * gz())
        if not borrows:
            j = int(math.floor(R() * n_core))
            W[j, i] = ta[i] * math.exp(0.9 * gz())
    s = W.sum(axis=1, keepdims=True)
    W = np.divide(W, s, out=np.zeros_like(W), where=s > 0)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(math.floor(R() * (i + 1)))
        perm[i], perm[j] = perm[j], perm[i]
    return Network(names, core, ta, eq, k, W, np.array(perm))


def debtrank(net: Network, lev: float, seeds, shock: float, max_rounds: int = 400,
             rtol: float = 2e-4) -> np.ndarray:
    """Linear DebtRank history, shape (rounds + 2, n): row 0 before the shock, row 1 the direct hit."""
    h = np.zeros(net.n)
    h[np.asarray(seeds, int)] = min(1.0, shock)
    prev = np.zeros(net.n)
    hist = [np.zeros(net.n), h.copy()]
    for _ in range(max_rounds):
        d = np.clip(h - prev, 0.0, None)
        acc = net.W @ d
        upd = (h < 1.0) & (acc > 0)
        nh = h.copy()
        nh[upd] = np.minimum(1.0, h[upd] + lev * net.k[upd] * acc[upd])
        moved = float((nh - h)[upd].sum())
        prev, h = h, nh
        if moved < rtol * float(nh.sum()) + 1e-9:
            break
        hist.append(h.copy())
    return np.array(hist)


def equity_loss(net: Network, h: np.ndarray) -> float:
    return float(h @ net.eq)


def multiplier(net: Network, hist: np.ndarray) -> float:
    return equity_loss(net, hist[-1]) / equity_loss(net, hist[1])


def phase_sweep(net: Network, seeds, shock: float, levs: np.ndarray) -> np.ndarray:
    return np.array([multiplier(net, debtrank(net, L, seeds, shock)) for L in levs])


def systemic_ranking(net: Network, lev: float, shock: float) -> np.ndarray:
    """Contagion multiplier when each bank alone is hit (DebtRank-style systemic importance)."""
    return np.array([multiplier(net, debtrank(net, lev, [i], shock)) for i in range(net.n)])
