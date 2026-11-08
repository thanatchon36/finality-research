# FINALITY study: results

Settings: 1000 records per coverage cell, bootstrap B = 199, truth from 2,000,000 simulated years, seed 20261108. All parameters are illustrative.

## 1. Attacker target selection

| Venue | Risk-adjusted payoff |
|---|---:|
| Bank core (SWIFT-connected) | -$10.5M |
| Custodial crypto exchange | $42.8M |
| DeFi cross-chain bridge | $110.2M |
| Hardware-wallet phishing | $1.1M |

Bank break-even P(recovery): 0.625. Log-gap bank to bridge +2.246, of which finality +1.504 (67%).

## 2. Systemic loss model

Mean annual loss $464.8M; VaR 99.9% $14.23B; ES 99.9% $35.55B. Truth from the long run: 1-in-100 $3.53B, 1-in-1,000 $14.39B, a $20B year once per 1,860 years.

## 3. EVT on single records

1,000-year record: xi = 1.047 (95% CI 0.60 to 1.50; planted 0.625); 1-in-1,000 estimate $32.24B.

100-year record: xi = 0.584, 1-in-1,000 estimate $11.58B (truth $14.39B).

| Method | 95% interval for the 1-in-1,000 loss |
|---|---|
| delta | $1.59B to $108.68B |
| profile | $3.48B to $1.02T |
| bootstrap | $1.89B to $131.76B |

## 4. Coverage study (nominal 95%)

| Exp. | Record | mean k | T | Method | Coverage (±1.96 SE) | Truth above interval | Median upper/lower | Unbounded |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| A | 30 y | 15 | 100 | delta | 73.0% ± 2.8% | 27.0% | 14.6× | 0% |
| A | 30 y | 15 | 100 | profile | 89.5% ± 1.9% | 8.7% | 54.8× | 0% |
| A | 30 y | 15 | 100 | bootstrap | 72.5% ± 2.8% | 27.5% | 13.6× | 0% |
| A | 30 y | 15 | 1,000 | delta | 71.3% ± 2.8% | 28.7% | 212.7× | 0% |
| A | 30 y | 15 | 1,000 | profile | 90.4% ± 1.8% | 8.2% | 1654.2× | 0% |
| A | 30 y | 15 | 1,000 | bootstrap | 72.8% ± 2.8% | 27.2% | 137.1× | 0% |
| A | 100 y | 50 | 100 | delta | 85.6% ± 2.2% | 14.4% | 5.7× | 0% |
| A | 100 y | 50 | 100 | profile | 93.4% ± 1.5% | 4.7% | 7.2× | 0% |
| A | 100 y | 50 | 100 | bootstrap | 85.9% ± 2.2% | 13.9% | 5.5× | 0% |
| A | 100 y | 50 | 1,000 | delta | 85.1% ± 2.2% | 14.7% | 32.4× | 0% |
| A | 100 y | 50 | 1,000 | profile | 92.9% ± 1.6% | 5.1% | 45.5× | 0% |
| A | 100 y | 50 | 1,000 | bootstrap | 85.0% ± 2.2% | 14.7% | 29.7× | 0% |
| A | 300 y | 150 | 100 | delta | 92.4% ± 1.6% | 7.2% | 2.9× | 0% |
| A | 300 y | 150 | 100 | profile | 94.1% ± 1.5% | 4.0% | 3.0× | 0% |
| A | 300 y | 150 | 100 | bootstrap | 91.3% ± 1.7% | 7.9% | 2.9× | 0% |
| A | 300 y | 150 | 1,000 | delta | 92.1% ± 1.7% | 7.4% | 8.4× | 0% |
| A | 300 y | 150 | 1,000 | profile | 94.4% ± 1.4% | 3.8% | 8.7× | 0% |
| A | 300 y | 150 | 1,000 | bootstrap | 91.2% ± 1.8% | 7.8% | 8.0× | 0% |
| B | 30 y | 18 | 100 | delta | 58.8% ± 3.1% | 40.9% | 6.3× | 0% |
| B | 30 y | 18 | 100 | profile | 80.7% ± 2.4% | 16.2% | 18.3× | 0% |
| B | 30 y | 18 | 100 | bootstrap | 60.5% ± 3.0% | 39.0% | 6.2× | 0% |
| B | 30 y | 18 | 1,000 | delta | 60.2% ± 3.0% | 39.8% | 47.5× | 0% |
| B | 30 y | 18 | 1,000 | profile | 82.9% ± 2.3% | 15.7% | 285.5× | 0% |
| B | 30 y | 18 | 1,000 | bootstrap | 62.3% ± 3.0% | 37.7% | 43.8× | 0% |
| B | 100 y | 20 | 100 | delta | 76.8% ± 2.6% | 23.1% | 5.6× | 0% |
| B | 100 y | 20 | 100 | profile | 90.3% ± 1.8% | 8.3% | 11.0× | 0% |
| B | 100 y | 20 | 100 | bootstrap | 75.7% ± 2.7% | 24.2% | 5.4× | 0% |
| B | 100 y | 20 | 1,000 | delta | 80.6% ± 2.5% | 19.4% | 85.8× | 0% |
| B | 100 y | 20 | 1,000 | profile | 92.1% ± 1.7% | 6.8% | 256.8× | 0% |
| B | 100 y | 20 | 1,000 | bootstrap | 81.6% ± 2.4% | 18.4% | 77.0× | 0% |
| B | 1,000 y | 80 | 100 | delta | 88.1% ± 2.0% | 11.7% | 2.0× | 0% |
| B | 1,000 y | 80 | 100 | profile | 93.2% ± 1.6% | 5.7% | 2.1× | 0% |
| B | 1,000 y | 80 | 100 | bootstrap | 87.4% ± 2.1% | 12.2% | 2.0× | 0% |
| B | 1,000 y | 80 | 1,000 | delta | 97.1% ± 1.0% | 1.4% | 9.4× | 0% |
| B | 1,000 y | 80 | 1,000 | profile | 92.5% ± 1.6% | 0.3% | 11.2× | 0% |
| B | 1,000 y | 80 | 1,000 | bootstrap | 96.1% ± 1.2% | 1.5% | 8.9× | 0% |

## 5. Network contagion (seed 7, identical to the web instrument)

rho(K W) = 1.1519; critical exposure 0.868× equity. Most systemic: C1 ×3.71, P1 ×3.04, C2 ×2.93; the most systemic bank is number 5 by size.

| Scenario | Direct | After contagion | Multiplier | Defaults | Rounds |
|---|---:|---:|---:|---:|---:|
| C5 hit, exposure 0.5x | $26.68B | $68.95B | ×2.58 | 0 | 14 |
| vendor hit, exposure 0.5x | $19.78B | $25.01B | ×1.26 | 0 | 10 |
| vendor hit, exposure 1.2x | $19.78B | $643.83B | ×32.55 | 13 | 14 |

Runtime 3s.
