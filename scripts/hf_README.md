---
license: cc-by-4.0
pretty_name: "finality: simulated data for extreme-value uncertainty and contagion in banking-sector cyber risk"
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - tabular-regression
tags:
  - cyber-risk
  - extreme-value-theory
  - return-levels
  - profile-likelihood
  - interval-coverage
  - monte-carlo
  - systemic-risk
  - debtrank
  - finance
  - synthetic
configs:
  - config_name: coverage_intervals
    data_files: coverage/intervals.csv
    default: true
  - config_name: coverage_summary
    data_files: coverage/summary.csv
  - config_name: annual_losses
    data_files: loss/annual_losses_200k.csv
  - config_name: truth_return_levels
    data_files: loss/truth_return_levels.csv
  - config_name: record_1000y
    data_files: loss/record_1000y_incidents.csv
  - config_name: record_100y
    data_files: loss/record_100y_incidents.csv
  - config_name: venues
    data_files: attacker/venues.csv
  - config_name: banks
    data_files: contagion/banks.csv
  - config_name: exposures
    data_files: contagion/exposures.csv
  - config_name: phase_sweep
    data_files: contagion/phase_sweep.csv
  - config_name: cascades
    data_files: contagion/cascades.csv
---

# finality: simulated data for extreme-value uncertainty and contagion in banking-sector cyber risk

> **Every number here is simulated from illustrative parameters.** Nothing in this dataset is an observed
> cyber loss, a real bank or a real exposure. The data exist to audit statistical methods against a known truth
> and to reproduce the paper; recalibrate with real incident data before citing any number.

Data for the preprint *False Confidence in the Tail: Finality, Extreme-Value Uncertainty, and Contagion in
Banking-Sector Cyber Risk* by Thanatchon Chongmankhong ([read online](https://thanatchon36.github.io/finality-research/paper/) ·
[PDF](https://github.com/thanatchon36/finality-research/blob/main/paper/paper.pdf)).
The paper asks how confidently one can state the return period of a catastrophic banking-sector cyber loss from a
few decades of data. A loss process with a planted Pareto tail gives a known truth; 36,000 interval estimates for
the 1-in-100 and 1-in-1,000-year loss are scored against it. A rational-choice attacker model and a linear
DebtRank interbank network complete the framework. Code: <https://github.com/thanatchon36/finality-research>;
archived on Zenodo: [10.5281/zenodo.24050729](https://doi.org/10.5281/zenodo.24050729).

*Independent personal research. Defensive, actuarial and policy research: the models describe attacker incentives
and aggregate losses only, and the dataset contains no attack technique.*

## Files

| path | content |
|---|---|
| `coverage/intervals.csv` | **The coverage study, record by record** (36,000 rows): for each experiment (A, B), record length, simulated record (1 to 1,000), return period (100, 1,000 years) and method (`delta`, `profile`, `bootstrap`): the number of threshold exceedances, the point estimate, the nominal 95% interval, the true return level, and whether the interval covered the truth or lay wholly below it (`truth_above`) or above it (`truth_below`). Records whose fit failed (fewer than eight exceedances) are kept and marked `fit failed` |
| `coverage/summary.csv` | the 36 cells of the study: coverage with its Monte Carlo standard error, miss rates on each side, median upper/lower ratio, share of unbounded intervals, median point estimate over truth, failed records |
| `coverage/designs.json` | the two designs, their record lengths, threshold rules and true return levels |
| `loss/annual_losses_200k.csv` | 200,000 simulated years of the banking-sector loss process: incidents per year and the annual aggregate loss (USD) |
| `loss/truth_return_levels.csv` | the true annual return-level curve from 2,000,000 simulated years, return periods 1.25 to 100,000 years |
| `loss/record_1000y_incidents.csv`, `loss/record_100y_incidents.csv` | the two single records analysed in Section 5.3 of the paper: every incident loss with its year |
| `loss/parameters.json`, `loss/risk_metrics.json`, `loss/single_record_fits.json` | the loss-process parameters (planted tail index 0.625), VaR and expected shortfall of the 200,000-year sample, and the peaks-over-threshold fits to the two records |
| `attacker/venues.csv` | the four attack venues: breach probability, value at risk, recovery probability (one minus finality), laundering haircut, attribution probability, sanction, cost, retained value, expected payoff and break-even recovery |
| `attacker/log_gap_bank_to_bridge.csv` | the exact decomposition of the log payoff gap between a bank core and a cross-chain bridge (finality: 67%) |
| `attacker/recovery_curves.csv` | each venue's expected payoff as its recovery probability varies from 0 to 0.95 |
| `contagion/banks.csv` | the 28-bank core-periphery network (seed 7): tier, total assets, equity, interbank intensity, shared-vendor client flag and contagion multiplier when the bank alone loses 20% of equity |
| `contagion/exposures.csv` | the 92 interbank exposures: `weight` is the share of the creditor's interbank assets lent to the debtor (rows sum to one) |
| `contagion/phase_sweep.csv` | the contagion multiplier against the exposure scale for a hit on the largest bank and on a shared vendor; the critical scale is 0.868 |
| `contagion/cascades.csv` | the three scenarios of the paper round by round: every bank's distress (0 to 1) after each propagation step |
| `raw/` | the study's own outputs: `results.json`, `results.md` and the per-record interval cache in `raw/cache/` |
| `SHA256SUMS` | checksums of every file above |

## How the data were generated

`scripts/export_hf.py` in the code repository regenerates every file from master seed 20261108 exactly as
`run_study.py` draws it, and refuses to write unless each table reproduces the stored results to nine
significant digits.

- **Loss process.** Incidents arrive as a Poisson process (4 a year). Severities are log-normal (median $30M,
  sigma 1.4) with probability 0.95 and Pareto (scale $200M, alpha 1.6, so xi = 0.625) with probability 0.05; an
  incident above $1B triggers a cascade with probability 0.3, multiplying its loss by one plus an exponential
  with mean 2.
- **Experiment A** draws exceedances exactly from a generalized Pareto distribution (threshold $400M, sigma $250M,
  xi 0.625, 0.5 exceedances a year) for 30, 100 and 300-year records: the model is exactly right.
- **Experiment B** simulates the full loss process for 30, 100 and 1,000-year records and fits a GPD above the
  85th, 95th and 98th percentiles of incident losses; return levels are corrected from incident level to annual
  aggregate.
- **Intervals**: delta method on the log scale, profile likelihood, and parametric bootstrap with B = 199.
- **Network**: a 5-core, 23-periphery network from the same random number generator as the web instrument;
  linear DebtRank with a 20% equity shock.

## Licence

CC BY 4.0 (`LICENSE`). The code that generates the data is MIT licensed.

## Citation

```bibtex
@misc{chongmankhong2026false,
  author = {Chongmankhong, Thanatchon},
  title  = {False Confidence in the Tail: Finality, Extreme-Value Uncertainty,
            and Contagion in Banking-Sector Cyber Risk},
  year   = {2026},
  doi    = {10.5281/zenodo.24050729},
  url    = {https://thanatchon36.github.io/finality-research/paper/},
  note   = {Preprint; all parameters are illustrative}
}
```
