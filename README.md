# RL for Dynamic Paging (DCMA-Page): simulator, code and data

Code and data for the manuscript *Delay-Constrained, Mobility-Aware Reinforcement Learning for Dynamic Paging in 4G/5G/Beyond-5G Networks: A Simulation Study with Held-Out Seeds* (K. Popat, D. Meva).

**All data are simulated.** No real mobility traces or operator data are used.

## Files
| File | Purpose |
|---|---|
| `paging_sim.py` | Simulator (hex grid; Gauss–Markov, random-waypoint, Markov-chain, routine mobility), baselines, DCMA-Page |
| `dev_tune.py` | Hyper-parameter tuning on **development seeds 100–105** → `dev_results.csv` |
| `run_all2.py` | Final experiments F1–F8 on **held-out test seeds 200–219** → `res2_*.csv`, `curves2_*.csv`, `wall2_*.txt` |
| `analyze2.py` | Tables, statistics (Wilcoxon + Holm), figures → `results.csv`, `RESULTS_SUMMARY_auto.md`, `table_sig.csv`, `src/tables/*.md`, `figs/` |
| `figs/` | The eight figures of the paper |
| `REFERENCES_VERIFIED.csv` | The 50 references with DOI, venue, open-access status and verification notes |

## Reproduce
```
pip install -r requirements.txt
python dev_tune.py           # optional: re-derive tuned constants (dev seeds)
python run_all2.py F1a F1b F3a F3b F4 F5 F6 F7 F8   # all test experiments (~35 min on 2 cores in total)
python analyze2.py           # tables, statistics, figures
```
Pass any subset of the experiment names (e.g. `python run_all2.py F1a`) to run only those. Runs are deterministic given the seeds. Wall-clock times in the paper were measured on 2 CPU cores without GPU; the neural baselines are implemented in NumPy.

## Seed discipline
Development seeds 100–105 were used only for tuning. All reported results use test seeds 200–219. An earlier preliminary study (seeds 0–19, earlier method version) is described in the paper (Section 6.9) and is not part of this repository.

## Licence
MIT (see `LICENSE`). Please cite the paper if you use this code.
