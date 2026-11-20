# C_1a (first autocorrelation) upper bound: 1.502664829454529727 (262,144-cell step function)

An exact certificate that the first autocorrelation constant (constant 1a in
[teorth/optimizationproblems](https://github.com/teorth/optimizationproblems)) satisfies

    C_1a <= 735013052674165058348112707747577856/489139719162107651610797699158672969 = 1.502664829454529727...

| Field | Result |
|---|---|
| Mathematical objective | Minimize max_t (f*f)(t) / (∫f)^2 over nonnegative step functions f on [-1/4, 1/4] with 262,144 equal cells; any f certifies an upper bound on C_1a. |
| Baseline | intervalplace/frontier, branch c1a-1.5026648295 (commit 546831c, 2026-11-16, not yet posted to the registry): exact 1.502664829454795070. Registry best: PR #367 (Esc2026), 1.5026648332. |
| Candidate result | Exact 1.502664829454529727: 2.654e-13 below frontier's value, about 3.7e-9 below the registry's best. |
| Verification | Exact big-integer arithmetic, two independent checkers: frontier's own verify.py with --beat (below by 2.654304e-13) and registry PR #374's verify_standalone.py (Kronecker substitution, --exact: ACCEPTED; for nonnegative heights max\|f*f\| = max f*f). |
| Record eligibility | Registry: eligible (no minimum improvement; exact certificate). The EinsteinArena first-autocorrelation problem needs an improvement of at least 1e-8 over its own leader. |
| Novelty | Improved construction, and a trivially small one: a single float-level polish win on frontier's construction (1 of 18 samples). Same construction, no new technique. |
| Search economics | NP-beta1, 2 workers x 9 samples (polishes overran their 25 s budget at 262,144 cells), 9.9 min on 2 cores; 1 win in 18 (6%); 2.7e-14 per minute. |
| Decision | Abandon NP search on 1a: the record is tight (115,400 lags within 1e-9 of the max), so it fails the NP-beta2 tightness gate. |

## Folders
- `certificate/`: heights (frontier's text format, max height 2^50), the same heights as `_values.json` for PR #374's verifier, both checker outputs, and frontier's verifier output on frontier's own certificate (the bar).
- `runs/`: per-sample logs of both workers and the 5-target result cards.
- `code/npg.py`: the generic NP-beta1 worker used for this search.

## Sources and credit
- Starting construction and first checker: [intervalplace/frontier](https://github.com/intervalplace/frontier), branch c1a-1.5026648295.
- Second checker: PR #374's verify_standalone.py, [cweill/c1d-upper-bound](https://github.com/cweill/c1d-upper-bound).
