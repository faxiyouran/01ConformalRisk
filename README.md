# ConformalRisk: frozen simulation records and offline reproduction

This repository provides research data and reproducibility materials for the ConformalRisk study of risk-calibrated authorization for AI-proposed decentralized-finance actions. The observations are **simulated decision opportunities**, not real customers, market transactions, or a field trial. Historical transaction proposals were generated through the DeepSeek API; the structured proposal fields, model identifiers, fallback flags, original prompts, and frozen scores preserve that provenance.

The repository contains saved experimental records for 4,000 episodes: TRAIN 1,200, CALIBRATION 400, TEST 400, and INDEPENDENT_CONFIRMATION 2,000. It retains all five fallback cases, including one technical failure. No replacement observations were generated.

## Files

- `data/episodes.csv`: one row per opportunity, recorded structured proposal, task/action, score, loss/accounting variables, and historical model provenance.
- `data/policy_decisions.csv`: 16,000 G0–G3 records. TEST/confirmation comprise 9,600 frozen stored evaluation records; TRAIN/CAL comprise 6,400 derived frozen-rule records, explicitly identified by `decision_provenance`. They are not additional evaluation trials.
- `data/exploratory_decisions.csv`: 11,200 recorded CAL/TEST/confirmation decisions for two G4/G5 configurations each.
- `data/calibration_candidates.csv`, `data/frozen_config.json`: archived selection evidence and fixed thresholds/configurations.
- `data/tau_features.csv`, `data/sensitivity_config.json`: minimal archived numeric feature projection, frozen TRAIN scaling, historical fitted coefficients, exact tau cutoffs, and sampling/mixture specifications. These are research inputs, not proprietary external-LLM weights.
- `methods/`: original experimental prompts and the synthetic scenario specification.
- `scripts/`: offline entry point and retained historical AUC, blocking, and frontier algorithms.
- `DATA_DICTIONARY.md`: field definitions and denominator conventions.
- `SHA256SUMS.txt`: checksums of every other public file.

## Run

Tested with Python 3.12.14, NumPy 2.2.6, pandas 2.3.3 and SciPy 1.18.1. In an environment with those packages, no installation or network connection is needed. Otherwise install the three dependencies with `python -m pip install -r requirements.txt`.

From the repository root:

```bash
python -B scripts/reproduce_results.py
```

Outputs go to a new or empty `results/` directory. Use `--output-dir another-results` for another run. The script checks input hashes before and after computation. It does not call an LLM, fit a model, tune new policies, or modify research inputs. The retained AUC/blocking diagnostics use 2,000 bootstrap draws; primary paired intervals use their original 10,000 draws. SciPy is used only for the archived logistic sigmoid evaluation.

## Reproduced results and interpretation

| Cohort | Policy | Authorized / N (ARR) | Harmful authorized / N (HERISK) |
|---|---|---|---|
| TEST | G1 | 198/400 (49.50%) | 2/400 (0.50%) |
| TEST | G3 | 366/400 (91.50%) | 22/400 (5.50%) |
| Confirmation | G1 | 1,058/2,000 (52.90%) | 8/2,000 (0.40%) |
| Confirmation | G3 | 1,867/2,000 (93.35%) | 92/2,000 (4.60%) |

The entry also recomputes G0/G2, exact financial loss and harm labels, pooled/conditional/task AUC, action support, fixed 10/20/30% blocking diagnostics, G4/G5 replay, and retrospective frontier matching. Confirmation pooled/within-task/between-task AUC rounds to 0.9044/0.7404/0.9525; between-task pair share is 77.30%.

G0 authorizes all opportunities. G1 uses pre-action **actual** exposure <=0.10. G2/G3 use exact stored binary64 scores strictly below their frozen cutoffs; equality is denied. An authorized HOLD is a no-op. Permission, a proposed state change, and successful execution are separate concepts.

G4/G5, conditional/task diagnostics, AUC decomposition, fixed blocking, and frontier comparisons are **retrospective diagnostics**. G4/G5 families were designed after original outcomes were inspected; selection from the old CAL table does not make them prospective baselines. Frontier lower-HERISK counts are 33/37 TEST and 136/138 confirmation nearest-tie pairs, drawn from 36/137 risk Pareto points. Every selected pair has zero observed ARR difference. The duplicate all-authorize match adds one pair per cohort; pairs are not independent experiments. Ties and unfavorable results are retained.

## Additional reported analyses

The same command writes the following under `results/sensitivity/`. The implementation is a later offline reproduction interface for archived studies, not newly conducted original experiments.

| Analysis | Historical scope and reproducible output |
|---|---|
| Primary confidence intervals | Frozen TEST/confirmation decisions; original 10,000 paired draws, separate seeds and payment denominators; G3 minus G0/G1/G2. |
| Task AUC intervals | Retrospective confirmation task-by-label bootstrap, 2,000 draws; original continuing RNG stream across approval then rebalance. |
| CAL sample size | Post-primary sensitivity; 500 fixed no-replacement subsamples each at n=100/200/300, plus one n=400 full-CAL reference. |
| Risk budget | Archived alpha .025/.05/.075/.10 sensitivity, using the same CAL and evaluation scores; primary thresholds unchanged. |
| Composition | Post-primary category-only in-support reweighting of 2,400 TEST+confirmation episodes; no new shifted-population trial. |
| Harm threshold tau | Archived post-primary sensitivity at .025/.05/.10; fitted coefficients and preprocessing replayed without fitting. |

The primary G3-G2 harmful-authorization intervals are [-1.25,0.00] percentage points in TEST and [-0.40,-0.05] in confirmation. Task AUC intervals round to [0.5492,0.6706] for approval and [0.9394,0.9742] for rebalance. Median CAL-size cutoff gaps round to .0225/.0168/.0168/.0138. Severe joint reweighting gives G3 ARR 76.74% and HERISK 7.52%. At tau=.10 the archived sensitivity rules both authorize all proposals (TEST HERISK .0525; confirmation .0475). Sampling uncertainty, sensitivity, and retrospective comparisons retain their original distinct interpretations and supply no new prospective or external-validity guarantee.

## Relation to the paper and limitations

This package supports the ConformalRisk main manuscript and Electronic Companion: the primary authorization table, task/action and HOLD analyses, AUC decomposition and fixed-capacity tables, the archived G4/G5/frontier descriptions, primary and task confidence intervals, and the reported risk-budget, CAL-size, composition and tau sensitivities. Manuscript and companion files are not included. Cite the ConformalRisk manuscript using its actual bibliographic details when available.

Reproduction starts from recorded proposals, scores and outcomes. Historical proprietary weights and service state are unavailable; **exact regeneration of original online DeepSeek responses is not supported**. The original model-fitting stages are not rerun. The harm-threshold analysis replays archived fitted coefficients with the frozen TRAIN preprocessing and original exact cutoffs; it checks all six calibration boundaries and never replaces the frozen primary scores. Its historical tau=0.05 refit is distinct from the original primary model. No new fit or online generation is performed.

## License

Research reproduction code in `scripts/` is licensed under [MIT](LICENSE_CODE.txt), Copyright (c) 2026 ConformalRisk Researcher. Research data, methods and applicable documentation are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as specified in [LICENSE_SCOPE.md](LICENSE_SCOPE.md), only for rights the licensor owns or is authorized to license. Third-party components retain their own licenses and rights; the original DeepSeek model, proprietary weights and historical service environment are not included. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Recommended dataset citation:

> ConformalRisk Researcher (2026). ConformalRisk Research Dataset. Dataset and reproducibility materials.
