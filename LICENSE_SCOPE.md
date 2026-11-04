# License scope and data attribution

## Research reproduction code — MIT

The seven research reproduction scripts listed below are licensed under the MIT License in [LICENSE_CODE.txt](LICENSE_CODE.txt). Copyright (c) 2026 ConformalRisk Researcher.

- `scripts/auc_decomposition.py`
- `scripts/common.py`
- `scripts/fixed_budget_screening.py`
- `scripts/frontier_methods.py`
- `scripts/historical_statistics.py`
- `scripts/reproduce_results.py`
- `scripts/sensitivity_results.py`

## Research data, methods and documentation — CC BY 4.0

**Dataset:** ConformalRisk Research Dataset  
**Attribution:** ConformalRisk Researcher  
**License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)).

The following materials are licensed under CC BY 4.0 only to the extent of copyright and similar rights that ConformalRisk Researcher owns or is authorized to license, including any applicable research compilation rights:

- `DATA_DICTIONARY.md`
- `LICENSE_SCOPE.md`
- `README.md`
- `THIRD_PARTY_NOTICES.md`
- `data/calibration_candidates.csv`
- `data/episodes.csv`
- `data/exploratory_decisions.csv`
- `data/frozen_config.json`
- `data/policy_decisions.csv`
- `data/sensitivity_config.json`
- `data/tau_features.csv`
- `methods/scenario_protocol.json`
- `methods/system_prompt.txt`
- `methods/user_template.txt`

Recommended citation:

> ConformalRisk Researcher (2026). ConformalRisk Research Dataset. Dataset and reproducibility materials.

When sharing or adapting licensed material, provide appropriate attribution, a link to CC BY 4.0, and an indication of changes, retaining notices required by that license. A genuine repository URL or DOI may be added to the citation after publication; no such identifier is claimed here.

## Mixed experimental records and exclusions

`data/episodes.csv` contains recorded structured DeepSeek-generated proposals, validated normalization and explicitly labeled research fallback values, alongside researcher-designed simulated inputs, scores and outcomes. `data/tau_features.csv` contains a synthetic episode key and eight numeric features, including researcher calculations conditioned on the recorded actions. These are saved experimental records, not regenerated observations. Estimated slippage and target risk signals originate in the simulated environment and are selected or zeroed by the recorded action.

The CC BY 4.0 scope includes the licensor's research rights and available user output rights, if any, in these records. The official [DeepSeek Open Platform terms effective 2026-04-29, section 4.2](https://cdn.deepseek.com/policies/zh-CN/deepseek-open-platform-terms-of-service.html) and [general terms effective 2025-09-05, section 4.2](https://cdn.deepseek.com/policies/zh-CN/deepseek-terms-of-use.html) are the basis for user output rights. This does not assert that every cell is exclusively copyrighted by the licensor or that all proposals were independently created by a person.

No exclusive copyright is asserted in raw facts, purely calculated numbers, enums or factual model/service identifiers. `provider_request_model`, `provider_response_model` and `system_fingerprint` remain descriptive research provenance. Retaining these facts does not license any independent provider rights.

Third-party software, DeepSeek models and service implementations, proprietary weights, trademarks, patents and other independent third-party rights are excluded from the project license grants. Their applicable terms remain in force. The original DeepSeek model and historical service environment are not included. The MIT standard text in LICENSE_CODE.txt is a license instrument, not a claim that its wording or third-party software is original research code. `requirements.txt` and `SHA256SUMS.txt` are dependency/checksum facts and do not relicense dependencies or create exclusive rights in facts. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
