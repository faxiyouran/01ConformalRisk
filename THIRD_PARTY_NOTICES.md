# Third-party software and experimental sources

The package imports independently installed software. It does not bundle third-party source trees, wheels, DLLs, external-LLM weights, or vendor service code. The project licenses do not relicense these components.

| Runtime component | Tested version | License information |
|---|---|---|
| NumPy | 2.2.6 | BSD 3-Clause project license; binary distributions carry additional component notices. |
| pandas | 2.3.3 | BSD 3-Clause. |
| SciPy | 1.18.1 | BSD 3-Clause project license; binary distributions carry additional component notices. |
| python-dateutil | 3.0.0 | Distribution provides dual-license terms; retain its installed notices. |
| pytz | 2026.6 | MIT. |
| tzdata | 2026.6 | Distribution includes software and time-zone-data notices. |
| six | 1.17.0 | MIT. |

Only the first three are direct requirements; the others are installed transitive components. Their license files remain with the installed distributions. The software uses standard-library APIs supplied with Python, which is installed separately under its own terms.

DeepSeek is the source of the historical AI-generated action proposals identified in the data and methods. It is not represented as an author of the reproduction implementation or as an endorser. The mixed experimental records are covered by CC BY 4.0 only to the extent of user output rights and research rights available to the licensor; see LICENSE_SCOPE.md and the official API terms referenced there. This does not relicense the DeepSeek service or model. Model/service names are descriptive references, not a grant of trademark or model rights.
