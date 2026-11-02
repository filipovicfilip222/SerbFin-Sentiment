# Annotation Guidelines — SerbFin-Sentiment

## Core principle
Label from the **investor / market perspective**, not from an emotional or social perspective.

| Label | Use when |
|-------|----------|
| `positive` | Bullish signal — good for company/sector/economy: profit up, beat expectations, new deal, growth, upgrade, record revenue |
| `negative` | Bearish signal — bad for company/sector/economy: loss, miss, layoffs, downgrade, default, legal penalty, credit cut |
| `neutral` | No clear directional signal: appointment, schedule, mixed result, purely descriptive, regulatory announcement without clear outcome |

## Rules

1. **Investor lens.** "Company cuts 5,000 jobs to boost margins" → `positive` (cost-cutting story).
   "Inflation rose 0.1%" without further context → `neutral`.

2. **Mixed sentiment → dominant signal.** If both positive and negative elements are present,
   pick the dominant one. If truly balanced → `neutral`.

3. **Pure facts / schedules / appointments** → `neutral` unless the event itself has clear
   market impact (e.g., a scheduled NBS rate decision with no content is `neutral`; an NBS
   announcement of an unexpected rate hike is `negative` or `positive` depending on direction).

4. **Hedged / speculative language** ("could", "may", "is expected to") → weigh the realized
   framing if available; otherwise default to `neutral` if no clear direction.

5. **Macro vs company.** Apply the same logic to macroeconomic headlines:
   "GDP grows 3.5%" → `positive`; "Unemployment rises to 11%" → `negative`.

## Common traps

- Layoffs / restructuring → often `positive` (cost story), not `negative`.
- Strike / lawsuit announcement without outcome → `neutral` or `negative` depending on severity.
- "Company signs MoU" with no financial details → `neutral`.
- Record losses → `negative` even if the company says "we are working on it".

## Label source values

- `human` — hand-labeled by a human annotator (required for the test set).
- `llm-silver` — Claude Haiku auto-label (train pool only).
- `llm-reviewed` — Claude Haiku label spot-checked and confirmed/corrected by a human.

## Inter-annotator agreement

Target Cohen's κ ≥ 0.6 on the test set. Document disagreements and their resolutions.
