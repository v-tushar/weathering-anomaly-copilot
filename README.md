# Weathering chamber anomaly detection + LLM diagnostic copilot (proof of concept)

Two layers, built as a self-contained proof of concept on **synthetic data**. No
Atlas or customer data is used.

1. **Detection:** early-warning anomaly detection on chamber telemetry (statistical ML).
2. **Copilot:** an LLM agent that investigates each alert with tools, retrieves the
   relevant troubleshooting sections, and explains the likely cause and next steps.
   Guardrails check every answer before an operator sees it.

```
sensors -> detector -> alert -> copilot agent --tools--> telemetry summaries
                                     |        \--RAG---> troubleshooting guide
                                     v
                        submit_diagnosis -> guardrails -> operator
                                   (fix once, else "needs human review")
```

## Result (held-out test set, 50 faulty and 20 clean 30-day runs)

| Fault | Static setpoint alarm (today's baseline) | Residual z + CUSUM (shipped) |
|---|---|---|
| Lamp degradation | 100 % caught, **median 221 h (~9 days)** after onset | 100 % caught, **median 3.2 h** |
| Seal / humidity leak | 100 %, median 4.0 h | 100 %, median 2.4 h |
| Control-loop fault | 100 %, median 0.3 h | 100 %, median 0.2 h |
| Sensor spike / dropout | **74 %** | 100 % |
| False alarms, healthy runs | 0 in 600 instrument-days | 0 in 600 instrument-days (95 % upper bound ≈ 1 per 200 days) |
| False alarms, sensors 50 % noisier than calibration | 0 | 0.11 / day. **Recalibrate per instrument.** |

The Isolation Forest variant was tested and **not shipped**. It caught only 8 %
of seal leaks, and under the noisier-sensor shift it raised about 3 false
alarms per day. Full tables: `outputs/results.md`.

## Why lamp degradation is the interesting case

Xenon instruments typically hold irradiance at setpoint with closed-loop lamp
control. As a lamp degrades, the controller drives it harder and the
irradiance reading stays perfect, so a setpoint alarm sees nothing until the
lamp hits 100 % drive and irradiance finally falls. The detector watches
**lamp drive relative to this lamp's own baseline and normal aging rate**.
That catches abnormal aging within hours instead of days, before any test
specimen gets an out-of-spec exposure.

## Detection approach

1. **Phase-aware residuals.** Each reading is compared with what a healthy
   chamber shows at the same point in the light/dark cycle (a profile learned
   from healthy calibration runs), giving a z-score per channel.
2. **Point anomalies.** |z| > 5, debounced to 2 of 3 samples.
3. **Drift.** One-sided CUSUM on lamp drive (up) and irradiance (down), light
   phase only, reset after each signal. The threshold is set from the largest
   excursion seen on healthy runs, times 1.5.
4. **Alerts** are per channel, so an open alert on one sensor can't hide a new
   fault on another. Each alert names the channel for root cause.

## Copilot (LLM layer)

| Piece | What it does | File |
|---|---|---|
| Tools (function calling) | `get_alert_context`, `get_channel_trend`, `search_manual`, `submit_diagnosis` (structured output) | `copilot/tools.py` |
| RAG | TF-IDF retrieval over a **fictional** "X-100" troubleshooting guide + field notes. Swappable for embeddings. | `copilot/kb.py`, `copilot/knowledge/` |
| Agent loop | Provider-agnostic: Anthropic, OpenAI, or an offline scripted stand-in. Max 8 steps. | `copilot/agent.py`, `copilot/llm.py` |
| Guardrails | Input: redact prompt-injection text from retrieved content. Output: fixed cause taxonomy; citations must exist and have been retrieved; every number must match a tool result; cause must agree with the sensors. One retry, then `needs_human_review`. | `copilot/guardrails.py` |
| Baseline | Rule-based diagnoser the LLM is compared against | `copilot/baseline.py` |
| Eval | 45 scenarios: real alerts, fabricated false alarms, prompt-injection (filter on/off). Accuracy vs. baseline, guardrail pass rate, review rate, latency, tokens. | `copilot/eval.py` |

**Finding so far:** on these cleanly separated simulated faults, the rule-based
baseline classifies the cause 100 % correctly. So the LLM's value is **not**
classification. It is grounded explanations, next steps pulled from the manual,
and handling messy cases the rules don't cover. In production the rule output
could be passed to the LLM as a hint.

**Live-LLM eval** (Claude Haiku 4.5, all 45 scenarios, $1.37, median 19 s per alert;
`outputs/copilot_eval_anthropic_claude-haiku-4-5.*`):

| Metric | Result |
|---|---|
| Cause accuracy | 44/45 (97.8 %); rule baseline 45/45 |
| Prompt injection resisted (filter on / off) | 5/5 and 5/5 |
| False alarms called "no fault" | 5/5 |
| Passed guardrails first try / after one retry | 58 % / 29 % |
| Needs human review | 13 % (all for unverifiable numbers) |

The one miss passed the guardrails. On a 328 h alert the tool's "72 h lamp trend" covers the
end of the alert, where drive has flattened, so it read -0.027 %/day and the model reasonably
concluded "sensor fault". Lessons: guardrails do not replace an eval, and a misleading tool
summary misleads the model. Planned fix (from manual section IRR-01, to be confirmed on fresh
seeds): trend from the alert start, and reject a sensor-fault diagnosis while lamp drive is
moving.

The copilot is advisory: its tools are read-only and it cannot operate the chamber.

## Browser demo

```bash
python -m demo          # http://127.0.0.1:8000
```

One page that runs the whole pipeline live, for a screen-shared walkthrough:

1. **Simulate** a 30-day run with any combination of faults, any seed, and a sensor-noise
   slider. Telemetry is charted with the injected faults shaded, and an alert lane shows
   where each of the three detectors fires.
2. **Compare** time-to-detect per fault, side by side. This is where the lamp-degradation
   point lands: the static setpoint alarm takes ~180 h, the shipped detector ~3 h.
3. **Investigate** any alert with the copilot. Tool calls stream in as the agent makes
   them, the guardrail verdict is shown, and the diagnosis is displayed next to the
   ground truth and the rule-based baseline — so the "the rules already get the cause
   right" finding is visible rather than asserted.

4. **Report** for the chamber engineer (see below): what went wrong, whether the running
   test is affected, what to do, printable.
5. **Monitor**: guardrail pass rate, human-review rate, latency, tokens, engineer-confirmed
   accuracy, and the audit trail with its integrity check.

Provider is switchable in the page (Anthropic / OpenAI / offline scripted stand-in);
providers with no API key set are disabled with a note saying which variable to set.
A **prompt injection** toggle attaches the malicious operator note, and an **input filter**
toggle turns the redaction defense off, so the model's own robustness can be shown
separately from the filter's. **Evidence up to** chooses what the copilot sees: the first
12 h after the alert fires (what an engineer paged in real time has) or the whole alert
(hindsight).

## Incident report for the chamber engineer

The raw diagnosis is for engineers building the system. The person at the chamber gets a
report (`copilot/report.py`) with two kinds of content, labelled on the page:

| Measured (code, no AI) | AI (shown with its guardrail status) |
|---|---|
| Severity: act now / plan maintenance / investigate / monitor / no action | Plain-English explanation |
| Running-test impact: each controlled condition vs. setpoint ± tolerance, hours out | Recommended steps, chosen from the manual |
| Lamp headroom and **days until drive hits 100 %** | Test-validity comment |
| Chart of the affected channels with the tolerance band, cut at the evidence time | |

Manual steps and safety notes ("qualified technician… locked out") are quoted verbatim
from the cited sections. If the AI answer fails verification, the headline does not
repeat it; the report says an engineer review is needed and the measured half still stands.

The headroom forecast fits the lamp-drive rate on readings since the alert fired. A
72 h window that reaches back before the fault mixes healthy and faulty aging; in
development it forecast 66 days when the lamp actually maxed out in 7. The shipped
estimate is within about a day of the simulated truth from 6 h after the alert (tested).

## Monitoring, audit trail, feedback

`copilot/audit.py`, standard library only:

- **Every diagnosis is logged** with provider, model, prompt version (content hash),
  manual version (content hash), detector version, every tool call, guardrail result,
  tokens and latency. A one-line JSON event also goes to stdout for a log platform.
- **Tamper-evident:** records are append-only and hash-chained. Editing or deleting a past
  record breaks verification (tested), and the page shows the chain status.
- **Engineer feedback** ("was this right? if not, what was it?") is stored as its own audit
  event. `GET /api/audit/export` returns the rated cases as JSONL, ready to add to the eval set.
- The monitoring panel shows pass rates, human-review rate, latency p50/p90, tokens per
  diagnosis, injections caught, engineer-confirmed accuracy and agreement with the rules.

The demo uses SQLite. On the free hosted demo the disk is ephemeral, so the log resets when
the server restarts. Production would use the same schema in a managed database with
write-once storage and role-based access.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs lint (ruff) and the full offline test suite
on every push and pull request. Render deploys only when those checks pass
(`autoDeployTrigger: checksPass`), so a red build never reaches the live demo.

## Methodology guarantees (each enforced by a test)

- Fit on healthy calibration runs only. Test seeds never touch fitting or design choices.
- Development decisions were made on a separate validation set. An earlier
  test set exposed a bug, so it was retired and final numbers come from fresh seeds.
- Every feature is **causal** (t uses data ≤ t), so the same code runs on a live stream.
- Evaluation is **event-level**: detection rate, time to detect, false alarms per day.
  Point-level F1 is dominated by long drift windows and was misleading in v1.
- The test suite was mutation-checked: 11 deliberate bugs each make at least one
  test fail. Detector: leaky features, latching CUSUM, no warm-up handling, no
  aging model, leaky drift smoothing. Copilot: grounding check off, citation
  check off, no retry, injection filter off, telemetry check off, provider
  errors not contained.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env                # optional: add an API key for the live copilot
python -m pytest -q                 # 96 tests, ~15 s, fully offline
python -m demo                      # >>> browser demo: the whole pipeline in one page <<<
python run_demo.py                  # full detector evaluation, ~3 min -> outputs/

# copilot, offline plumbing check (no key needed; NOT an LLM)
python -m copilot.cli  --provider scripted --seed 3001
# copilot, live (set ANTHROPIC_API_KEY or OPENAI_API_KEY first; COPILOT_MODEL to override the model)
python -m copilot.cli  --provider anthropic --seed 3001            # report -> outputs/copilot_report_3001.md
python -m copilot.cli  --provider anthropic --seed 3001 --inject   # with a malicious operator note
python -m copilot.eval --provider anthropic --limit 20             # eval (cap scenarios to control cost)

MODEL_PATH=outputs/residual_z.joblib uvicorn weathering_ad.api:app   # detector scoring API
```

## Layout

```
weathering_ad/simulator.py   synthetic chamber + fault injection with ground truth
weathering_ad/detectors.py   static baseline, residual z + CUSUM, Isolation Forest variant, alerting
weathering_ad/evaluate.py    event-level metrics
weathering_ad/api.py         FastAPI scoring service with input validation
copilot/                     LLM copilot: tools, RAG, agent loop, guardrails, eval, CLI,
                             engineer report (report.py), audit trail + monitoring (audit.py)
demo/                        browser demo UI (FastAPI + one static page, no JS deps)
run_demo.py                  calibrate -> evaluate -> chart -> save model + metadata
tests/                       simulator, detectors, evaluation, API, copilot, report, audit, demo server
.github/workflows/ci.yml     lint + tests on every push; Render deploys only when green
v1_original/                 first prototype, kept for comparison (has known bugs)
```

## Known limitations

- **Synthetic data.** Noise is Gaussian and stationary. Real chambers have nuisance
  events (door openings, specimen loading, sensor calibration, program
  changes) that would generate false alarms until they are labeled or masked.
- Thresholds are calibrated on one simulated instrument type. Real use needs
  per-model or per-unit calibration and periodic recalibration; the noisy-sensor
  test shows why.
- The copilot has been tested end to end offline (scripted model, fake provider
  clients, guardrail mutation checks) but not yet against a live LLM in this
  environment. Run the eval with a key before quoting copilot accuracy.
- The knowledge base is fictional and small (11 sections). Real use needs real
  manuals, service history and a retrieval eval (did the right section come back?).
- The API is stateless and rescores the run from its start on each call. A
  production version keeps per-instrument CUSUM state and scores incrementally.

## Next steps toward production

Real telemetry export (e.g. from the instrument's data-acquisition software) →
per-instrument baselines → shadow mode alongside existing alarms → alert
review loop to label nuisance events → versioned models with an audit trail
of the model, threshold and data behind each alert.
