"""Run the detector on one simulated test run and have the copilot explain each alert.

    python -m copilot.cli --provider anthropic --seed 3001
    python -m copilot.cli --provider scripted  --seed 3001      # offline
    python -m copilot.cli --provider anthropic --seed 3001 --inject   # add a malicious note

Writes a readable report to outputs/copilot_report_<seed>.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from weathering_ad.detectors import ResidualZ, to_alerts
from weathering_ad.simulator import FAULT_TYPES, simulate_run

from .agent import diagnose
from .kb import KnowledgeBase
from .llm import make_llm


def render(alert, r, df) -> str:
    lines = [f"## Alert: `{alert['channel']}` at {df.timestamp[alert['start']]}",
             f"**Status:** {r.status}  |  steps: {r.steps}  |  tools: "
             f"{', '.join(c['tool'] for c in r.tool_calls)}  |  {r.latency_s:.1f}s  |  "
             f"tokens: {r.input_tokens + r.output_tokens}"]
    if r.injection_seen:
        lines.append("> ⚠️ Suspicious instruction-like text was found in retrieved content and redacted.")
    if r.error:
        lines.append(f"**Error:** {r.error}")
    d = r.diagnosis
    if d:
        lines += [f"**Likely cause:** {d['likely_cause']} ({d['confidence']} confidence)", "",
                  d["summary"], "", "**Evidence**", *[f"- {e}" for e in d["evidence"]], "",
                  "**Recommended actions**", *[f"- {a}" for a in d["recommended_actions"]], "",
                  f"**Test validity:** {d['test_validity_impact']}",
                  f"**Sources:** {', '.join(d['citations'])}"]
    if r.violations:
        lines += ["", "**Guardrail violations (not shown to operator as trusted):**",
                  *[f"- {v}" for v in r.violations]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted", choices=["anthropic", "openai", "scripted"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=3001)
    ap.add_argument("--inject", action="store_true", help="attach a malicious operator note")
    ap.add_argument("--max-alerts", type=int, default=6)
    args = ap.parse_args()

    detector = ResidualZ().fit([simulate_run(s)[0] for s in range(10)])
    df, truth = simulate_run(args.seed, FAULT_TYPES)
    kb = KnowledgeBase()
    llm = make_llm(args.provider, args.model)
    notes = [kb.by_id["FN-02"].text] if args.inject else None

    parts = [f"# Copilot report: simulated run {args.seed} ({args.provider})",
             "_Synthetic data; fictional X-100 manual. Advisory output only._", "",
             "Injected faults (ground truth): "
             + ", ".join(f"{f.type} @ {df.timestamp[f.start]}" for f in truth), ""]
    for a in to_alerts(detector.score(df))[: args.max_alerts]:
        r = diagnose(llm, df, detector, a, kb=kb, operator_notes=notes)
        block = render(a, r, df)
        print(block, "\n")
        parts += [block, ""]
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    path = out / f"copilot_report_{args.seed}.md"
    path.write_text("\n".join(parts))
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
