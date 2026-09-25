"""Copilot tests. All offline: scripted models stand in for the LLM.

They cover the parts that must hold whatever model is plugged in: tool
dispatch, the agent loop, guardrails (including a model that fabricates
citations and numbers), injection filtering, and provider message formatting.
"""

import types

import pytest

from copilot import guardrails
from copilot.agent import diagnose
from copilot.baseline import CAUSE_QUERY, rule_diagnosis, scripted_policy
from copilot.eval import build_scenarios, run_eval, summarize
from copilot.kb import KnowledgeBase
from copilot.llm import AnthropicLLM, OpenAILLM, ScriptedLLM, ToolCall, Turn
from copilot.tools import ToolSession
from weathering_ad.detectors import to_alerts
from weathering_ad.simulator import FAULT_TYPES, simulate_run


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


@pytest.fixture(scope="module")
def run700(fitted):
    df, truth = simulate_run(700, FAULT_TYPES)
    alerts = to_alerts(fitted["residual_z"].score(df))
    return df, truth, alerts


def lamp_alert(alerts):
    return next(a for a in alerts if a["channel"] == "lamp_power_pct")


# --- knowledge base --------------------------------------------------------------

def test_kb_ids_unique_and_every_cause_query_finds_its_section(kb):
    assert len(kb.by_id) == len(kb.chunks)
    expected_top = {"lamp_degradation": "LAMP", "irradiance_sensor_fault": "IRR",
                    "temperature_control_fault": "TEMP", "humidity_seal_leak": "HUM"}
    for cause, prefix in expected_top.items():
        top = kb.search(CAUSE_QUERY[cause], k=1)[0][0]
        assert top.id.startswith(prefix), (cause, top.id)


# --- injection filter ---------------------------------------------------------------

def test_injection_text_is_redacted(kb):
    text, found = guardrails.redact_injections(kb.by_id["FN-02"].text)
    assert found and "ignore all previous instructions" not in text.lower()
    assert "noisy" in text  # the harmless part of the note survives
    assert guardrails.redact_injections(kb.by_id["LAMP-01"].text) == (kb.by_id["LAMP-01"].text, False)


def test_tools_redact_injection_in_search_and_notes(fitted, run700, kb):
    df, _, alerts = run700
    s = ToolSession(df, fitted["residual_z"], lamp_alert(alerts), kb,
                    operator_notes=[kb.by_id["FN-02"].text])
    out = s.call("search_manual", {"query": "lamp drive irradiance readings noisy"})
    ctx = s.call("get_alert_context", {})
    assert "FN-02" in out and "ignore all previous" not in out.lower()
    assert "ignore all previous" not in ctx.lower() and "REDACTED" in ctx
    assert s.injection_seen


def test_tool_errors_are_returned_not_raised(fitted, run700, kb):
    df, _, alerts = run700
    s = ToolSession(df, fitted["residual_z"], alerts[0], kb)
    assert "unknown tool" in s.call("delete_everything", {})
    assert "bad arguments" in s.call("get_channel_trend", {"channel": "rh_pct"})
    assert "unknown channel" in s.call("get_channel_trend",
                                       {"channel": "x", "hours_before": 5, "hours_after": 1})


# --- agent loop + guardrails ----------------------------------------------------------

def test_scripted_copilot_diagnoses_each_alert(fitted, run700, kb):
    df, truth, alerts = run700
    llm = ScriptedLLM(scripted_policy)
    for a in alerts:
        r = diagnose(llm, df, fitted["residual_z"], a, kb=kb)
        assert r.status == "ok", r.violations
        assert r.diagnosis["citations"]
        assert [c["tool"] for c in r.tool_calls][:1] == ["get_alert_context"]


def _hallucinator(fix_on_retry: bool):
    """Cites a section that doesn't exist and invents a sensor value; optionally fixes it."""
    def policy(transcript):
        rejected = any(m["role"] == "tool" and m["content"].startswith("REJECTED") for m in transcript)
        turn = scripted_policy([m for m in transcript
                                if not (m["role"] == "tool" and m["name"] == "submit_diagnosis")])
        call = turn.tool_calls[0]
        if call.name == "submit_diagnosis" and not (rejected and fix_on_retry):
            bad = dict(call.args)
            bad["citations"] = call.args["citations"] + ["LAMP-99"]
            bad["evidence"] = call.args["evidence"] + ["lamp drive reached 87.31 % at 03:00"]
            return Turn(None, [ToolCall(call.id, "submit_diagnosis", bad)])
        return turn
    return policy


def test_guardrails_catch_fabrication_and_allow_one_fix(fitted, run700, kb):
    df, _, alerts = run700
    r = diagnose(ScriptedLLM(_hallucinator(fix_on_retry=True)), df, fitted["residual_z"],
                 lamp_alert(alerts), kb=kb)
    assert r.status == "ok_after_retry"


def test_persistent_fabrication_goes_to_human_review(fitted, run700, kb):
    df, _, alerts = run700
    r = diagnose(ScriptedLLM(_hallucinator(fix_on_retry=False)), df, fitted["residual_z"],
                 lamp_alert(alerts), kb=kb)
    assert r.status == "needs_human_review"
    joined = " ".join(r.violations)
    assert "LAMP-99" in joined and "does not exist" in joined
    assert "87.31" in joined  # invented number caught by grounding check


def test_timestamps_and_keys_do_not_ground_invented_integers(fitted, run700, kb):
    # Tool results carry timestamps ("2026-01-01 12:36") and keys ("prior_24h").
    # Their digits must not make a made-up "17 % in 12 hours" look grounded.
    df, _, alerts = run700
    s = ToolSession(df, fitted["residual_z"], lamp_alert(alerts), kb)
    s.call("get_alert_context", {})
    s.call("get_channel_trend", {"channel": "rh_pct", "hours_before": 24, "hours_after": 6})
    s.call("search_manual", {"query": "lamp drive rising faster than normal aging"})
    d = {"likely_cause": "lamp_degradation", "confidence": "high",
         "summary": "Lamp drive rose 17 % in 12 hours.", "evidence": [],
         "recommended_actions": [], "citations": sorted(s.retrieved_ids)[:1],
         "test_validity_impact": "possible"}
    v = guardrails.check_diagnosis(d, kb=kb, retrieved_ids=s.retrieved_ids,
                                   returned_numbers=s.returned_numbers,
                                   context=s.get_alert_context())
    joined = " ".join(v)
    assert "17.0" in joined and "12.0" in joined


def test_telemetry_check_rejects_no_fault_on_real_fault(fitted, run700, kb):
    df, _, alerts = run700
    s = ToolSession(df, fitted["residual_z"], lamp_alert(alerts), kb)
    ctx = s.get_alert_context()
    s.search_manual("alert triage false alarm")
    d = {"likely_cause": "no_fault_or_false_alarm", "confidence": "high", "summary": "All fine.",
         "evidence": [], "recommended_actions": [], "citations": ["GEN-01"],
         "test_validity_impact": "none"}
    v = guardrails.check_diagnosis(d, kb=kb, retrieved_ids=s.retrieved_ids,
                                   returned_numbers=[], context=ctx)
    assert any(x.startswith("telemetry_check") for x in v)


def test_uncited_or_unretrieved_sections_rejected(fitted, run700, kb):
    df, _, alerts = run700
    s = ToolSession(df, fitted["residual_z"], lamp_alert(alerts), kb)
    ctx = s.get_alert_context()
    d = {"likely_cause": "lamp_degradation", "confidence": "high", "summary": "Lamp aging.",
         "evidence": [], "recommended_actions": [], "citations": ["LAMP-02"],
         "test_validity_impact": "possible"}
    v = guardrails.check_diagnosis(d, kb=kb, retrieved_ids=set(), returned_numbers=[], context=ctx)
    assert any("not retrieved" in x for x in v)


def test_model_that_never_submits_hits_step_limit(fitted, run700, kb):
    df, _, alerts = run700
    chatty = ScriptedLLM(lambda t: Turn("Let me think more...", []))
    r = diagnose(chatty, df, fitted["residual_z"], alerts[0], kb=kb)
    assert r.status == "error" and "step limit" in r.error


def test_provider_failure_is_contained(fitted, run700, kb):
    df, _, alerts = run700

    def boom(t):
        raise TimeoutError("API timed out")
    r = diagnose(ScriptedLLM(boom), df, fitted["residual_z"], alerts[0], kb=kb)
    assert r.status == "error" and "TimeoutError" in r.error


# --- rule baseline + eval harness ---------------------------------------------------

def test_eval_harness_end_to_end_offline(fitted, kb):
    scen = build_scenarios(fitted["residual_z"], seeds=[700], clean_seeds=[800], kb=kb)
    kinds = {s["kind"] for s in scen}
    assert {"fault", "false_alarm", "injection", "injection_unfiltered"} <= kinds
    res = run_eval(ScriptedLLM(scripted_policy), fitted["residual_z"], scen, kb=kb, verbose=False)
    summ = summarize(res)
    assert summ["n_scenarios"] == len(scen)
    assert summ["errors"] == 0.0
    assert summ["cause_accuracy"] == pytest.approx(summ["rule_baseline_accuracy"])


def test_rule_baseline_on_healthy_context_says_no_fault(fitted, kb):
    df, _ = simulate_run(801)
    ctx = ToolSession(df, fitted["residual_z"], {"start": 3000, "end": 3010, "channel": "rh_pct"},
                      kb).get_alert_context()
    assert rule_diagnosis(ctx) == "no_fault_or_false_alarm"


# --- provider adapters (no network: fake clients capture the request) -------------------

TRANSCRIPT = [
    {"role": "user", "content": "Alert on rh_pct."},
    {"role": "assistant", "text": None, "tool_calls": [ToolCall("t1", "get_alert_context", {}),
                                                       ToolCall("t2", "search_manual", {"query": "rh"})]},
    {"role": "tool", "id": "t1", "name": "get_alert_context", "content": "{}"},
    {"role": "tool", "id": "t2", "name": "search_manual", "content": "{}"},
]


def test_anthropic_adapter_formats_tool_use(monkeypatch):
    captured = {}

    def create(**kw):
        captured.update(kw)
        block = types.SimpleNamespace(type="tool_use", id="t3", name="submit_diagnosis", input={"a": 1})
        return types.SimpleNamespace(content=[block], usage=types.SimpleNamespace(
            input_tokens=10, output_tokens=5))
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    llm.model, llm.max_tokens = "test-model", 100
    turn = llm.chat("sys", TRANSCRIPT, [{"name": "x", "description": "d", "parameters": {}}])
    msgs = captured["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]  # both results in one turn
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["t1", "t2"]
    assert captured["tools"][0]["input_schema"] == {}
    assert turn.tool_calls[0].name == "submit_diagnosis" and turn.input_tokens == 10


def test_openai_adapter_formats_tool_calls():
    captured = {}

    def create(**kw):
        captured.update(kw)
        fn = types.SimpleNamespace(name="submit_diagnosis", arguments="{not json")
        msg = types.SimpleNamespace(content=None, tool_calls=[types.SimpleNamespace(id="t3", function=fn)])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    llm.model = "test-model"
    turn = llm.chat("sys", TRANSCRIPT, [{"name": "x", "description": "d", "parameters": {}}])
    msgs = captured["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool"]
    assert msgs[2]["tool_calls"][1]["function"]["arguments"] == '{"query": "rh"}'
    # malformed JSON from the model becomes a visible bad-arguments case, not a crash
    assert "_unparseable_arguments" in turn.tool_calls[0].args
