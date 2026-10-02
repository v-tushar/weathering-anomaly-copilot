"""Provider-agnostic chat-with-tools interface.

The agent keeps a neutral transcript:
    {"role": "user", "content": str}
    {"role": "assistant", "text": str | None, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "id": str, "name": str, "content": str}
Each provider converts that transcript to its own API format on every call,
so the agent loop, guardrails and eval stay the same for Anthropic, OpenAI,
or the scripted offline stand-in used in tests.

Model names come from env vars (COPILOT_MODEL) because they change often.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

try:  # optional: pick up API keys from a local .env (see .env.example)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str | None
    tool_calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class AnthropicLLM:
    name = "anthropic"

    # 1500 was too small: a live run truncated submit_diagnosis mid-arguments, which the
    # schema check rejected and which used up the one retry.
    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        # `or`, not a get() default: an empty COPILOT_MODEL= line in .env means "use the default".
        self.model = model or os.environ.get("COPILOT_MODEL") or "claude-sonnet-5"
        self.max_tokens = max_tokens

    def chat(self, system: str, transcript: list[dict], tools: list[dict]) -> Turn:
        msgs: list[dict] = []
        for m in transcript:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                           for c in m["tool_calls"]]
                msgs.append({"role": "assistant", "content": blocks})
            else:  # tool result; consecutive results share one user message
                block = {"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
        t = time.time()
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=msgs,
            tools=[{"name": s["name"], "description": s["description"],
                    "input_schema": s["parameters"]} for s in tools])
        text = "".join(b.text for b in r.content if b.type == "text") or None
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        return Turn(text, calls, r.usage.input_tokens, r.usage.output_tokens, time.time() - t)


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None):
        import openai
        self.client = openai.OpenAI()  # reads OPENAI_API_KEY
        self.model = model or os.environ.get("COPILOT_MODEL") or "gpt-4o-mini"

    def chat(self, system: str, transcript: list[dict], tools: list[dict]) -> Turn:
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in transcript:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("text")}
                if m["tool_calls"]:
                    msg["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                         for c in m["tool_calls"]]
                msgs.append(msg)
            else:
                msgs.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
        t = time.time()
        r = self.client.chat.completions.create(
            model=self.model, messages=msgs,
            tools=[{"type": "function", "function": s} for s in tools])
        m = r.choices[0].message
        calls = []
        for c in m.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": c.function.arguments}
            calls.append(ToolCall(c.id, c.function.name, args))
        u = r.usage
        return Turn(m.content, calls, u.prompt_tokens if u else 0, u.completion_tokens if u else 0,
                    time.time() - t)


@dataclass
class ScriptedLLM:
    """Offline stand-in that follows a fixed policy. Not a language model.

    `policy(transcript) -> Turn` decides the next turn from the transcript so far.
    It exercises the agent loop, tool dispatch and guardrails in tests and in
    offline eval runs. Its accuracy says nothing about an LLM's accuracy.
    """
    policy: Callable[[list[dict]], Turn]
    name: str = "scripted"
    calls: int = field(default=0)

    def chat(self, system: str, transcript: list[dict], tools: list[dict]) -> Turn:
        self.calls += 1
        return self.policy(transcript)


def make_llm(provider: str, model: str | None = None):
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    if key and not os.environ.get(key):
        raise SystemExit(f"{key} is not set. Set it first, e.g.\n"
                         f"  macOS/Linux:  export {key}=...\n"
                         f"  Windows (PowerShell):  $env:{key}=\"...\"\n"
                         f"or run with --provider scripted for the offline version.")
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "scripted":
        from .baseline import scripted_policy
        return ScriptedLLM(scripted_policy)
    raise ValueError(f"unknown provider {provider!r} (anthropic | openai | scripted)")
