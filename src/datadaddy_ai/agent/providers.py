"""LLM provider interface. complete(messages, tools) -> LLMResponse(text, tool_calls, usage, latency).

Canonical message format (provider-agnostic):
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall-as-dict]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Selection by LLM_PROVIDER=mock|anthropic|openai_compat. Missing credentials raise ProviderConfigError;
there is never a silent fallback to the mock.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class ProviderConfigError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    latency_s: float = 0.0


class LLMProvider:
    name = "base"
    model = ""

    def complete(
        self, messages: list[dict], tools: list[dict], temperature: float = 0.0, max_tokens: int = 1024
    ) -> LLMResponse:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic, scriptable. `script` is a list of LLMResponse/dicts consumed in order, or a callable
    (messages, tools) -> LLMResponse|dict. Raises when a list script is exhausted (tests must be exact)."""

    name = "mock"
    model = "mock"

    def __init__(self, script: list | Callable | None = None):
        self.script = script if script is not None else []
        self.i = 0
        self.calls: list[list[dict]] = []

    @staticmethod
    def _coerce(r) -> LLMResponse:
        if isinstance(r, LLMResponse):
            return r
        calls = [
            ToolCall(c.get("id", f"call_{k}"), c["name"], c.get("arguments", {}))
            for k, c in enumerate(r.get("tool_calls", []))
        ]
        return LLMResponse(
            r.get("text", ""), calls, {"input_tokens": r.get("in", 10), "output_tokens": r.get("out", 5)}
        )

    def complete(self, messages, tools, temperature=0.0, max_tokens=1024) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        t = time.perf_counter()
        if callable(self.script):
            r = self._coerce(self.script(messages, tools))
        else:
            if self.i >= len(self.script):
                raise RuntimeError("MockProvider script exhausted")
            r = self._coerce(self.script[self.i])
            self.i += 1
        r.latency_s = time.perf_counter() - t
        return r


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderConfigError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        import anthropic

        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or ""
        if not self.model:
            raise ProviderConfigError("ANTHROPIC_MODEL is not set")

    @staticmethod
    def _convert(messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                    for c in m.get("tool_calls", [])
                ]
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return system, out

    def complete(self, messages, tools, temperature=0.0, max_tokens=1024) -> LLMResponse:
        system, msgs = self._convert(messages)
        t = time.perf_counter()
        r = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=msgs,
            tools=[
                {"name": x["name"], "description": x["description"], "input_schema": x["parameters"]}
                for x in tools
            ],
        )
        lat = time.perf_counter() - t
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        return LLMResponse(
            text, calls, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}, lat
        )


class OpenAICompatProvider(LLMProvider):
    """OpenAI-compatible /chat/completions (Ollama, vLLM, llama.cpp server...)."""

    name = "openai_compat"

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ.get("OPENAI_COMPAT_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("OPENAI_COMPAT_MODEL") or ""
        self.api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "")
        if not self.base_url or not self.model:
            raise ProviderConfigError(
                "LLM_PROVIDER=openai_compat needs OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_MODEL"
            )

    @staticmethod
    def _convert(messages):
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                            }
                            for c in m["tool_calls"]
                        ],
                    }
                )
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def complete(self, messages, tools, temperature=0.0, max_tokens=1024) -> LLMResponse:
        import httpx

        body = {
            "model": self.model,
            "messages": self._convert(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [{"type": "function", "function": x} for x in tools],
        }
        t = time.perf_counter()
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            json=body,
            timeout=300,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        r.raise_for_status()
        lat = time.perf_counter() - t
        d = r.json()
        msg = d["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            a = c["function"].get("arguments", "{}")
            try:
                args = json.loads(a) if isinstance(a, str) else dict(a)
            except json.JSONDecodeError:
                args = {"__unparsable__": a}
            calls.append(ToolCall(c.get("id", f"call_{len(calls)}"), c["function"]["name"], args))
        u = d.get("usage", {})
        return LLMResponse(
            msg.get("content") or "",
            calls,
            {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)},
            lat,
        )


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai_compat":
        return OpenAICompatProvider()
    raise ProviderConfigError(f"LLM_PROVIDER must be one of mock|anthropic|openai_compat, got {name!r}")
