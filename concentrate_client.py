"""
Minimal requests-based client for the Concentrate AI API.

Docs: https://docs.concentrate.ai

Implements:
- POST /v1/responses (streaming + non-streaming)
- GET  /v1/responses/health (200 with empty body)
- GET  /v1/models (no auth required)
- GET  /v1/models/providers (no auth required)
- POST /v1/messages (beta)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


@dataclass
class CallStats:
    model: str
    status_code: int
    latency_ms: int
    used_fallback: bool
    error: Optional[str] = None


class ConcentrateClient:
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.concentrate.ai",
        timeout_s: int = 60,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = (api_key or "").strip()
        self.api_base = (api_base or "https://api.concentrate.ai").rstrip("/")
        self.timeout_s = int(timeout_s)
        self._session = session or requests.Session()

    def _headers(self, accept: str = "application/json") -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ---------------- Discovery endpoints ----------------
    def list_models(self) -> List[Dict[str, Any]]:
        url = f"{self.api_base}/v1/models"
        r = self._session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("data", [])

    def list_providers(self) -> List[Dict[str, Any]]:
        url = f"{self.api_base}/v1/models/providers"
        r = self._session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("data", [])

    def health(self) -> Dict[str, Any]:
        """
        GET /v1/responses/health
        Docs: success is 200 with an empty body.
        We return a small dict for UI convenience.
        """
        url = f"{self.api_base}/v1/responses/health"
        start = time.time()
        r = self._session.get(url, headers={"Accept": "application/json"}, timeout=self.timeout_s)
        latency_ms = int((time.time() - start) * 1000)
        r.raise_for_status()
        return {"ok": True, "status_code": r.status_code, "latency_ms": latency_ms}

    # ---------------- Responses ----------------
    def create_response(self, payload: Dict[str, Any], *, used_fallback: bool = False) -> Tuple[Dict[str, Any], CallStats]:
        url = f"{self.api_base}/v1/responses"
        start = time.time()
        r = self._session.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        latency_ms = int((time.time() - start) * 1000)

        if r.status_code == 200:
            return (
                r.json(),
                CallStats(
                    model=str(payload.get("model", "")),
                    status_code=200,
                    latency_ms=latency_ms,
                    used_fallback=used_fallback,
                ),
            )

        try:
            err_text = json.dumps(r.json(), ensure_ascii=False)
        except Exception:
            err_text = (r.text or "")[:4000]

        return (
            {"http_status": r.status_code, "error_body": err_text},
            CallStats(
                model=str(payload.get("model", "")),
                status_code=r.status_code,
                latency_ms=latency_ms,
                used_fallback=used_fallback,
                error=err_text,
            ),
        )

    def stream_response_events(self, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:

        url = f"{self.api_base}/v1/responses"
        payload = {**payload, "stream": True}

        with self._session.post(
            url,
            headers=self._headers(accept="text/event-stream"),
            json=payload,
            stream=True,
            timeout=self.timeout_s,
        ) as r:
            if r.status_code != 200:
                try:
                    yield {"type": "error", "status_code": r.status_code, "error": r.json()}
                except Exception:
                    yield {"type": "error", "status_code": r.status_code, "error": (r.text or "")[:4000]}
                return

            buffer = ""
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        data_str = line.split("data:", 1)[1].strip()
                        if data_str == "[DONE]":
                            return
                        try:
                            evt = json.loads(data_str)
                        except Exception:
                            continue
                        yield evt

    def stream_response_text(self, payload: Dict[str, Any]) -> Iterable[str]:
        """
        Yields incremental text tokens from stream events.
        Also emits the final full text when we see output_text.done or response.completed.
        """
        for evt in self.stream_response_events(payload):
            if isinstance(evt, dict) and evt.get("type") == "error":
                yield f"\n[ERROR] {json.dumps(evt, ensure_ascii=False)}\n"
                return

            # 1) token deltas
            if isinstance(evt, dict) and isinstance(evt.get("delta"), str) and evt["delta"]:
                yield evt["delta"]
                continue

            # 2) done events carry full text
            if isinstance(evt, dict) and isinstance(evt.get("text"), str) and evt["text"]:
                yield evt["text"]
                continue

            # 3) response.completed includes full response object; extract text from response.output[*].content[*].text
            if isinstance(evt, dict) and evt.get("type") == "response.completed" and isinstance(evt.get("response"), dict):
                resp = evt["response"]
                out = resp.get("output")
                if isinstance(out, list):
                    parts = []
                    for item in out:
                        if isinstance(item, dict):
                            content = item.get("content")
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"]:
                                        parts.append(c["text"])
                    if parts:
                        yield "".join(parts)
                return


    # ---------------- Messages ----------------
    def create_message(self, payload: Dict[str, Any], *, used_fallback: bool = False) -> Tuple[Dict[str, Any], CallStats]:
        url = f"{self.api_base}/v1/messages"
        start = time.time()
        r = self._session.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        latency_ms = int((time.time() - start) * 1000)

        if r.status_code == 200:
            return (
                r.json(),
                CallStats(
                    model=str(payload.get("model", "")),
                    status_code=200,
                    latency_ms=latency_ms,
                    used_fallback=used_fallback,
                ),
            )

        try:
            err_text = json.dumps(r.json(), ensure_ascii=False)
        except Exception:
            err_text = (r.text or "")[:4000]

        return (
            {"http_status": r.status_code, "error_body": err_text},
            CallStats(
                model=str(payload.get("model", "")),
                status_code=r.status_code,
                latency_ms=latency_ms,
                used_fallback=used_fallback,
                error=err_text,
            ),
        )


def extract_text(resp: Any) -> str:
    """
    Best-effort extraction across Concentrate/OpenAI-compatible payloads.

    Handles:
    - Streaming events: {"delta": "..."} or {"text": "..."}
    - Non-stream: {"output":[{"content":[{"text":"..."}]}]}
    - Nested: {"response": {...}} or {"item": {...}} or {"part": {...}}
    - ChatCompletions compat: {"choices":[{"delta":{"content":"..."}}]}
    """
    def _from_obj(obj: Any) -> str:
        if not isinstance(obj, dict):
            return ""

        if isinstance(obj.get("delta"), str) and obj["delta"]:
            return obj["delta"]
        if isinstance(obj.get("text"), str) and obj["text"]:
            return obj["text"]
        if isinstance(obj.get("output_text"), str) and obj["output_text"]:
            return obj["output_text"]

        out = obj.get("output")
        if isinstance(out, list):
            parts: List[str] = []
            for item in out:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            t = c.get("text")
                            if isinstance(t, str) and t:
                                parts.append(t)
            if parts:
                return "".join(parts)

        nested = obj.get("response")
        if isinstance(nested, dict):
            t = _from_obj(nested)
            if t:
                return t

        item = obj.get("item")
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                parts: List[str] = []
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("text")
                        if isinstance(t, str) and t:
                            parts.append(t)
                if parts:
                    return "".join(parts)

        part = obj.get("part")
        if isinstance(part, dict):
            t = part.get("text")
            if isinstance(t, str) and t:
                return t

        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            d = choices[0].get("delta")
            if isinstance(d, dict):
                c = d.get("content")
                if isinstance(c, str) and c:
                    return c

        return ""

    return _from_obj(resp)
