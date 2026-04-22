"""
simulate_attacks.py — Trace fetchers.
"""

import json
import math
import time
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse

from config import cfg


def _fetch_langsmith_trace(run_id: str) -> dict:
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError

    client = Client()
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            run = client.read_run(run_id, load_child_runs=True)
        except LangSmithNotFoundError:
            time.sleep(2)
            continue
        return {
            "inputs": run.inputs,
            "outputs": run.outputs,
            "status": run.status,
            "start_time": str(run.start_time),
            "end_time": str(run.end_time),
            "total_tokens": getattr(run, "total_tokens", None),
            "prompt_tokens": getattr(run, "prompt_tokens", None),
            "completion_tokens": getattr(run, "completion_tokens", None),
            "child_runs": [
                {
                    "name": child.name,
                    "inputs": child.inputs,
                    "outputs": child.outputs,
                    "total_tokens": getattr(child, "total_tokens", None),
                    "prompt_tokens": getattr(child, "prompt_tokens", None),
                    "completion_tokens": getattr(child, "completion_tokens", None),
                    "child_runs": [
                        {
                            "name": grandchild.name,
                            "inputs": grandchild.inputs,
                            "outputs": grandchild.outputs,
                            "total_tokens": getattr(grandchild, "total_tokens", None),
                            "prompt_tokens": getattr(grandchild, "prompt_tokens", None),
                            "completion_tokens": getattr(grandchild, "completion_tokens", None),
                        }
                        for grandchild in (child.child_runs or [])
                    ],
                }
                for child in (run.child_runs or [])
            ],
        }
        time.sleep(2)
    raise TimeoutError(f"LangSmith run {run_id} was not readable within 20s")


def _fetch_phoenix_spans(start_time: datetime, session_id: str | None = None) -> List[dict]:
    from phoenix.session.client import Client as PhoenixClient

    client = PhoenixClient(endpoint=_phoenix_client_endpoint(), warn_if_server_not_running=False)
    deadline = time.time() + 20
    while time.time() < deadline:
        frame = client.query_spans(
            start_time=start_time,
            project_name=cfg.phoenix_project_name,
            limit=500,
            timeout=10,
        )
        if isinstance(frame, list):
            import pandas as pd

            frame = pd.concat([item for item in frame if not getattr(item, "empty", False)], ignore_index=True)
        if frame is None or getattr(frame, "empty", False):
            time.sleep(2)
            continue

        session_col = None
        for candidate in ("attributes.session.id", "session.id", "session_id", "attributes.session_id"):
            if candidate in frame.columns:
                session_col = candidate
                break
        if session_id and session_col is not None:
            frame = frame[frame[session_col].astype(str) == session_id]
        if getattr(frame, "empty", False):
            time.sleep(2)
            continue

        spans = frame.to_dict(orient="records")
        if spans:
            trimmed = []
            for span in spans[:200]:
                trimmed.append(
                    {
                        key: value
                        for key, value in span.items()
                        if not isinstance(value, str) or len(value) <= 4000
                    }
                    | {
                        key: value[:4000] + f"... [truncated {len(value) - 4000} chars]"
                        for key, value in span.items()
                        if isinstance(value, str) and len(value) > 4000
                    }
                )
            has_answer_evidence = False
            for span in trimmed:
                name = str(span.get("name", "")).lower()
                kind = str(
                    span.get("attributes.openinference.span.kind")
                    or span.get("span_kind")
                    or ""
                ).upper()
                output_value = (
                    span.get("attributes.output.value")
                    or span.get("output")
                    or span.get("attributes.llm.output_messages")
                )
                if (
                    kind == "LLM"
                    or name == "answer"
                    or name == "chatgooglegenerativeai"
                    or span.get("attributes.llm.token_count.total") is not None
                    or span.get("attributes.llm.token_count.prompt") is not None
                    or span.get("attributes.llm.token_count.completion") is not None
                    or output_value
                ):
                    has_answer_evidence = True
                    break
            if has_answer_evidence:
                return trimmed
        time.sleep(2)
    raise TimeoutError(f"Phoenix spans were not readable within 20s for session {session_id}")


def _fetch_langfuse_trace(client, trace_id: str, session_id: str, start_time: datetime) -> dict:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if trace_id:
                trace = client.api.trace.get(trace_id)
                trace_dict = trace.dict()
                if trace_dict.get("observations"):
                    return trace_dict
        except Exception:
            pass
        try:
            trace_list = client.api.trace.list(
                limit=1,
                session_id=session_id,
                from_timestamp=start_time,
                order_by="timestamp.desc",
            )
            candidates = getattr(trace_list, "data", None) or []
            if candidates:
                trace_id = getattr(candidates[0], "id", None)
                if trace_id:
                    trace = client.api.trace.get(trace_id)
                    trace_dict = trace.dict()
                    if trace_dict.get("observations"):
                        return trace_dict
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Langfuse trace was not readable within 20s for session {session_id}")


def _phoenix_client_endpoint() -> str:
    parsed = urlparse((cfg.phoenix_collector_endpoint or "").strip() or "http://localhost:6006/v1/traces")
    if parsed.scheme and parsed.netloc:
        host = parsed.hostname or "localhost"
        if parsed.port == 4317:
            return f"{parsed.scheme}://{host}:6006"
        return f"{parsed.scheme}://{parsed.netloc}"
    return "http://localhost:6006"
