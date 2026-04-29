import json
import time
import pandas as pd
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse
from phoenix.session.client import Client as PhoenixClient
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from config import cfg


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _jsonable(method())
            except TypeError:
                pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _fetch_langsmith_trace(run_id: str) -> dict:
    client = Client()
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            run = client.read_run(run_id, load_child_runs=True)
        except LangSmithNotFoundError:
            time.sleep(2)
            continue
        return _jsonable(run)
        time.sleep(2)
    raise TimeoutError(f"LangSmith run {run_id} was not readable within 20s")


def _fetch_phoenix_spans(start_time: datetime, session_id: str | None = None) -> List[dict]:
    client = PhoenixClient(endpoint=_phoenix_client_endpoint(), warn_if_server_not_running=False)
    deadline = time.time() + 120
    last_seen_count = 0
    while time.time() < deadline:
        frame = client.query_spans(
            start_time=start_time,
            project_name=cfg.phoenix_project_name,
            limit=5000,
            timeout=10,
        )
        if isinstance(frame, list):
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

        spans = _jsonable(frame.to_dict(orient="records"))
        if spans:
            last_seen_count = len(spans)
            return spans
        time.sleep(2)
    raise TimeoutError(
        f"Phoenix spans were not readable within 120s for session {session_id}. "
        f"Last seen matching span count: {last_seen_count}"
    )


def _fetch_langfuse_trace(client, trace_id: str, session_id: str, start_time: datetime) -> dict:
    deadline = time.time() + 120
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
