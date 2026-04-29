import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import click

import database
import observers
from config import cfg
from langchain_core.messages import HumanMessage
from langsmith import Client as LangSmithClient
from langsmith.utils import LangSmithNotFoundError
from phoenix.session.client import Client as PhoenixClient


BACKEND_FACTORIES = {
    "langsmith": "create_langsmith_pipeline",
    "langfuse": "create_langfuse_pipeline",
    "arize phoenix": "create_phoenix_pipeline",
}


def _load_cases(dataset_path: Path) -> list[dict]:
    return json.loads(dataset_path.read_text())


def _choose_case(cases: list[dict], case_id: str | None) -> dict:
    if case_id:
        for case in cases:
            if case.get("id") == case_id:
                return case
        raise ValueError(f"Case id '{case_id}' not found in dataset")
    for case in cases:
        if not case.get("benign") and case.get("poisoned_document") is not None:
            return case
    raise ValueError("No non-benign poisoned case found in dataset")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
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


def _fetch_langsmith_raw(run_id: str) -> dict:
    client = LangSmithClient()
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            run = client.read_run(run_id, load_child_runs=True)
            return _jsonable(run)
        except LangSmithNotFoundError:
            time.sleep(2)
    raise TimeoutError(f"LangSmith run {run_id} was not readable within 120s")


def _fetch_langfuse_raw(client, trace_id: str, session_id: str, start_time: datetime) -> dict:
    deadline = time.time() + 120
    current_trace_id = trace_id
    while time.time() < deadline:
        try:
            if current_trace_id:
                trace = client.api.trace.get(current_trace_id)
                return _jsonable(trace)
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
                current_trace_id = getattr(candidates[0], "id", None) or current_trace_id
                if current_trace_id:
                    trace = client.api.trace.get(current_trace_id)
                    return _jsonable(trace)
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Langfuse trace was not readable within 120s for session {session_id}")


def _phoenix_client_endpoint() -> str:
    from simulate_attacks import _phoenix_client_endpoint as helper

    return helper()


def _fetch_phoenix_raw(start_time: datetime, session_id: str) -> list[dict]:
    client = PhoenixClient(endpoint=_phoenix_client_endpoint(), warn_if_server_not_running=False)
    deadline = time.time() + 120
    while time.time() < deadline:
        frame = client.query_spans(
            start_time=start_time,
            project_name=cfg.phoenix_project_name,
            limit=10000,
            timeout=10,
        )
        if frame is None or getattr(frame, "empty", False):
            time.sleep(2)
            continue
        session_col = None
        for candidate in ("attributes.session.id", "session.id", "session_id", "attributes.session_id"):
            if candidate in frame.columns:
                session_col = candidate
                break
        if session_col is not None:
            frame = frame[frame[session_col].astype(str) == session_id]
        if getattr(frame, "empty", False):
            time.sleep(2)
            continue
        return _jsonable(frame.to_dict(orient="records"))
    raise TimeoutError(f"Phoenix spans were not readable within 120s for session {session_id}")


def _fetch_backend_raw(ctx) -> dict | list[dict]:
    debug = ctx.trace_debug or {}
    backend = debug.get("backend")
    if backend == "langsmith":
        return _fetch_langsmith_raw(ctx.run_id)
    if backend == "langfuse":
        return _fetch_langfuse_raw(
            client=debug["client"],
            trace_id=debug.get("trace_id", ""),
            session_id=ctx.run_id,
            start_time=ctx.start_time,
        )
    if backend == "arize phoenix":
        return _fetch_phoenix_raw(start_time=ctx.start_time, session_id=ctx.run_id)
    raise ValueError(f"Unsupported backend for raw fetch: {backend}")


@click.command()
@click.option(
    "--dataset",
    default="fifteen_case_subset.json",
    type=click.Path(exists=True),
    help="Dataset JSON to pull one example from.",
)
@click.option(
    "--case-id",
    default=None,
    help="Specific dataset case id to run. Defaults to the first non-benign poisoned case.",
)
@click.option(
    "--output-dir",
    default="raw_trace_samples",
    type=click.Path(),
    help="Directory where raw trace dumps should be written.",
)
def main(dataset: str, case_id: str | None, output_dir: str) -> None:
    cfg.validate()
    dataset_path = Path(dataset)
    cases = _load_cases(dataset_path)
    case = _choose_case(cases, case_id)

    collection = database.get_or_create_collection()
    selected_case_id = case.get("id") or "sample_case"
    output_root = Path(output_dir) / selected_case_id
    output_root.mkdir(parents=True, exist_ok=True)

    prompt = case["input_prompt"]
    attack_type = case["attack_type"]
    poisoned_document = case.get("poisoned_document")
    poisoned_doc_id = f"poison-{selected_case_id}"
    injected = False

    try:
        if poisoned_document is not None:
            database.inject_poisoned_document(
                collection,
                poisoned_doc_id,
                poisoned_document,
                {"source_id": selected_case_id or f"reference-{uuid.uuid4().hex[:8]}"},
            )
            injected = True

        runs = {}
        for backend_name, factory_name in BACKEND_FACTORIES.items():
            factory_fn = getattr(observers, factory_name)
            ctx = factory_fn(collection, selected_case_id, attack_type)
            try:
                output = ctx.graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=ctx.invoke_config,
                )
            finally:
                ctx.cleanup()

            final_response = ""
            if output and "messages" in output and output["messages"]:
                content = output["messages"][-1].content
                if isinstance(content, list):
                    final_response = " ".join(
                        block.get("text", "") for block in content if isinstance(block, dict)
                    )
                else:
                    final_response = str(content)

            raw_trace = _fetch_backend_raw(ctx)
            backend_dir = output_root / backend_name
            backend_dir.mkdir(parents=True, exist_ok=True)
            (backend_dir / "trace.json").write_text(json.dumps(raw_trace, indent=2, default=str))
            (backend_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backend": backend_name,
                        "run_id": ctx.run_id,
                        "start_time": ctx.start_time.isoformat(),
                        "final_response": final_response,
                    },
                    indent=2,
                )
            )
            runs[backend_name] = {"run_id": ctx.run_id, "trace_file": str(backend_dir / "trace.json")}

        (output_root / "case.json").write_text(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "case": case,
                    "runs": runs,
                },
                indent=2,
                default=str,
            )
        )
    finally:
        if injected:
            database.remove_poisoned_document(collection, poisoned_doc_id)
        observers.shutdown_phoenix()


if __name__ == "__main__":
    main()