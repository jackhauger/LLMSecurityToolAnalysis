import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from config import cfg
from graph import build_agent
from langchain_core.tracers import LangChainTracer
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from simulate_attacks import _fetch_langfuse_trace, _fetch_phoenix_spans, _fetch_langsmith_trace
from openinference.instrumentation import dangerously_using_project, using_attributes
from phoenix.otel import register

_phoenix_provider = None
_phoenix_instrumentor = None


@dataclass
class PipelineContext:
    graph: Any
    invoke_config: dict
    run_id: str
    start_time: datetime
    cleanup: Callable[[], None]
    fetch_traces: Callable[[], dict]
    trace_debug: dict[str, Any] | None = None


def create_langsmith_pipeline(collection, _test_case_id: str, _attack_type: str | None) -> PipelineContext:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    graph = build_agent(collection)
    tracer = LangChainTracer(project_name=cfg.langsmith_project)
    metadata = {
        "session_id": run_id,
        "backend": "langsmith",
    }

    def cleanup():
        tracer.wait_for_futures()

    def fetch_traces():
        return _fetch_langsmith_trace(run_id)

    return PipelineContext(
        graph=graph,
        invoke_config={
            "run_id": run_id,
            "callbacks": [tracer],
            "metadata": metadata,
        },
        run_id=run_id,
        start_time=start_time,
        cleanup=cleanup,
        fetch_traces=fetch_traces,
        trace_debug={"backend": "langsmith"},
    )


def create_langfuse_pipeline(collection, _test_case_id: str, _attack_type: str | None) -> PipelineContext:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    graph = build_agent(collection)
    metadata = {
        "session_id": run_id,
        "backend": "langfuse",
        "langfuse_session_id": run_id,
    }

    client = Langfuse(
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key,
        base_url=cfg.langfuse_host,
        timeout=20,
        tracing_enabled=True,
        flush_at=5,
        flush_interval=1.0,
    )
    trace_id = client.create_trace_id(seed=run_id)

    os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = cfg.langfuse_host

    handler = CallbackHandler(
        public_key=cfg.langfuse_public_key,
        trace_context={"trace_id": trace_id},
        update_trace=True,
    )
    
    def cleanup():
        if hasattr(handler, "flush"):
            handler.flush()
        else:
            (handler.client or client).flush()

    def fetch_traces():
        return _fetch_langfuse_trace(
            client=handler.client or client,
            trace_id=handler.last_trace_id or trace_id,
            session_id=run_id,
            start_time=start_time,
        )

    return PipelineContext(
        graph=graph,
        invoke_config={
            "run_id": run_id,
            "callbacks": [handler],
            "metadata": metadata,
        },
        run_id=run_id,
        start_time=start_time,
        cleanup=cleanup,
        fetch_traces=fetch_traces,
        trace_debug={
            "backend": "langfuse",
            "client": handler.client or client,
            "trace_id": handler.last_trace_id or trace_id,
        },
    )


def create_phoenix_pipeline(collection, _test_case_id: str, _attack_type: str | None) -> PipelineContext:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["PHOENIX_PROJECT_NAME"] = cfg.phoenix_project_name

    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    graph = build_agent(collection)
    metadata = {
        "session_id": run_id,
        "backend": "arize phoenix",
    }

    global _phoenix_provider, _phoenix_instrumentor
    if _phoenix_provider is None or _phoenix_instrumentor is None:
        _phoenix_provider = register(
            project_name=cfg.phoenix_project_name,
            endpoint=cfg.phoenix_collector_endpoint,
            protocol="grpc" if ":4317" in (cfg.phoenix_collector_endpoint or "") else "http/protobuf",
            batch=True,
            auto_instrument=True,
            set_global_tracer_provider=False,
            verbose=False,
        )
        _phoenix_instrumentor = True

    class PhoenixGraph:
        def invoke(self, *args, **kwargs):
            with (
                dangerously_using_project(cfg.phoenix_project_name),
                using_attributes(session_id=run_id, metadata=metadata),
            ):
                return graph.invoke(*args, **kwargs)

    def cleanup():
        _phoenix_provider.force_flush()

    def fetch_traces():
        return _fetch_phoenix_spans(start_time=start_time, session_id=run_id)

    return PipelineContext(
        graph=PhoenixGraph(),
        invoke_config={
            "run_id": run_id,
            "metadata": metadata,
        },
        run_id=run_id,
        start_time=start_time,
        cleanup=cleanup,
        fetch_traces=fetch_traces,
        trace_debug={"backend": "arize phoenix"},
    )


def shutdown_phoenix() -> None:
    global _phoenix_provider, _phoenix_instrumentor

    if _phoenix_provider is not None:
        _phoenix_provider.force_flush()
    if _phoenix_provider is not None:
        _phoenix_provider.shutdown()
    _phoenix_provider = None
    _phoenix_instrumentor = None
