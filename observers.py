"""
observers.py — Three isolated pipeline factories (LangSmith, Langfuse, Phoenix).

Each factory returns a PipelineContext with its own agent, invoke config,
and cleanup/fetch functions. Backends are isolated: each gets a fresh agent
and scoped tracing.
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from graph import build_agent
from config import cfg


@dataclass
class PipelineContext:
    graph: Any                          # compiled LangGraph agent
    invoke_config: dict                 # passed to graph.invoke(config=this)
    run_id: str
    start_time: datetime
    cleanup: Callable[[], None]         # tear down tracing after invoke
    fetch_traces: Callable[[], dict]    # retrieve backend-specific traces


def create_langsmith_pipeline(collection, test_case_id: str, attack_type: str | None) -> PipelineContext:
    """Create an isolated pipeline with LangSmith tracing via explicit callback."""
    from langchain_core.tracers import LangChainTracer

    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)

    tags = [test_case_id]
    if attack_type:
        tags.append(attack_type)

    tracer = LangChainTracer(
        project_name=cfg.langsmith_project,
        tags=tags,
    )

    graph = build_agent(collection)

    invoke_config = {
        "run_id": run_id,
        "callbacks": [tracer],
    }

    def cleanup():
        pass  

    def fetch_traces():
        from simulate_attacks import _fetch_langsmith_trace
        return _fetch_langsmith_trace(run_id)

    return PipelineContext(
        graph=graph,
        invoke_config=invoke_config,
        run_id=run_id,
        start_time=start_time,
        cleanup=cleanup,
        fetch_traces=fetch_traces,
    )


def create_langfuse_pipeline(collection, test_case_id: str, attack_type: str | None) -> PipelineContext:
    """Create an isolated pipeline with Langfuse tracing via callback handler."""

    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)

    tags = [test_case_id]
    if attack_type:
        tags.append(attack_type)

    handler = None
    try:
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError:
            from langfuse import CallbackHandler 

        try:
            handler = CallbackHandler(
                secret_key=cfg.langfuse_secret_key,
                public_key=cfg.langfuse_public_key,
                host=cfg.langfuse_host,
                trace_name=f"{test_case_id}_{attack_type or 'benign'}",
                tags=tags,
            )
        except TypeError:
            os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
            os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
            os.environ["LANGFUSE_HOST"] = cfg.langfuse_host
            handler = CallbackHandler()
    except Exception as e:
        print(f"[Langfuse] Init error ({e}); tracing disabled for this run.", flush=True)

    graph = build_agent(collection)

    invoke_config = {
        "run_id": run_id,
        "callbacks": [handler] if handler else [],
    }

    def cleanup():
        if handler is not None:
            try:
                if hasattr(handler, "flush"):
                    handler.flush()
                elif hasattr(handler, "langfuse") and hasattr(handler.langfuse, "flush"):
                    handler.langfuse.flush()
            except Exception:
                pass

    def fetch_traces():
        from simulate_attacks import _fetch_langfuse_traces
        return _fetch_langfuse_traces(start_time)

    return PipelineContext(
        graph=graph,
        invoke_config=invoke_config,
        run_id=run_id,
        start_time=start_time,
        cleanup=cleanup,
        fetch_traces=fetch_traces,
    )


def create_phoenix_pipeline(collection, test_case_id: str, attack_type: str | None) -> PipelineContext:
    """Create an isolated pipeline with Phoenix OTEL tracing."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)

    instrumentor = None
    provider = None

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.langchain import LangChainInstrumentor

        import logging
        logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

        resource = Resource(attributes={
            "service.name": cfg.phoenix_project_name,
            "test_case_id": test_case_id,
        })
        exporter = OTLPSpanExporter(endpoint=cfg.phoenix_collector_endpoint, timeout=2)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        instrumentor = LangChainInstrumentor()
        instrumentor.instrument(tracer_provider=provider)
    except Exception as e:
        print(f"[Phoenix] Init error ({e}); tracing disabled for this run.", flush=True)

    graph = build_agent(collection)

    invoke_config = {
        "run_id": run_id,
    }

    def cleanup():
        if instrumentor is not None:
            try:
                instrumentor.uninstrument()
            except Exception:
                pass
        if provider is not None:
            try:
                provider.force_flush()
                provider.shutdown()
            except Exception:
                pass

    def fetch_traces():
        from simulate_attacks import _fetch_phoenix_spans
        return _fetch_phoenix_spans(start_time)

    return PipelineContext(
        graph=graph,
        invoke_config=invoke_config,
        run_id=run_id,
        start_time=start_time,
        cleanup=cleanup,
        fetch_traces=fetch_traces,
    )
