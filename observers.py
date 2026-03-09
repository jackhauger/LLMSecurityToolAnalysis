"""
observers.py — ObservabilityManager: LangSmith + Phoenix (OTEL) + Langfuse.

Initialization order:
  1. LangSmith  — sets LANGCHAIN_PROJECT env var; auto-activates via LangChain env hooks
  2. Phoenix    — launches UI, creates TracerProvider with OTLP exporter, instruments LangChain
  3. Langfuse   — creates CallbackHandler for injection into LLM calls

Public API:
  obs.get_callbacks()            -> list of callbacks for LLM calls
  obs.wrap_node(name, fn)        -> wrapped node function with OTEL span
  obs.shutdown()                 -> flush all backends
"""

import json
import os
import threading
import time
from typing import Any, Callable, Dict, List

from config import cfg


class ObservabilityManager:
    def __init__(self) -> None:
        self._tracer_provider = None
        self._tracer = None
        self._langfuse_handler = None
        self._phoenix_available = False
        self._langfuse_available = False

        self._init_langsmith()

        # Phoenix and Langfuse import heavy packages that can be slow on first load.
        # Start each in a daemon thread and do NOT join — startup returns immediately.
        # wrap_node() reads self._tracer at call time, so it benefits from async init.
        for name, fn in [("Phoenix", self._init_phoenix), ("Langfuse", self._init_langfuse)]:
            t = threading.Thread(target=fn, daemon=True, name=f"obs-init-{name}")
            t.start()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_langsmith(self) -> None:
        """Configure LangSmith via environment variables."""
        os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project
        if cfg.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
        if cfg.langsmith_tracing.lower() in ("true", "1", "yes"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        print("[ObservabilityManager] LangSmith configured.", flush=True)

    def _init_phoenix(self) -> None:
        """Register OTEL TracerProvider and export spans to a running Phoenix server.

        Phoenix must be started separately (in a separate terminal):
            phoenix serve
        UI will be at http://localhost:6006; HTTP OTLP collector at http://localhost:6006/v1/traces.
        """
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Suppress the verbose connection-refused stack trace when Phoenix isn't running
            import logging
            logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

            resource = Resource(attributes={"service.name": cfg.phoenix_project_name})
            exporter = OTLPSpanExporter(endpoint=cfg.phoenix_collector_endpoint, timeout=2)
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(exporter))

            otel_trace.set_tracer_provider(provider)
            self._tracer_provider = provider
            self._tracer = otel_trace.get_tracer(__name__)
            self._phoenix_available = True
            print(
                "[ObservabilityManager] Phoenix OTEL tracing active "
                f"(exporting to {cfg.phoenix_collector_endpoint}). "
                "Start Phoenix with: phoenix serve",
                flush=True,
            )
        except ImportError as e:
            print(f"[ObservabilityManager] Phoenix not available ({e}); skipping.", flush=True)
        except Exception as e:
            print(f"[ObservabilityManager] Phoenix init error ({e}); skipping.", flush=True)

    def _init_langfuse(self) -> None:
        """Create Langfuse LangChain callback handler."""
        if not (cfg.langfuse_secret_key and cfg.langfuse_public_key):
            print("[ObservabilityManager] Langfuse keys not set; skipping.", flush=True)
            return
        try:
            # Try v2 import path first, fall back to v3
            try:
                from langfuse.langchain import CallbackHandler
            except ImportError:
                from langfuse import CallbackHandler  # type: ignore[no-redef]

            try:
                self._langfuse_handler = CallbackHandler(
                    secret_key=cfg.langfuse_secret_key,
                    public_key=cfg.langfuse_public_key,
                    host=cfg.langfuse_host,
                )
            except TypeError:
                # Newer Langfuse reads credentials from env vars directly
                os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
                os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
                os.environ["LANGFUSE_HOST"] = cfg.langfuse_host
                self._langfuse_handler = CallbackHandler()
            self._langfuse_available = True
            print("[ObservabilityManager] Langfuse callback handler created.", flush=True)
        except ImportError as e:
            print(f"[ObservabilityManager] Langfuse not available ({e}); skipping.", flush=True)
        except Exception as e:
            print(f"[ObservabilityManager] Langfuse init error ({e}); skipping.", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_callbacks(self) -> List[Any]:
        """Return callbacks to inject into LLM invocations."""
        callbacks = []
        if self._langfuse_handler:
            callbacks.append(self._langfuse_handler)
        return callbacks

    def wrap_node(self, node_name: str, fn: Callable) -> Callable:
        """
        Wrap a LangGraph node function with an OTEL span.

        - Records node.name, node.status, node.latency_ms
        - Pops _token_usage from the result dict (consumed here only)
        - For "Retrieve" nodes: emits retrieval.context_docs span event;
          if poisoned docs present, emits retrieval.poisoned_docs_detected
        """
        obs = self

        def wrapped(state: Dict) -> Dict:
            tracer = obs._tracer  # read at call time so async init is visible
            start_time = time.time()
            print(f"[Pipeline] {node_name}...", flush=True)

            if tracer:
                with tracer.start_as_current_span(f"node.{node_name}") as span:
                    span.set_attribute("node.name", node_name)
                    try:
                        result = fn(state)
                        latency_ms = (time.time() - start_time) * 1000
                        span.set_attribute("node.status", "success")
                        span.set_attribute("node.latency_ms", round(latency_ms, 2))

                        # Consume token usage — never reaches AgentState
                        token_usage = result.pop("_token_usage", {})
                        if token_usage:
                            span.set_attribute("llm.usage.prompt_tokens", token_usage.get("prompt_tokens", 0))
                            span.set_attribute("llm.usage.completion_tokens", token_usage.get("completion_tokens", 0))
                            span.set_attribute("llm.usage.total_tokens", token_usage.get("total_tokens", 0))

                        # Retrieval-specific events
                        if node_name == "Retrieve":
                            context_docs = result.get("context_docs", [])
                            docs_payload = [
                                {
                                    "page_content": d.page_content[:500],
                                    "metadata": d.metadata,
                                }
                                for d in context_docs
                            ]
                            span.add_event(
                                "retrieval.context_docs",
                                attributes={"docs": json.dumps(docs_payload, default=str)},
                            )
                            poisoned = [d for d in context_docs if d.metadata.get("is_poisoned")]
                            if poisoned:
                                span.add_event(
                                    "retrieval.poisoned_docs_detected",
                                    attributes={
                                        "count": len(poisoned),
                                        "ids": json.dumps(
                                            [d.metadata.get("source_id", "") for d in poisoned],
                                            default=str,
                                        ),
                                    },
                                )

                        print(f"[Pipeline] {node_name} done ({latency_ms:.0f}ms)", flush=True)
                        return result
                    except Exception as exc:
                        latency_ms = (time.time() - start_time) * 1000
                        span.set_attribute("node.status", "error")
                        span.set_attribute("node.latency_ms", round(latency_ms, 2))
                        span.record_exception(exc)
                        print(f"[Pipeline] {node_name} ERROR ({latency_ms:.0f}ms)", flush=True)
                        raise
            else:
                # No OTEL — still pop _token_usage so it never reaches AgentState
                result = fn(state)
                result.pop("_token_usage", None)
                latency_ms = (time.time() - start_time) * 1000
                print(f"[Pipeline] {node_name} done ({latency_ms:.0f}ms)", flush=True)
                return result

        wrapped.__name__ = fn.__name__ if hasattr(fn, "__name__") else node_name
        return wrapped

    def shutdown(self) -> None:
        """Flush and shut down all observability backends."""
        if self._tracer_provider:
            try:
                self._tracer_provider.shutdown()
                print("[ObservabilityManager] Phoenix TracerProvider flushed.")
            except Exception as e:
                print(f"[ObservabilityManager] Phoenix shutdown error: {e}")

        if self._langfuse_handler:
            try:
                if hasattr(self._langfuse_handler, "flush"):
                    self._langfuse_handler.flush()
                elif hasattr(self._langfuse_handler, "langfuse") and hasattr(self._langfuse_handler.langfuse, "flush"):
                    self._langfuse_handler.langfuse.flush()
                print("[ObservabilityManager] Langfuse handler flushed.")
            except Exception as e:
                print(f"[ObservabilityManager] Langfuse flush error: {e}")
