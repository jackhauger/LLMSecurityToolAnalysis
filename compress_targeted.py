from __future__ import annotations
import json
import math

from compress import serialize


def _msg_key(msg) -> str:
    if isinstance(msg, dict):
        stripped = {k: v for k, v in msg.items() if k != "id"}
        return json.dumps(stripped, sort_keys=True, default=str)
    return json.dumps(msg, sort_keys=True, default=str)


def _suffix_after(messages: list, baseline: list) -> list:
    if not isinstance(messages, list) or not isinstance(baseline, list):
        return messages
    if not baseline:
        return list(messages)
    if len(baseline) > len(messages):
        return list(messages)
    for i, b in enumerate(baseline):
        if _msg_key(messages[i]) != _msg_key(b):
            return list(messages)
    return list(messages[len(baseline):])


_LANGSMITH_DROP = frozenset({
    "total_tokens", "prompt_tokens", "completion_tokens",
    "status", "start_time", "end_time",
})

_LANGFUSE_TRACE_DROP = frozenset({
    "sessionId", "tags", "public", "environment", "htmlPath",
    "totalCost", "bookmarked", "updatedAt", "createdAt", "projectId",
    "release", "version", "userId", "externalId", "id", "timestamp",
    "metadata",
})

_LANGFUSE_OBS_KEEP = frozenset({
    "name", "type", "model", "input", "output",
    "parentObservationId",
})

_PHOENIX_DROP = frozenset({
    "context.span_id", "context.trace_id",
    "attributes.input.mime_type", "attributes.output.mime_type",
    "attributes.metadata", "attributes.session.id",
    "attributes.llm.invocation_parameters",
    "attributes.openinference.span.kind",
    "attributes.llm.token_count.prompt",
    "attributes.llm.token_count.completion",
    "attributes.llm.token_count.total",
    "attributes.llm.token_count.completion_details.reasoning",
    "attributes.llm.token_count.prompt_details.cache_read",
    "attributes.llm.provider",
    "attributes.llm.system",
    "events",
    "status_code", "status_message",
    "start_time", "end_time",
})

_PHOENIX_RENAME = {
    "attributes.input.value":           "input",
    "attributes.output.value":          "output",
    "attributes.llm.input_messages":    "input_messages",
    "attributes.llm.output_messages":   "output_messages",
    "attributes.llm.model_name":        "model",
    "attributes.llm.function_call":     "function_call",
    "attributes.tool.name":             "tool_name",
    "attributes.tool.description":      "tool_description",
    "attributes.llm.tools":             "tools",
}


def _is_nan(v) -> bool:
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def _dedupe_lc_run(run: dict, baseline: list) -> tuple[dict, list]:
    cleaned = {k: v for k, v in run.items() if k not in _LANGSMITH_DROP}

    in_state = baseline
    inputs = cleaned.get("inputs")
    if isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
        full_in = inputs["messages"]
        in_state = full_in
        delta = _suffix_after(full_in, baseline)
        new_inputs = {k: v for k, v in inputs.items() if k != "messages"}
        if delta:
            new_inputs["messages"] = delta
        cleaned["inputs"] = new_inputs

    out_state = in_state
    outputs = cleaned.get("outputs")
    if isinstance(outputs, dict) and isinstance(outputs.get("messages"), list):
        full_out = outputs["messages"]
        out_state = full_out
        delta = _suffix_after(full_out, in_state)
        new_outputs = {k: v for k, v in outputs.items() if k != "messages"}
        if delta:
            new_outputs["messages"] = delta
        cleaned["outputs"] = new_outputs

    if isinstance(cleaned.get("child_runs"), list):
        running = in_state
        new_children = []
        for child in cleaned["child_runs"]:
            cleaned_child, running = _dedupe_lc_run(child, running)
            new_children.append(cleaned_child)
        cleaned["child_runs"] = new_children

    return cleaned, out_state


def _clean_langsmith(node: dict) -> dict:
    cleaned = {k: v for k, v in node.items() if k not in _LANGSMITH_DROP}
    if isinstance(cleaned.get("child_runs"), list):
        baseline = []
        inputs = cleaned.get("inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
            baseline = inputs["messages"]
        running = baseline
        new_children = []
        for child in cleaned["child_runs"]:
            cleaned_child, running = _dedupe_lc_run(child, running)
            new_children.append(cleaned_child)
        cleaned["child_runs"] = new_children
    return cleaned


def _clean_langfuse(trace: dict) -> dict:
    result = {k: v for k, v in trace.items()
              if k not in _LANGFUSE_TRACE_DROP and v not in (None, "", [], {})}

    raw_obs = trace.get("observations", [])
    if not raw_obs:
        return result

    obs_ids = {o.get("id") for o in raw_obs}
    _SORT_KEYS = {"parentObservationId", "startTime"}
    cleaned_obs: dict[str, dict] = {}
    for obs in raw_obs:
        obs_id = obs.get("id")
        cleaned = {k: v for k, v in obs.items()
                   if k in (_LANGFUSE_OBS_KEEP | _SORT_KEYS)
                   and v not in (None, "", [], {})}
        if obs_id:
            cleaned_obs[obs_id] = cleaned

    child_map: dict[str | None, list[dict]] = {}
    for raw_obs_item in raw_obs:
        obs_id = raw_obs_item.get("id")
        parent_id = raw_obs_item.get("parentObservationId") or None
        if parent_id not in obs_ids:
            parent_id = None
        if obs_id in cleaned_obs:
            child_map.setdefault(parent_id, []).append(cleaned_obs[obs_id])
    for siblings in child_map.values():
        siblings.sort(key=lambda o: str(o.get("startTime", "")))

    def _build_tree(obs_id: str | None) -> list[dict]:
        nodes = []
        for obs in child_map.get(obs_id, []):
            raw_id = next(
                (r["id"] for r in raw_obs if r.get("id") in cleaned_obs
                 and cleaned_obs[r["id"]] is obs),
                None,
            )
            kids = _build_tree(raw_id)
            if kids:
                obs["children"] = kids
            obs.pop("parentObservationId", None)
            obs.pop("startTime", None)
            nodes.append(obs)
        return nodes

    obs_list = _build_tree(None) or list(cleaned_obs.values())

    seen: set[str] = set()
    trace_input = result.get("input")
    if isinstance(trace_input, dict):
        for m in trace_input.get("messages", []):
            seen.add(_msg_key(m))

    result["observations"] = [_dedupe_lf_obs(obs, seen) for obs in obs_list]
    return result


def _strip_system_messages(role_list: list) -> list:
    return [m for m in role_list if not (isinstance(m, dict) and m.get("role") == "system")]


def _dedupe_lf_obs(obs: dict, seen: set[str]) -> dict:
    obs = dict(obs)

    inp = obs.get("input")
    if isinstance(inp, dict) and isinstance(inp.get("messages"), list):
        new_msgs = []
        for m in inp["messages"]:
            k = _msg_key(m)
            if k not in seen:
                seen.add(k)
                new_msgs.append(m)
        new_inp = {key: v for key, v in inp.items() if key != "messages"}
        if new_msgs:
            new_inp["messages"] = new_msgs
        obs["input"] = new_inp
    elif isinstance(inp, list) and any(isinstance(m, dict) and "role" in m for m in inp):
        obs["input"] = _strip_system_messages(inp)

    out = obs.get("output")
    if isinstance(out, dict) and isinstance(out.get("messages"), list):
        new_msgs = []
        for m in out["messages"]:
            k = _msg_key(m)
            if k not in seen:
                seen.add(k)
                new_msgs.append(m)
        new_out = {key: v for key, v in out.items() if key != "messages"}
        if new_msgs:
            new_out["messages"] = new_msgs
        obs["output"] = new_out

    if isinstance(obs.get("children"), list):
        obs["children"] = [_dedupe_lf_obs(kid, seen) for kid in obs["children"]]

    return obs


def _clean_phoenix(spans: list[dict]) -> list[dict]:
    all_span_ids = {s.get("context.span_id") for s in spans}

    cleaned = []
    for span in spans:
        span_id = span.get("context.span_id")
        parent_id = span.get("parent_id")
        is_root = not parent_id or parent_id not in all_span_ids
        is_chain = span.get("span_kind") == "CHAIN"

        node: dict = {}
        for k, v in span.items():
            if k in _PHOENIX_DROP:
                continue
            if _is_nan(v) or v is None or v == "" or v == [] or v == {}:
                continue
            if k in _PHOENIX_RENAME:
                renamed = _PHOENIX_RENAME[k]
                if is_chain and not is_root and renamed in ("input", "output"):
                    continue
                val = v
                if renamed == "input_messages" and isinstance(v, list):
                    val = [m for m in v if not (isinstance(m, dict) and m.get("message.role") == "system")]
                node[renamed] = val
            elif k.startswith("attributes."):
                continue
            else:
                node[k] = v
        if node:
            cleaned.append(node)
    return cleaned


def compress_targeted(traces, backend: str) -> str:
    backend_key = backend.lower()
    if backend_key == "langsmith":
        cleaned = _clean_langsmith(traces)
    elif backend_key == "langfuse":
        cleaned = _clean_langfuse(traces)
    elif backend_key in ("arize phoenix", "phoenix"):
        cleaned = _clean_phoenix(traces)
    else:
        cleaned = traces
    return serialize(cleaned)
