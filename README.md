# LLM Security RAG Pipeline

Forensic analysis of adversarial attacks against an LLM-powered RAG system, using MITRE ATT&CK as the knowledge base. Instruments three observability backends simultaneously (LangSmith, Arize Phoenix, Langfuse) and includes a built-in attack simulation loop with judge-LLM evaluation from real traces.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in GOOGLE_API_KEY — only required key
```

## Usage

```bash
# One-time: ingest MITRE ATT&CK into ChromaDB
python main.py ingest

# Query the pipeline
python main.py query "How does MITRE T1059 work?"

# Run all attack simulations
python main.py simulate

# Run a specific attack
python main.py simulate --attack indirect_prompt_injection
```

## How it works

Each query passes through a LangGraph pipeline:

```
RewriteQuery → Retrieve → AnalyzeContext → Generate → SecurityGuardrail
```

The **SecurityGuardrail** node is a passthrough — it does not score or block responses. Attack detection is entirely trace-based: after each simulated attack, the pipeline fetches real observability traces from LangSmith and Phoenix, then sends them to a standalone judge LLM for forensic analysis.

### Simulated attacks

| Attack | Technique | Trace evidence |
|---|---|---|
| `indirect_prompt_injection` | Poisons ChromaDB with an instruction-hijacking document | LLM inputs/outputs in child runs reveal instruction leakage |
| `pii_exfiltration` | Crafts a query to elicit markdown image exfiltration to an attacker URL | Generate node output contains exfiltration URL pattern |
| `dos_token_exhaustion` | Requests exhaustive output to trigger excessive token usage | Token counts in LangSmith trace span show anomalous usage |

After each attack, a standalone judge LLM (no tracing callbacks) analyzes the actual trace data from LangSmith and Phoenix and returns a structured verdict: `attack_detectable`, `evidence`, `confidence`, `reasoning`.

## Observability

All three backends are optional except for the pipeline itself. Configure keys in `.env` — any backend without credentials is skipped silently.

- **LangSmith** — env var driven (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`)
- **Arize Phoenix** — launches local UI at `http://localhost:6006`, exports traces via OTLP gRPC
    - `python -m phoenix.server.main serve`
- **Langfuse** — callback handler injected into LLM calls (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`)

### Verifying traces

After running a query or simulation, verify all three backends received traces:

```bash
# LangSmith — list recent traces
langsmith trace list --last-n-minutes 5 --project llm-security-rag

# Langfuse
langfuse api traces list --limit 5

# Phoenix — open UI directly
# http://localhost:6006
```
