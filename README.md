# LLM Security RAG Pipeline

Forensic analysis of adversarial attacks against an LLM-powered RAG system, using MITRE ATT&CK as the knowledge base. Instruments three observability backends simultaneously (LangSmith, Arize Phoenix, Langfuse) and includes a built-in attack simulation loop with judge-LLM evaluation.

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

The **SecurityGuardrail** node scores every response from 1.0 downward based on detected signals (prompt injection markers, exfiltration URLs, poisoned context, excessive length). Responses scoring below 0.7 are blocked.

### Simulated attacks

| Attack | Technique |
|---|---|
| `indirect_prompt_injection` | Poisons ChromaDB with an instruction-hijacking document |
| `pii_exfiltration` | Crafts a query to elicit markdown image exfiltration to an attacker URL |
| `dos_token_exhaustion` | Requests exhaustive output to trigger the response-length guardrail |

After each attack, a standalone judge LLM (no tracing) evaluates the run logs and returns a structured verdict.

## Observability

All three backends are optional except for the pipeline itself. Configure keys in `.env` — any backend without credentials is skipped silently.

- **LangSmith** — env var driven (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`)
- **Arize Phoenix** — launches local UI at `http://localhost:6006`, exports traces via OTLP gRPC
    - python -m phoenix.server.main serve
- **Langfuse** — callback handler injected into LLM calls (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`)
