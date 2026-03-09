"""
main.py — CLI entry point for the LLM Security RAG Pipeline.

Commands:
  ingest   — Download and ingest MITRE ATT&CK into ChromaDB
  query    — Run a single query through the full RAG pipeline
  simulate — Run adversarial attack simulations with judge evaluation
"""

import json
import sys
import uuid
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def _check_phoenix() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:6006", timeout=2)
        return r.status_code < 500
    except Exception:
        return False


def _startup() -> tuple:
    """
    Common startup sequence: validate config, init observability, build graph.
    Returns (obs, graph).
    """
    from config import cfg
    from observers import ObservabilityManager
    from graph import build_graph

    cfg.validate()
    obs = ObservabilityManager()
    print("[Startup] Building graph (loading ChromaDB + LangGraph)...", flush=True)
    graph = build_graph(obs)
    return obs, graph


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """LLM Security RAG Pipeline — forensic analysis of adversarial ML attacks."""
    pass


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--force", is_flag=True, default=False, help="Re-ingest even if collection already populated.")
def ingest(force: bool):
    """Download and ingest MITRE ATT&CK techniques into ChromaDB."""
    from config import cfg

    cfg.validate()

    console.rule("[bold blue]MITRE ATT&CK Ingestion")
    import database

    count = database.ingest_mitre_attack(force=force)
    console.print(f"\n[green]Done.[/green] ChromaDB collection '[bold]{cfg.chroma_collection_name}[/bold]' "
                  f"has [bold]{count}[/bold] documents.")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query_text")
@click.option("--attack-type", default=None, help="Label to tag this query in metadata (for tracing).")
def query(query_text: str, attack_type: str):
    """Run a single query through the full RAG pipeline."""
    obs, graph = None, None
    try:
        print("Starting up (first run may take ~30s for dependency loading)...", flush=True)
        obs, graph = _startup()

        console.rule("[bold blue]RAG Query")
        console.print(f"[dim]Query:[/dim] {query_text}\n")

        initial_state = {
            "query": query_text,
            "context_docs": [],
            "llm_response": "",
            "metadata": {"attack_type": attack_type or "none"},
        }

        run_id = str(uuid.uuid4())
        print(f"[Query] Starting pipeline (this may take 60-90s)...", flush=True)
        final_state = graph.invoke(initial_state, config={"run_id": run_id})
        print(f"[Query] Pipeline complete. Run ID: {run_id[:8]}...", flush=True)

        llm_response = final_state.get("llm_response", "")
        context_analysis = final_state.get("metadata", {}).get("context_analysis", {})
        original_query = final_state.get("metadata", {}).get("original_query", query_text)
        rewritten_query = final_state.get("query", query_text)

        # Query rewriting info
        if original_query != rewritten_query:
            console.print(f"[dim]Original query:[/dim] {original_query}")
            console.print(f"[dim]Rewritten query:[/dim] {rewritten_query}\n")

        # Context analysis
        if context_analysis:
            console.print(
                f"[bold]Context Analysis:[/bold] "
                f"poisoned={context_analysis.get('poisoned_doc_count', 0)}, "
                f"avg_relevance={context_analysis.get('avg_relevance_score', 0.0):.4f}, "
                f"diversity={context_analysis.get('source_diversity_ratio', 0.0):.4f}"
            )

        # Response
        console.print("\n[bold]Response:[/bold]")
        console.print(llm_response)

        # Retrieved docs table
        context_docs = final_state.get("context_docs", [])
        if context_docs:
            console.print()
            table = Table(title="Retrieved Context Documents", box=box.SIMPLE)
            table.add_column("Source ID", style="cyan", no_wrap=True)
            table.add_column("Relevance", justify="right")
            table.add_column("Poisoned?", justify="center")
            table.add_column("Chunk Type", style="dim")
            table.add_column("Preview", max_width=60)

            for doc in context_docs:
                m = doc.metadata
                is_poisoned = m.get("is_poisoned", False)
                table.add_row(
                    str(m.get("source_id", "")),
                    f"{m.get('relevance_score', 0.0):.4f}",
                    "[red]YES[/red]" if is_poisoned else "[green]no[/green]",
                    str(m.get("chunk_type", "")),
                    doc.page_content[:80].replace("\n", " "),
                )
            console.print(table)

    finally:
        if obs:
            obs.shutdown()


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def _write_attack_log(result, log_path: str) -> None:
    """Append one JSON line per attack result to the log file."""
    context_docs_summary = [
        {
            "page_content": doc.page_content[:500],
            "metadata": doc.metadata,
        }
        for doc in result.final_state.get("context_docs", [])
    ]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attack_name": result.attack_name,
        "run_id": result.run_id,
        "query_used": result.query_used,
        "attack_detectable": result.attack_detectable,
        "judge_verdict": result.judge_verdict,
        "traces": result.traces,
        "final_state": {
            "query": result.final_state.get("query", ""),
            "llm_response": result.final_state.get("llm_response", ""),
            "metadata": result.final_state.get("metadata", {}),
            "context_docs": context_docs_summary,
        },
        "error": result.error,
        "poisoned_doc_id": result.poisoned_doc_id,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


@cli.command()
@click.option(
    "--attack",
    default="all",
    help="Attack name to run, or 'all' to run every registered attack.",
)
def simulate(attack: str):
    """Run adversarial attack simulations with judge-LLM evaluation from real traces."""
    from simulate_attacks import ATTACK_REGISTRY, AttackResult
    from config import cfg

    obs, graph = None, None
    try:
        print("Starting up (first run may take ~30s for dependency loading)...", flush=True)
        obs, graph = _startup()

        if not _check_phoenix():
            console.print(
                "[yellow]Warning:[/yellow] Phoenix not reachable at localhost:6006. "
                "Start it with: phoenix serve\nPhoenix spans will be missing from judge traces."
            )

        if attack == "all":
            attack_names = list(ATTACK_REGISTRY.keys())
        elif attack in ATTACK_REGISTRY:
            attack_names = [attack]
        else:
            console.print(
                f"[red]Unknown attack:[/red] '{attack}'. "
                f"Available: {', '.join(ATTACK_REGISTRY.keys())}"
            )
            sys.exit(1)

        console.rule("[bold red]Attack Simulation")
        results: list[AttackResult] = []

        for name in attack_names:
            console.print(f"\n[bold yellow]Running:[/bold yellow] {name} ...")
            attack_fn = ATTACK_REGISTRY[name]
            result = attack_fn(graph, obs)
            results.append(result)
            _write_attack_log(result, cfg.log_file)

            if result.error:
                console.print(f"  [red]ERROR:[/red] {result.error[:200]}")
            else:
                detectable_str = (
                    "[red]DETECTABLE[/red]" if result.attack_detectable else "[green]not detected[/green]"
                )
                console.print(f"  Run ID: {result.run_id}")
                console.print(f"  Attack: {detectable_str}")
                judge = result.judge_verdict
                console.print(f"  Evidence: {judge.get('evidence', '')[:200]}")
                console.print(
                    f"  Judge: confidence={judge.get('confidence', 0.0):.2f} | "
                    f"{judge.get('reasoning', '')[:200]}"
                )

        # Summary table
        console.print()
        table = Table(title="Attack Simulation Summary", box=box.SIMPLE)
        table.add_column("Attack", style="cyan")
        table.add_column("Run ID", style="dim")
        table.add_column("Detectable?", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Evidence", max_width=50)

        for r in results:
            judge = r.judge_verdict
            table.add_row(
                r.attack_name,
                r.run_id[:8] + "..." if r.run_id else "ERROR",
                "[red]YES[/red]" if r.attack_detectable else "[green]no[/green]",
                f"{judge.get('confidence', 0.0):.2f}",
                str(judge.get("evidence", ""))[:50],
            )

        console.print(table)

        detectable_count = sum(1 for r in results if r.attack_detectable)
        console.print(
            f"\n[bold]Summary:[/bold] {detectable_count}/{len(results)} attacks detectable from traces"
        )
        console.print(f"[dim]Log:[/dim] {cfg.log_file}")

    finally:
        if obs:
            obs.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
