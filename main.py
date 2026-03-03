"""
main.py — CLI entry point for the LLM Security RAG Pipeline.

Commands:
  ingest   — Download and ingest MITRE ATT&CK into ChromaDB
  query    — Run a single query through the full RAG pipeline
  simulate — Run adversarial attack simulations with judge evaluation
"""

import sys

import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


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
        obs, graph = _startup()

        console.rule("[bold blue]RAG Query")
        console.print(f"[dim]Query:[/dim] {query_text}\n")

        initial_state = {
            "query": query_text,
            "context_docs": [],
            "llm_response": "",
            "security_score": 1.0,
            "metadata": {"attack_type": attack_type or "none"},
        }

        final_state = graph.invoke(initial_state)

        security_score = final_state.get("security_score", 0.0)
        flags = final_state.get("metadata", {}).get("guardrail_flags", [])
        llm_response = final_state.get("llm_response", "")
        context_analysis = final_state.get("metadata", {}).get("context_analysis", {})
        original_query = final_state.get("metadata", {}).get("original_query", query_text)
        rewritten_query = final_state.get("query", query_text)

        # Security score display
        score_color = "green" if security_score >= 0.7 else "red"
        console.print(f"[bold]Security Score:[/bold] [{score_color}]{security_score:.4f}[/{score_color}]")
        console.print(f"[bold]Guardrail Flags:[/bold] {', '.join(flags) if flags else '[dim]none[/dim]'}")

        # Query rewriting info
        if original_query != rewritten_query:
            console.print(f"\n[dim]Original query:[/dim] {original_query}")
            console.print(f"[dim]Rewritten query:[/dim] {rewritten_query}")

        # Context analysis
        if context_analysis:
            console.print(
                f"\n[bold]Context Analysis:[/bold] "
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


@cli.command()
@click.option(
    "--attack",
    default="all",
    help="Attack name to run, or 'all' to run every registered attack.",
)
def simulate(attack: str):
    """Run adversarial attack simulations with judge-LLM evaluation."""
    from simulate_attacks import ATTACK_REGISTRY, AttackResult

    obs, graph = None, None
    try:
        obs, graph = _startup()

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

            if result.error:
                console.print(f"  [red]ERROR:[/red] {result.error[:200]}")
            else:
                blocked_str = "[green]BLOCKED[/green]" if result.success else "[red]NOT BLOCKED[/red]"
                console.print(f"  Score: {result.security_score:.4f} | {blocked_str}")
                console.print(f"  Flags: {', '.join(result.guardrail_flags) or 'none'}")
                console.print(f"  Response: {result.final_state.get('llm_response', '')}")
                judge = result.judge_verdict
                console.print(
                    f"  Judge: detected={judge.get('attack_detected')}, "
                    f"source={judge.get('source_of_failure')}, "
                    f"confidence={judge.get('confidence', 0.0):.2f}"
                )

        # Summary table
        console.print()
        table = Table(title="Attack Simulation Summary", box=box.SIMPLE)
        table.add_column("Attack", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Blocked?", justify="center")
        table.add_column("Judge Detected", justify="center")
        table.add_column("Source", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Flags", style="dim")

        for r in results:
            judge = r.judge_verdict
            table.add_row(
                r.attack_name,
                f"{r.security_score:.4f}" if not r.error else "ERROR",
                "[green]YES[/green]" if r.success else "[red]NO[/red]",
                "[red]YES[/red]" if judge.get("attack_detected") else "[green]no[/green]",
                str(judge.get("source_of_failure", "?")),
                f"{judge.get('confidence', 0.0):.2f}",
                ", ".join(r.guardrail_flags) if r.guardrail_flags else "none",
            )

        console.print(table)

        blocked_count = sum(1 for r in results if r.success)
        console.print(
            f"\n[bold]Summary:[/bold] {blocked_count}/{len(results)} attacks blocked "
            f"({'100%' if len(results) == 0 else f'{blocked_count/len(results)*100:.0f}%'})"
        )

    finally:
        if obs:
            obs.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
