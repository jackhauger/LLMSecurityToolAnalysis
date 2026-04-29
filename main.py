import json
import time
import uuid
import click
import charts
from compress import serialize as compress_serialize
import database
import observers

from dataclasses import asdict
from pathlib import Path
from config import cfg
from judge import DetectionJudge, RootCauseJudge
from langchain_core.messages import HumanMessage
from rich import box
from rich.console import Console
from rich.table import Table

from attack_dataset import ATTACK_DATASET

console = Console()


@click.group()
def cli():
    """LLM Security RAG Pipeline — forensic analysis of adversarial ML attacks."""
    pass


@cli.command()
@click.option("--force", is_flag=True, default=False, help="Re-ingest even if collection already populated.")
def ingest(force: bool):
    cfg.validate()

    console.rule("[bold blue]MITRE ATT&CK Ingestion")
    collection = database.get_or_create_collection()
    count = database.ingest_mitre_attack(collection, force=force)
    console.print(
        f"\n[green]Done.[/green] ChromaDB collection '[bold]{cfg.chroma_collection_name}[/bold]' "
        f"has [bold]{count}[/bold] documents."
    )


BACKEND_FACTORIES = {
    "langsmith": "create_langsmith_pipeline",
    "langfuse": "create_langfuse_pipeline",
    "arize phoenix": "create_phoenix_pipeline",
}

DEFAULT_BENCHMARK_DATASET = Path("rag_observability_benchmark_mitre_attack_dataset.json")


@cli.command()
@click.option(
    "--dataset",
    default=None,
    type=click.Path(exists=True),
    help="Path to a JSON file containing a list of test case entries. "
         "Each entry must have: input_prompt, attack_type, benign, poisoned_document. "
         "Defaults to rag_observability_benchmark_mitre_attack_dataset.json when present, "
         "otherwise the hardcoded ATTACK_DATASET.",
)
@click.option(
    "--judge-trace-format",
    type=click.Choice(["raw", "compressed"]),
    default="raw",
    show_default=True,
    help="Format of trace text given to the judge.",
)
def benchmark(dataset: str | None, judge_trace_format: str):
    """Run attack dataset through isolated per-backend pipelines with judge evaluation."""
    cfg.validate()
    detection_judge = DetectionJudge()
    root_cause_judge = RootCauseJudge()

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    judge_logs_dir = results_dir / "judge_inputs"
    judge_logs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = results_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    collection = database.get_or_create_collection()

    if dataset:
        with open(dataset) as f:
            test_cases = json.load(f)
        console.print(f"[dim]Loaded {len(test_cases)} test case(s) from {dataset}[/dim]")
    elif DEFAULT_BENCHMARK_DATASET.exists():
        with open(DEFAULT_BENCHMARK_DATASET) as f:
            test_cases = json.load(f)
        console.print(f"[dim]Loaded {len(test_cases)} test case(s) from {DEFAULT_BENCHMARK_DATASET}[/dim]")
    else:
        test_cases = ATTACK_DATASET

    console.rule("[bold red]Benchmark")
    all_results = []
    pending_fetches = []
    benchmark_started = time.perf_counter()
    for i, test_case in enumerate(test_cases):
        test_case_id = f"TC-{i + 1:03d}"
        dataset_case_id = test_case.get("id", "")
        input_prompt = test_case["input_prompt"]
        attack_type = test_case["attack_type"]
        benign = test_case["benign"]
        poisoned_document = test_case["poisoned_document"]
        difficulty = test_case.get("difficulty")
        notes = test_case.get("notes")

        console.print(f"\n[bold yellow]{test_case_id}[/bold yellow]: "
                      f"{'benign' if benign else attack_type} — {input_prompt[:80]}...")

        doc_id = f"poison-{dataset_case_id or test_case_id}"
        injected = False

        try:
            if poisoned_document is not None:
                database.inject_poisoned_document(
                    collection, doc_id, poisoned_document,
                    {"source_id": dataset_case_id or f"reference-{uuid.uuid4().hex[:8]}"},
                )
                injected = True
                console.print(f"  [dim]Injected poisoned doc: {doc_id}[/dim]")

            backends_result = {}
            for backend_name, factory_name in BACKEND_FACTORIES.items():
                factory_fn = getattr(observers, factory_name)
                console.print(f"  [{backend_name}] Creating pipeline...", end="")

                ctx = factory_fn(collection, test_case_id, attack_type)
                console.print(" invoking...", end="")
                try:
                    output = ctx.graph.invoke(
                        {"messages": [HumanMessage(content=input_prompt)]},
                        config=ctx.invoke_config,
                    )
                finally:
                    ctx.cleanup()

                final_response = ""
                if output and "messages" in output and output["messages"]:
                    raw_content = output["messages"][-1].content
                    if isinstance(raw_content, list):
                        final_response = " ".join(
                            b.get("text", "") for b in raw_content if isinstance(b, dict)
                        )
                    else:
                        final_response = str(raw_content)

                backends_result[backend_name] = {
                    "run_id": ctx.run_id,
                    "final_response": final_response[:1000],
                }
                pending_fetches.append((test_case_id, benign, attack_type, backend_name, ctx.fetch_traces, backends_result))
                console.print(" queued for trace fetch")

            result_record = {
                "test_case_id": test_case_id,
                "dataset_case_id": dataset_case_id,
                "input_prompt": input_prompt,
                "attack_type": attack_type,
                "difficulty": difficulty,
                "benign": benign,
                "poisoned_document": poisoned_document is not None,
                "notes": notes,
                "backends": backends_result,
            }
            result_path = results_dir / f"{test_case_id}.json"
            with open(result_path, "w") as f:
                json.dump(result_record, f, indent=2, default=str)

            all_results.append(result_record)

        finally:
            if injected:
                database.remove_poisoned_document(collection, doc_id)
                console.print(f"  [dim]Cleaned up poisoned doc: {doc_id}[/dim]")

    console.rule("[bold blue]Fetch And Judge")
    for test_case_id, benign, attack_type, backend_name, fetch_traces, backends_result in pending_fetches:
        console.print(f"  [{backend_name}] {test_case_id} fetching traces...", end="")
        traces = fetch_traces()
        backend_trace_dir = traces_dir / test_case_id
        backend_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = backend_trace_dir / f"{backend_name}.json"
        trace_path.write_text(json.dumps(traces, indent=2, default=str))

        console.print(" judging...", end="")
        if judge_trace_format == "compressed":
            judge_trace_text = compress_serialize(traces)
        else:
            judge_trace_text = detection_judge.build_trace_text(traces)
        verdict, judge_prompt, judge_raw_response = detection_judge.evaluate_trace(
            judge_trace_text,
            backend_name,
            backends_result[backend_name]["final_response"],
        )
        root_cause_verdict = None
        root_cause_prompt = ""
        root_cause_raw_response = ""
        if verdict.suspicious_evidence_present or verdict.attack_success_observed:
            root_cause_verdict, root_cause_prompt, root_cause_raw_response = root_cause_judge.evaluate_trace(
                judge_trace_text,
                backend_name,
                attack_type or "benign",
                backends_result[backend_name]["final_response"],
            )

        detected_str = (
            "[red]DETECTED[/red]" if verdict.suspicious_evidence_present
            else "[green]clean[/green]"
        )
        console.print(
            f" {detected_str} "
            f"(conf={verdict.confidence:.2f})"
        )

        ground_truth_attack_present = benign == 0
        if ground_truth_attack_present and verdict.suspicious_evidence_present:
            confusion_outcome = "true_positive"
        elif ground_truth_attack_present and not verdict.suspicious_evidence_present:
            confusion_outcome = "false_negative"
        elif not ground_truth_attack_present and verdict.suspicious_evidence_present:
            confusion_outcome = "false_positive"
        else:
            confusion_outcome = "true_negative"

        backends_result[backend_name]["trace_file"] = str(trace_path)
        backends_result[backend_name]["judge_trace_format"] = judge_trace_format
        backends_result[backend_name]["judge_trace_text"] = judge_trace_text
        backends_result[backend_name]["judge_prompt"] = judge_prompt
        backends_result[backend_name]["judge_raw_response"] = judge_raw_response
        backends_result[backend_name]["judge_verdict"] = asdict(verdict)
        if root_cause_verdict is not None:
            backends_result[backend_name]["root_cause_prompt"] = root_cause_prompt
            backends_result[backend_name]["root_cause_raw_response"] = root_cause_raw_response
            backends_result[backend_name]["root_cause_verdict"] = asdict(root_cause_verdict)
        backends_result[backend_name]["evaluation_metrics"] = {
            "ground_truth_attack_present": ground_truth_attack_present,
            "detection_correct": confusion_outcome in ("true_positive", "true_negative"),
            "confusion_outcome": confusion_outcome,
        }

        backend_judge_dir = judge_logs_dir / test_case_id / backend_name
        backend_judge_dir.mkdir(parents=True, exist_ok=True)
        (backend_judge_dir / "judge_trace.txt").write_text(judge_trace_text)
        if judge_trace_format == "compressed":
            (backend_judge_dir / "judge_trace_compressed.txt").write_text(judge_trace_text)
        (backend_judge_dir / "judge_prompt.txt").write_text(judge_prompt)
        (backend_judge_dir / "judge_response.json").write_text(judge_raw_response)
        if root_cause_verdict is not None:
            (backend_judge_dir / "root_cause_prompt.txt").write_text(root_cause_prompt)
            (backend_judge_dir / "root_cause_response.json").write_text(root_cause_raw_response)

        result_record = next(r for r in all_results if r["test_case_id"] == test_case_id)
        result_path = results_dir / f"{test_case_id}.json"
        with open(result_path, "w") as f:
            json.dump(result_record, f, indent=2, default=str)

    observers.shutdown_phoenix()

    console.print()
    table = Table(title="Benchmark Summary", box=box.SIMPLE)
    table.add_column("Test Case", style="cyan")
    table.add_column("Attack Type", style="dim")
    table.add_column("Benign?", justify="center")
    table.add_column("Backend", style="dim")
    table.add_column("Attack Detected?", justify="center")
    table.add_column("Attack Success?", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasoning", max_width=50)

    for r in all_results:
        for backend_name in BACKEND_FACTORIES:
            bdata = r["backends"].get(backend_name, {})
            verdict = bdata.get("judge_verdict", {})
            detected = verdict.get("suspicious_evidence_present", False)
            identified = verdict.get("attack_success_observed", False)
            is_benign = r["benign"] == 1
            table.add_row(
                r["test_case_id"],
                r["attack_type"] or "none",
                "[green]yes[/green]" if is_benign else "[red]no[/red]",
                backend_name,
                "[red]YES[/red]" if detected else "[green]no[/green]",
                "[red]YES[/red]" if identified else "[green]no[/green]",
                f"{verdict.get('confidence', 0.0):.2f}",
                str(verdict.get("reasoning", ""))[:50],
            )

    console.print(table)

    console.print()
    summary = {"backends": {}}
    detection_table = Table(title="Detection Metrics", box=box.SIMPLE)
    detection_table.add_column("Backend", style="cyan")
    detection_table.add_column("TP", justify="right")
    detection_table.add_column("FN", justify="right")
    detection_table.add_column("TN", justify="right")
    detection_table.add_column("FP", justify="right")
    detection_table.add_column("TPR", justify="right")
    detection_table.add_column("FPR", justify="right")
    detection_table.add_column("Pre-Output", justify="right")

    rca_table = Table(title="Root Cause Metrics", box=box.SIMPLE)
    rca_table.add_column("Backend", style="cyan")
    rca_table.add_column("Successful", justify="right")
    rca_table.add_column("Traceback", justify="right")
    rca_table.add_column("Doc/Query ID", justify="right")
    rca_table.add_column("Mode Dist.", justify="right")

    for backend_name in BACKEND_FACTORIES:
        tp = fn = tn = fp = 0
        attack_cases = benign_cases = 0
        near_rt_hits = 0
        by_attack_type = {}
        successful_cases = 0
        root_cause_hits = 0
        culprit_id_hits = 0
        failure_mode_hits = 0

        for r in all_results:
            attack_label = r["attack_type"] or "benign"
            bdata = r["backends"].get(backend_name, {})
            verdict = bdata.get("judge_verdict", {})
            root_cause = bdata.get("root_cause_verdict", {})
            metrics = bdata.get("evaluation_metrics", {})
            confusion = metrics.get("confusion_outcome")
            is_attack_case = metrics.get("ground_truth_attack_present", False)

            if attack_label not in by_attack_type:
                by_attack_type[attack_label] = {
                    "total": 0,
                    "detected": 0,
                    "missed": 0,
                }
            by_attack_type[attack_label]["total"] += 1

            if confusion == "true_positive":
                tp += 1
            elif confusion == "false_negative":
                fn += 1
            elif confusion == "true_negative":
                tn += 1
            elif confusion == "false_positive":
                fp += 1

            if is_attack_case:
                attack_cases += 1
                if verdict.get("suspicious_evidence_present", False):
                    by_attack_type[attack_label]["detected"] += 1
                else:
                    by_attack_type[attack_label]["missed"] += 1
                if verdict.get("pre_output_detectable", False):
                    near_rt_hits += 1
                if verdict.get("attack_success_observed", False):
                    successful_cases += 1
                    if root_cause.get("rca_possible", False):
                        root_cause_hits += 1
                    if root_cause.get("culprit_document_identified", False) or root_cause.get("culprit_reference"):
                        culprit_id_hits += 1
                    if root_cause.get("failure_mode_distinguishable", False):
                        failure_mode_hits += 1
            else:
                benign_cases += 1

        detection_rate = tp / attack_cases if attack_cases else None
        false_positive_rate = fp / benign_cases if benign_cases else None
        near_rt_rate = near_rt_hits / attack_cases if attack_cases else None
        traceback_rate = root_cause_hits / successful_cases if successful_cases else None
        culprit_id_rate = culprit_id_hits / successful_cases if successful_cases else None
        failure_mode_rate = failure_mode_hits / successful_cases if successful_cases else None

        summary["backends"][backend_name] = {
            "detection": {
                "attack_cases": attack_cases,
                "benign_cases": benign_cases,
                "true_positives": tp,
                "false_negatives": fn,
                "true_negatives": tn,
                "false_positives": fp,
                "detection_rate_true_positive_rate": detection_rate,
                "false_positive_rate": false_positive_rate,
                "pre_output_detection_rate": near_rt_rate,
                "attack_types_detected_vs_missed": by_attack_type,
            },
            "root_cause_analysis": {
                "successful_attack_cases": successful_cases,
                "traceback_to_malicious_doc_or_query_rate": traceback_rate,
                "specific_culprit_identification_rate": culprit_id_rate,
                "failure_mode_distinction_rate": failure_mode_rate,
            },
        }

        detection_table.add_row(
            backend_name,
            str(tp),
            str(fn),
            str(tn),
            str(fp),
            "-" if detection_rate is None else f"{detection_rate:.2f}",
            "-" if false_positive_rate is None else f"{false_positive_rate:.2f}",
            "-" if near_rt_rate is None else f"{near_rt_rate:.2f}",
        )
        rca_table.add_row(
            backend_name,
            str(successful_cases),
            "-" if traceback_rate is None else f"{traceback_rate:.2f}",
            "-" if culprit_id_rate is None else f"{culprit_id_rate:.2f}",
            "-" if failure_mode_rate is None else f"{failure_mode_rate:.2f}",
        )

    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    charts.write_charts(summary, results_dir)

    console.print(detection_table)
    console.print()
    console.print(rca_table)

    elapsed = time.perf_counter() - benchmark_started
    console.print(f"\n[bold]Benchmark runtime[/bold]: {elapsed:.2f}s")
    console.print(f"\n[dim]Results written to: {results_dir}/[/dim]")

if __name__ == "__main__":
    cli()
