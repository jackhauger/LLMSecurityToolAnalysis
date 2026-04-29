"""
reanalyze.py — Re-judge saved traces without rerunning the pipeline.

Reads existing result JSON files plus their saved trace files, optionally
compresses the traces, reruns the current judges, and writes the outputs
to a separate reanalysis directory.

Usage:
    .venv/bin/python reanalyze.py
    .venv/bin/python reanalyze.py --results-dir results --trace-format compressed
    .venv/bin/python reanalyze.py --no-judge
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from compress import serialize as compress_serialize
from compress_targeted import compress_targeted
from config import cfg
from judge import DetectionJudge, RootCauseJudge
from rich import box
from rich.console import Console
from rich.table import Table


console = Console()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _judge_trace_text(traces, trace_format: str, backend_name: str = "") -> str:
    if trace_format == "compressed":
        return compress_serialize(traces)
    if trace_format == "targeted":
        return compress_targeted(traces, backend_name)
    return json.dumps(traces, indent=2, default=str)


def run(results_dir: Path, output_dir: Path, trace_format: str, run_judge: bool) -> None:
    result_files = sorted(results_dir.glob("TC-*.json"))
    if not result_files:
        print(f"No result JSON files found in {results_dir}")
        sys.exit(1)

    if run_judge:
        cfg.validate()
        detection_judge = DetectionJudge()
        root_cause_judge = RootCauseJudge()
    else:
        detection_judge = None
        root_cause_judge = None

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    console.print(f"[dim]Loaded {len(result_files)} result file(s) from {results_dir}[/dim]")
    console.rule("[bold blue]Reanalyze Saved Traces")

    for result_path in result_files:
        result = _load_json(result_path)
        tc_id = result.get("test_case_id", result_path.stem)
        tc_out_dir = output_dir / tc_id
        tc_out_dir.mkdir(parents=True, exist_ok=True)
        all_results.append(result)

        for backend_name, bdata in result.get("backends", {}).items():
            trace_file = bdata.get("trace_file")
            inline_traces = bdata.get("traces")
            if not trace_file and inline_traces is None:
                continue

            console.print(f"  [{backend_name}] {tc_id} loading saved trace...", end="")
            if trace_file:
                traces = _load_json(Path(trace_file))
            else:
                traces = inline_traces
            console.print(" compressing...", end="")
            trace_text = _judge_trace_text(traces, trace_format, backend_name)
            detection = None
            root_cause = None
            judge_prompt = ""
            judge_raw = ""
            root_prompt = ""
            root_raw = ""

            if run_judge:
                console.print(" judging...", end="")
                detection, judge_prompt, judge_raw = detection_judge.evaluate_trace(
                    trace_text,
                    backend_name,
                    bdata.get("final_response", ""),
                )
                if detection.suspicious_evidence_present or detection.attack_success_observed:
                    root_cause, root_prompt, root_raw = root_cause_judge.evaluate_trace(
                        trace_text,
                        backend_name,
                        result.get("attack_type") or "benign",
                        bdata.get("final_response", ""),
                    )
                detected_str = (
                    "[red]DETECTED[/red]" if detection.suspicious_evidence_present
                    else "[green]clean[/green]"
                )
                console.print(f" {detected_str} (conf={detection.confidence:.2f})")
            else:
                console.print(" saved")

            backend_out_dir = tc_out_dir / backend_name
            backend_out_dir.mkdir(parents=True, exist_ok=True)
            (backend_out_dir / f"judge_trace_{trace_format}.txt").write_text(trace_text)
            if run_judge:
                (backend_out_dir / "judge_prompt.txt").write_text(judge_prompt)
                (backend_out_dir / "judge_response.json").write_text(judge_raw)
                if root_cause is not None:
                    (backend_out_dir / "root_cause_prompt.txt").write_text(root_prompt)
                    (backend_out_dir / "root_cause_response.json").write_text(root_raw)

            if run_judge:
                bdata["reanalyzed_judge_trace_format"] = trace_format
                bdata["reanalyzed_judge_trace_file"] = str(backend_out_dir / f"judge_trace_{trace_format}.txt")
                bdata["reanalyzed_judge_verdict"] = asdict(detection)
                if root_cause is not None:
                    bdata["reanalyzed_root_cause_verdict"] = asdict(root_cause)
                else:
                    bdata.pop("reanalyzed_root_cause_verdict", None)
                ground_truth_attack_present = result.get("benign", 0) == 0
                if ground_truth_attack_present and detection.suspicious_evidence_present:
                    confusion_outcome = "true_positive"
                elif ground_truth_attack_present and not detection.suspicious_evidence_present:
                    confusion_outcome = "false_negative"
                elif not ground_truth_attack_present and detection.suspicious_evidence_present:
                    confusion_outcome = "false_positive"
                else:
                    confusion_outcome = "true_negative"
                bdata["reanalyzed_evaluation_metrics"] = {
                    "ground_truth_attack_present": ground_truth_attack_present,
                    "detection_correct": confusion_outcome in ("true_positive", "true_negative"),
                    "confusion_outcome": confusion_outcome,
                }

        out_result_path = output_dir / result_path.name
        out_result_path.write_text(json.dumps(result, indent=2, default=str))

    if not run_judge:
        summary = {
            "source_results_dir": str(results_dir),
            "output_dir": str(output_dir),
            "trace_format": trace_format,
            "run_judge": False,
        }
        (output_dir / "reanalyze_summary.json").write_text(json.dumps(summary, indent=2))
        console.print(f"\n[dim]Reanalysis written to: {output_dir}/[/dim]")
        return

    console.print()
    summary_table = Table(title="Benchmark Summary", box=box.SIMPLE)
    summary_table.add_column("Test Case", style="cyan")
    summary_table.add_column("Attack Type", style="dim")
    summary_table.add_column("Benign?", justify="center")
    summary_table.add_column("Backend", style="dim")
    summary_table.add_column("Attack Detected?", justify="center")
    summary_table.add_column("Attack Success?", justify="center")
    summary_table.add_column("Confidence", justify="right")
    summary_table.add_column("Reasoning", max_width=50)

    for result in all_results:
        for backend_name, bdata in result.get("backends", {}).items():
            verdict = bdata.get("reanalyzed_judge_verdict", {})
            if not verdict:
                continue
            summary_table.add_row(
                result.get("test_case_id", ""),
                result.get("attack_type") or "none",
                "[green]yes[/green]" if result.get("benign") == 1 else "[red]no[/red]",
                backend_name,
                "[red]YES[/red]" if verdict.get("suspicious_evidence_present", False) else "[green]no[/green]",
                "[red]YES[/red]" if verdict.get("attack_success_observed", False) else "[green]no[/green]",
                f"{verdict.get('confidence', 0.0):.2f}",
                str(verdict.get("reasoning", ""))[:50],
            )

    console.print(summary_table)
    console.print()

    summary = {
        "source_results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "trace_format": trace_format,
        "backends": {},
    }
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

    backend_names = []
    for result in all_results:
        for backend_name, bdata in result.get("backends", {}).items():
            if "reanalyzed_judge_verdict" in bdata and backend_name not in backend_names:
                backend_names.append(backend_name)

    for backend_name in backend_names:
        tp = fn = tn = fp = 0
        attack_cases = benign_cases = 0
        near_rt_hits = 0
        by_attack_type = {}
        successful_cases = 0
        root_cause_hits = 0
        culprit_id_hits = 0
        failure_mode_hits = 0

        for result in all_results:
            attack_label = result.get("attack_type") or "benign"
            bdata = result.get("backends", {}).get(backend_name, {})
            verdict = bdata.get("reanalyzed_judge_verdict", {})
            root_cause = bdata.get("reanalyzed_root_cause_verdict", {})
            metrics = bdata.get("reanalyzed_evaluation_metrics", {})
            if not verdict or not metrics:
                continue

            confusion = metrics.get("confusion_outcome")
            is_attack_case = metrics.get("ground_truth_attack_present", False)

            if attack_label not in by_attack_type:
                by_attack_type[attack_label] = {"total": 0, "detected": 0, "missed": 0}
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

    (output_dir / "reanalyze_summary.json").write_text(json.dumps(summary, indent=2))
    console.print(detection_table)
    console.print()
    console.print(rca_table)
    console.print(f"\n[dim]Reanalysis written to: {output_dir}/[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-judge saved traces")
    parser.add_argument("--results-dir", default="results", help="Directory containing saved benchmark result JSON files")
    parser.add_argument("--output-dir", default="reanalyzed_results", help="Directory where reanalysis outputs should be written")
    parser.add_argument("--trace-format", choices=["raw", "compressed", "targeted"], default="compressed", help="Format of trace text given to the judges")
    parser.add_argument("--no-judge", action="store_true", help="Only write transformed trace text, skip judge calls")
    args = parser.parse_args()

    run(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        trace_format=args.trace_format,
        run_judge=not args.no_judge,
    )
