import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BACKEND_COLORS = {
    "langsmith": "#2563eb",
    "langfuse": "#059669",
    "arize phoenix": "#dc2626",
}


def write_charts(summary: dict, results_dir: Path) -> None:
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    backends = _summary_backends(summary, results_dir)
    detection_summary = _summary_detection(summary, results_dir, backends)
    rca_summary = _summary_rca(summary, results_dir, backends)

    _write_detection_vs_fpr_chart(
        detection_summary=detection_summary,
        backends=backends,
        output_path=charts_dir / "attack_detection_vs_fpr.png",
    )
    _write_detection_by_attack_type_chart(
        detection_summary=detection_summary,
        backends=backends,
        output_path=charts_dir / "attack_detection_by_attack_type.png",
    )
    _write_detection_by_difficulty_chart(
        results_dir=results_dir,
        backends=backends,
        output_path=charts_dir / "attack_detection_by_difficulty.png",
    )
    _write_rca_chart(
        rca_summary=rca_summary,
        backends=backends,
        output_path=charts_dir / "rca_capabilities.png",
    )


def write_charts_from_summary_file(summary_path: Path, results_dir: Path) -> None:
    summary = json.loads(summary_path.read_text())
    write_charts(summary, results_dir)


def _write_detection_vs_fpr_chart(detection_summary: dict, backends: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=100)

    if not backends:
        ax.axis("off")
        ax.text(0.02, 0.9, "True Positive and False Positive Attack Detection Rates", fontsize=18, transform=ax.transAxes)
        ax.text(0.02, 0.72, "No data available", fontsize=12, transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    categories = ["True Positive Rate", "False Positive Rate"]
    x = np.arange(len(categories))
    width = 0.18 if len(backends) >= 3 else 0.24
    offsets = (np.arange(len(backends)) - (len(backends) - 1) / 2) * width

    for i, backend in enumerate(backends):
        metrics = detection_summary.get(backend, {})
        values = [
            metrics.get("detection_rate_true_positive_rate") or 0.0,
            metrics.get("false_positive_rate") or 0.0,
        ]
        values = [max(0.0, min(1.0, v)) for v in values]
        ax.bar(
            x + offsets[i],
            values,
            width=width * 0.9,
            color=BACKEND_COLORS.get(backend, "#374151"),
            label=backend,
        )

    ax.set_title("True Positive and False Positive Attack Detection Rates", fontsize=18, pad=14)
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_detection_by_difficulty_chart(results_dir: Path, backends: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=100)

    if not backends:
        ax.axis("off")
        ax.text(0.02, 0.9, "Attack Detection Rate by Task Difficulty", fontsize=18, transform=ax.transAxes)
        ax.text(0.02, 0.72, "No data available", fontsize=12, transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    difficulties = ["easy", "medium", "hard"]
    counts = {
        backend: {difficulty: {"total": 0, "detected": 0} for difficulty in difficulties}
        for backend in backends
    }

    for result_path in sorted(results_dir.glob("TC-*.json")):
        result = json.loads(result_path.read_text())
        if result.get("benign"):
            continue
        difficulty = str(result.get("difficulty") or "").lower()
        if difficulty not in difficulties:
            continue
        for backend in backends:
            verdict = _result_verdict(result, backend)
            counts[backend][difficulty]["total"] += 1
            if verdict.get("suspicious_evidence_present", False):
                counts[backend][difficulty]["detected"] += 1

    x = np.arange(len(difficulties))
    width = 0.18 if len(backends) >= 3 else 0.24
    offsets = (np.arange(len(backends)) - (len(backends) - 1) / 2) * width

    for i, backend in enumerate(backends):
        values = []
        for difficulty in difficulties:
            total = counts[backend][difficulty]["total"]
            detected = counts[backend][difficulty]["detected"]
            values.append((detected / total) if total else 0.0)
        ax.bar(
            x + offsets[i],
            values,
            width=width * 0.9,
            color=BACKEND_COLORS.get(backend, "#374151"),
            label=backend,
        )

    ax.set_title("Attack Detection Rate by Task Difficulty", fontsize=18, pad=14)
    ax.set_xticks(x, [d.title() for d in difficulties])
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_detection_by_attack_type_chart(detection_summary: dict, backends: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=100)

    attack_types = []
    for backend in backends:
        by_attack_type = detection_summary.get(backend, {}).get("attack_types_detected_vs_missed", {})
        for attack_type in by_attack_type:
            if attack_type == "benign" or attack_type in attack_types:
                continue
            attack_types.append(attack_type)

    if not backends or not attack_types:
        ax.axis("off")
        ax.text(0.02, 0.9, "Attack Detection Rate by Attack Type", fontsize=18, transform=ax.transAxes)
        ax.text(0.02, 0.72, "No data available", fontsize=12, transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(attack_types))
    width = min(0.24, 0.8 / max(len(backends), 1))
    offsets = (np.arange(len(backends)) - (len(backends) - 1) / 2) * width

    for i, backend in enumerate(backends):
        by_attack_type = detection_summary.get(backend, {}).get("attack_types_detected_vs_missed", {})
        values = []
        for attack_type in attack_types:
            counts = by_attack_type.get(attack_type, {})
            total = counts.get("total", 0)
            detected = counts.get("detected", 0)
            values.append((detected / total) if total else 0.0)

        ax.bar(
            x + offsets[i],
            values,
            width=width * 0.9,
            color=BACKEND_COLORS.get(backend, "#374151"),
            label=backend,
        )

    ax.set_title("Attack Detection Rate by Attack Type", fontsize=18, pad=14)
    ax.set_xticks(x, [_format_attack_type_label(attack_type) for attack_type in attack_types], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_rca_chart(rca_summary: dict, backends: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=100)

    if not backends:
        ax.axis("off")
        ax.text(0.02, 0.9, "Root Cause Analysis Capabilities", fontsize=18, transform=ax.transAxes)
        ax.text(0.02, 0.72, "No data available", fontsize=12, transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    categories = ["Specific Culprit Identified", "Failure Mode Distinguishable"]
    x = np.arange(len(categories))
    width = 0.18 if len(backends) >= 3 else 0.24
    offsets = (np.arange(len(backends)) - (len(backends) - 1) / 2) * width

    for i, backend in enumerate(backends):
        metrics = rca_summary.get(backend, {})
        values = [
            metrics.get("specific_culprit_identification_rate") or 0.0,
            metrics.get("failure_mode_distinction_rate") or 0.0,
        ]
        values = [max(0.0, min(1.0, v)) for v in values]
        ax.bar(
            x + offsets[i],
            values,
            width=width * 0.9,
            color=BACKEND_COLORS.get(backend, "#374151"),
            label=backend,
        )

    ax.set_title("Root Cause Analysis Capabilities", fontsize=18, pad=14)
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _summary_backends(summary: dict, results_dir: Path) -> list[str]:
    if "backends" in summary:
        return list(summary.get("backends", {}).keys())

    backends = []
    for result_path in sorted(results_dir.glob("TC-*.json")):
        result = json.loads(result_path.read_text())
        for backend_name, bdata in result.get("backends", {}).items():
            if (
                "judge_verdict" in bdata or "reanalyzed_judge_verdict" in bdata
            ) and backend_name not in backends:
                backends.append(backend_name)
    return backends


def _summary_detection(summary: dict, results_dir: Path, backends: list[str]) -> dict:
    if "backends" in summary:
        return {backend: data.get("detection", {}) for backend, data in summary.get("backends", {}).items()}

    detection = {}
    for backend in backends:
        tp = fn = tn = fp = 0
        attack_cases = benign_cases = 0

        for result_path in sorted(results_dir.glob("TC-*.json")):
            result = json.loads(result_path.read_text())
            verdict = _result_verdict(result, backend)
            if not verdict:
                continue

            is_attack = not bool(result.get("benign"))
            detected = verdict.get("suspicious_evidence_present", False)

            if is_attack:
                attack_cases += 1
                if detected:
                    tp += 1
                else:
                    fn += 1
            else:
                benign_cases += 1
                if detected:
                    fp += 1
                else:
                    tn += 1

        detection[backend] = {
            "detection_rate_true_positive_rate": (tp / attack_cases) if attack_cases else None,
            "false_positive_rate": (fp / benign_cases) if benign_cases else None,
            "attack_types_detected_vs_missed": _detection_by_attack_type(results_dir, backend),
        }
    return detection


def _summary_rca(summary: dict, results_dir: Path, backends: list[str]) -> dict:
    if "backends" in summary:
        return {backend: data.get("root_cause_analysis", {}) for backend, data in summary.get("backends", {}).items()}

    rca = {}
    for backend in backends:
        successful_cases = 0
        culprit_id_hits = 0
        failure_mode_hits = 0

        for result_path in sorted(results_dir.glob("TC-*.json")):
            result = json.loads(result_path.read_text())
            verdict = _result_verdict(result, backend)
            root_cause = _result_root_cause(result, backend)
            if not verdict or not root_cause:
                continue

            if verdict.get("attack_success_observed", False):
                successful_cases += 1
                if root_cause.get("culprit_document_identified", False) or root_cause.get("culprit_reference"):
                    culprit_id_hits += 1
                if root_cause.get("failure_mode_distinguishable", False):
                    failure_mode_hits += 1

        rca[backend] = {
            "specific_culprit_identification_rate": (
                culprit_id_hits / successful_cases if successful_cases else None
            ),
            "failure_mode_distinction_rate": (
                failure_mode_hits / successful_cases if successful_cases else None
            ),
        }
    return rca


def _result_verdict(result: dict, backend: str) -> dict:
    bdata = result.get("backends", {}).get(backend, {})
    return bdata.get("reanalyzed_judge_verdict") or bdata.get("judge_verdict") or {}


def _result_root_cause(result: dict, backend: str) -> dict:
    bdata = result.get("backends", {}).get(backend, {})
    return bdata.get("reanalyzed_root_cause_verdict") or bdata.get("root_cause_verdict") or {}


def _detection_by_attack_type(results_dir: Path, backend: str) -> dict:
    by_attack_type = {}

    for result_path in sorted(results_dir.glob("TC-*.json")):
        result = json.loads(result_path.read_text())
        if result.get("benign"):
            continue

        verdict = _result_verdict(result, backend)
        if not verdict:
            continue

        attack_type = result.get("attack_type") or "unknown"
        counts = by_attack_type.setdefault(attack_type, {"total": 0, "detected": 0, "missed": 0})
        counts["total"] += 1
        if verdict.get("suspicious_evidence_present", False):
            counts["detected"] += 1
        else:
            counts["missed"] += 1

    return by_attack_type


def _format_attack_type_label(attack_type: str) -> str:
    return str(attack_type).replace("_", " ").title()
