

import argparse
import json
from pathlib import Path

from reanalyze import run as reanalyze_run
from charts import write_compression_comparison_charts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generic vs targeted trace compression")
    parser.add_argument("--results-dir", default="results", help="Directory with benchmark result JSON files")
    parser.add_argument("--generic-dir", default="reanalyzed_results_generic", help="Output dir for generic compression run")
    parser.add_argument("--targeted-dir", default="reanalyzed_results_targeted", help="Output dir for targeted compression run")
    parser.add_argument("--charts-dir", default="compression_comparison/charts", help="Output dir for comparison charts")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge calls (only write trace text files)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    generic_dir = Path(args.generic_dir)
    targeted_dir = Path(args.targeted_dir)
    charts_dir = Path(args.charts_dir)
    run_judge = not args.no_judge

    print("=== Running generic (compressed) ===")
    reanalyze_run(
        results_dir=results_dir,
        output_dir=generic_dir,
        trace_format="compressed",
        run_judge=run_judge,
    )

    print("\n=== Running targeted compression ===")
    reanalyze_run(
        results_dir=results_dir,
        output_dir=targeted_dir,
        trace_format="targeted",
        run_judge=run_judge,
    )

    generic_summary = json.loads((generic_dir / "reanalyze_summary.json").read_text())
    targeted_summary = json.loads((targeted_dir / "reanalyze_summary.json").read_text())

    print(f"\n=== Writing comparison charts to {charts_dir}/ ===")
    write_compression_comparison_charts(
        generic_summary=generic_summary,
        targeted_summary=targeted_summary,
        generic_dir=generic_dir,
        targeted_dir=targeted_dir,
        out_dir=charts_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
