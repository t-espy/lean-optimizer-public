#!/usr/bin/env python3
"""Parse universe optimization logs to report when each symbol's GA found its
global best vs when early stopping triggered.

Usage:
    python scripts/ga_early_stop_analysis.py [log_dir]

Default log_dir: logs/
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
TIMESTAMP = "YYYYMMDD_HHMMSS"  # set to your universe run timestamp

# Patterns
INIT_RE = re.compile(
    r"\[GeneticStage\] Init: (\d+) individuals\s+best=([\d.e+-]+)"
)
GEN_RE = re.compile(
    r"\[GeneticStage\] Gen (\d+)\s+best=([\d.e+-]+)\s+worst=([\d.e+-]+)\s+avg=([\d.e+-]+)\s+accepted=(\d+)/(\d+)"
)
COMPLETE_RE = re.compile(
    r"\[GeneticStage\] Complete\. (\d+) evaluations \((\d+) generations\)\. best=([\d.e+-]+)"
)
EARLY_STOP_RE = re.compile(
    r"\[GeneticStage\] Early stopping.*Best=([\d.e+-]+) at gen_(\d+)"
)
SYMBOL_START_RE = re.compile(
    r"\[(\w+)\] Starting optimization"
)


def parse_log(filepath: Path) -> list[dict]:
    """Parse a single symbol log, returning a list of GA run summaries."""
    runs = []
    current_run = None

    with open(filepath) as f:
        for line in f:
            # Init line starts a new GA run
            m = INIT_RE.search(line)
            if m:
                if current_run and current_run.get("total_gens") is not None:
                    runs.append(current_run)
                current_run = {
                    "pop_size": int(m.group(1)),
                    "init_best": float(m.group(2)),
                    "best_fitness": float(m.group(2)),
                    "best_gen": "init",
                    "gen_history": [("init", float(m.group(2)))],
                }
                continue

            if current_run is None:
                continue

            # Per-generation line
            m = GEN_RE.search(line)
            if m:
                gen = int(m.group(1))
                best = float(m.group(2))
                current_run["gen_history"].append((f"gen_{gen}", best))
                if best > current_run["best_fitness"]:
                    current_run["best_fitness"] = best
                    current_run["best_gen"] = gen
                continue

            # Early stopping line
            m = EARLY_STOP_RE.search(line)
            if m:
                current_run["early_stop_gen"] = int(m.group(2))
                continue

            # Complete line
            m = COMPLETE_RE.search(line)
            if m:
                current_run["total_evals"] = int(m.group(1))
                current_run["total_gens"] = int(m.group(2))
                current_run["final_best"] = float(m.group(3))
                runs.append(current_run)
                current_run = None
                continue

    if current_run and current_run.get("total_gens") is not None:
        runs.append(current_run)

    return runs


def main():
    log_files = sorted(LOG_DIR.glob(f"universe_*_{TIMESTAMP}.log"))
    if not log_files:
        print(f"No log files found matching universe_*_{TIMESTAMP}.log in {LOG_DIR}")
        sys.exit(1)

    print(f"{'Symbol':<8} {'Window':<8} {'Best':>10} {'Found@Gen':>10} {'Stopped@Gen':>12} {'TotalGens':>10} {'Wasted':>8} {'Evals':>8} {'WastedEvals':>12}")
    print("-" * 100)

    all_wasted = []
    all_wasted_evals = []
    total_evals = 0
    total_runs = 0

    for logfile in log_files:
        symbol = logfile.stem.split("_")[1]  # universe_SYMBOL_... -> SYMBOL
        runs = parse_log(logfile)

        for i, run in enumerate(runs):
            best_gen = run["best_gen"]
            if best_gen == "init":
                best_gen_num = 0
            else:
                best_gen_num = best_gen

            stop_gen = run.get("early_stop_gen", run["total_gens"] - 1)
            total_gens = run["total_gens"]
            evals = run["total_evals"]
            wasted_gens = stop_gen - best_gen_num if isinstance(best_gen_num, int) else stop_gen

            # Estimate wasted evals: wasted_gens * batch_size (15)
            # More accurate: count from gen_history
            batch_size = 15  # from config
            wasted_evals = wasted_gens * batch_size

            all_wasted.append(wasted_gens)
            all_wasted_evals.append(wasted_evals)
            total_evals += evals
            total_runs += 1

            print(
                f"{symbol:<8} {'WF' + str(i+1):<8} {run['best_fitness']:>10.4f} "
                f"{'init' if run['best_gen'] == 'init' else 'gen_' + str(run['best_gen']):>10} "
                f"{'gen_' + str(stop_gen):>12} "
                f"{total_gens:>10} "
                f"{wasted_gens:>8} "
                f"{evals:>8} "
                f"{wasted_evals:>12}"
            )

    print("-" * 100)
    print()
    print("SUMMARY")
    print(f"  Total GA runs:         {total_runs}")
    print(f"  Total evaluations:     {total_evals}")
    print(f"  Total wasted evals:    {sum(all_wasted_evals)} ({100*sum(all_wasted_evals)/total_evals:.1f}% of total)")
    print()
    print(f"  Avg wasted gens:       {sum(all_wasted)/len(all_wasted):.1f}")
    print(f"  Max wasted gens:       {max(all_wasted)}")
    print(f"  Min wasted gens:       {min(all_wasted)}")
    print()

    # What-if analysis for different early_stopping_generations values
    print("WHAT-IF: Savings with different early_stopping_generations values")
    print(f"  {'Window':>8}  {'AvgWasted':>10}  {'TotalWastedEvals':>18}  {'% of Total':>10}")
    for window in [5, 8, 10, 12, 15]:
        saved = 0
        for logfile in log_files:
            symbol = logfile.stem.split("_")[1]
            runs = parse_log(logfile)
            for run in runs:
                best_gen = run["best_gen"]
                best_gen_num = 0 if best_gen == "init" else best_gen
                stop_gen = run.get("early_stop_gen", run["total_gens"] - 1)
                # With this window, would have stopped at best_gen + window
                new_stop = best_gen_num + window
                actual_stop = stop_gen
                # Evals saved = (actual_stop - new_stop) * batch_size, if new_stop < actual_stop
                if new_stop < actual_stop:
                    saved += (actual_stop - new_stop) * 15

        wasted_at_window = sum(all_wasted_evals) - saved  # remaining waste
        print(f"  {window:>8}  {window:>10}  {window * 15 * total_runs:>18}  (saves {saved:>6} evals = {100*saved/total_evals:.1f}%)")

    # Show per-run: when best was found relative to total
    print()
    print("BEST-FOUND DISTRIBUTION (gen where global best was found)")
    gen_buckets = defaultdict(int)
    for logfile in log_files:
        runs = parse_log(logfile)
        for run in runs:
            bg = run["best_gen"]
            if bg == "init":
                gen_buckets["0-5"] += 1
            elif bg <= 5:
                gen_buckets["0-5"] += 1
            elif bg <= 10:
                gen_buckets["6-10"] += 1
            elif bg <= 15:
                gen_buckets["11-15"] += 1
            elif bg <= 20:
                gen_buckets["16-20"] += 1
            elif bg <= 30:
                gen_buckets["21-30"] += 1
            elif bg <= 40:
                gen_buckets["31-40"] += 1
            else:
                gen_buckets["41+"] += 1

    for bucket in ["0-5", "6-10", "11-15", "16-20", "21-30", "31-40", "41+"]:
        count = gen_buckets.get(bucket, 0)
        bar = "#" * count
        print(f"  gen {bucket:>5}: {count:>3} runs  {bar}")


if __name__ == "__main__":
    main()
