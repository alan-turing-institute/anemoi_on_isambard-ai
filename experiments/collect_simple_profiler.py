#!/usr/bin/env python3
"""
Collect simple profiler data from time_profiler.csv and speed_profiler.csv
and emit a markdown table ready to paste into the report.

Usage:
    python collect_simple_profiler.py [--gpus-per-node N] <dir>:<n_gpus> [<dir>:<n_gpus> ...]

    Same directory conventions as collect_scaling_data.py:
        - numbered subdirectories (1/, 2/, 3/, ...) are averaged
        - or directly a profiler directory

Example:
    python collect_simple_profiler.py \\
        data/1gpu/baseline_final:1 \\
        data/1node/baseline_final:4 \\
        data/2nodes/baseline:8 \\
        data/10nodes/baseline:40 \\
        data/25nodes/baseline:100 \\
        data/50nodes/baseline:200 \\
        data/100nodes/baseline:400
"""

import sys
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_profiler_files(base_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Return [(label, time_csv, speed_csv), ...] for all experiments under base_dir.
    Looks for numbered subdirectories first; falls back to base_dir itself.
    """
    numbered = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    search_dirs = numbered if numbered else [base_dir]

    results = []
    for d in search_dirs:
        time_files  = list(d.glob("**/time_profiler.csv"))
        speed_files = list(d.glob("**/speed_profiler.csv"))
        if time_files and speed_files:
            results.append((d.name, time_files[0], speed_files[0]))
    return results


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_time_csv(path: Path) -> dict[str, float]:
    """Return {name: avg_time_seconds} for all rows."""
    result = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["name"]] = float(row["avg_time"])
    return result


def parse_speed_csv(path: Path) -> dict[str, float]:
    """Return {metric: value}."""
    result = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                result[row["metric"]] = float(row["value"])
            except ValueError:
                result[row["metric"]] = float("nan")
    return result


def collect_case(base_dir: Path) -> tuple[dict, dict, int]:
    """Load and average all experiments. Returns (time_avg, speed_avg, n_runs)."""
    files = find_profiler_files(base_dir)
    if not files:
        raise FileNotFoundError(f"No profiler CSVs found under {base_dir}")

    print(f"  {base_dir} — {len(files)} run(s): " +
          ", ".join(label for label, _, _ in files))

    all_time  = [parse_time_csv(t)  for _, t, _ in files]
    all_speed = [parse_speed_csv(s) for _, _, s in files]

    # Average time metrics (only keys present in all runs)
    time_keys = set(all_time[0].keys())
    for t in all_time[1:]:
        time_keys &= set(t.keys())
    time_avg = {k: sum(t[k] for t in all_time) / len(all_time) for k in time_keys}

    # Average speed metrics
    speed_keys = set(all_speed[0].keys())
    for s in all_speed[1:]:
        speed_keys &= set(s.keys())
    speed_avg = {k: sum(s[k] for s in all_speed) / len(all_speed) for k in speed_keys}

    return time_avg, speed_avg, len(files)


# ---------------------------------------------------------------------------
# Column header
# ---------------------------------------------------------------------------

def col_header(n_gpus: int, gpus_per_node: int, n_runs: int) -> str:
    runs = f" [{n_runs}]"
    if n_gpus == 1:
        return f"1-GPU{runs}"
    nodes = n_gpus // gpus_per_node
    return f"{n_gpus}-GPU ({nodes} node{'s' if nodes > 1 else ''}){runs}"


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def render_markdown_table(cases: list[dict], headers: list[str]) -> str:
    rows = [
        ("`run_training_batch` avg (ms)", [c["run_training_batch_ms"] for c in cases]),
        ("`backward` avg (ms)",           [c["backward_ms"]           for c in cases]),
        ("`training_step` avg (ms)",      [c["training_step_ms"]      for c in cases]),
        ("Total throughput (samples/s)",  [c["total_throughput"]       for c in cases]),
        ("Dataloader throughput (batches/s)", [c["dataloader_throughput"] for c in cases]),
    ]

    col_w = []
    col_w.append(max(len("Metric"), max(len(label) for label, _ in rows)))
    for idx, h in enumerate(headers):
        col_w.append(max(len(h), max(len(vals[idx]) for _, vals in rows)))

    def md_row(cells):
        parts = []
        for j, (cell, w) in enumerate(zip(cells, col_w)):
            parts.append(cell.ljust(w) if j == 0 else cell.rjust(w))
        return "| " + " | ".join(parts) + " |"

    sep = [":" + "-" * (col_w[0] - 1)] + ["-" * (w - 1) + ":" for w in col_w[1:]]

    lines = [
        md_row(["Metric"] + headers),
        "| " + " | ".join(sep) + " |",
    ]
    for label, vals in rows:
        lines.append(md_row([label] + vals))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(v: float, decimals: int = 1) -> str:
    if v != v:  # nan
        return "—"
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{decimals}f}"


def main():
    args = sys.argv[1:]

    gpus_per_node = 4
    if args and args[0].startswith("--gpus-per-node"):
        if "=" in args[0]:
            gpus_per_node = int(args[0].split("=")[1])
            args = args[1:]
        else:
            gpus_per_node = int(args[1])
            args = args[2:]

    if not args:
        print(__doc__)
        sys.exit(1)

    entries = []
    for arg in args:
        if ":" not in arg:
            print(f"Error: expected <dir>:<n_gpus>, got '{arg}'")
            sys.exit(1)
        dir_str, n_str = arg.rsplit(":", 1)
        entries.append((Path(dir_str), int(n_str)))

    print("Collecting data...")
    cases = []
    for base_dir, n_gpus in entries:
        time_avg, speed_avg, n_runs = collect_case(base_dir)

        # run_training_batch avg → ms
        rtb_ms = time_avg.get("run_training_batch", float("nan")) * 1000

        # backward: DDPGroupStrategy.backward → ms
        bwd_ms = time_avg.get(
            "[Strategy]DDPGroupStrategy.backward", float("nan")
        ) * 1000

        # training_step: DDPGroupStrategy.training_step → ms
        step_ms = time_avg.get(
            "[Strategy]DDPGroupStrategy.training_step", float("nan")
        ) * 1000

        # total throughput = per-rank throughput (batches/s) × n_gpus × samples_per_batch
        # training_avg_throughput is in batches/s; training_avg_throughput_per_sample
        # is in samples/s, so samples_per_batch = throughput / throughput_per_sample
        per_rank_batches = speed_avg.get("training_avg_throughput", float("nan"))
        per_rank_samples = speed_avg.get("training_avg_throughput_per_sample", float("nan"))
        if per_rank_samples and per_rank_samples == per_rank_samples:  # not nan
            samples_per_batch = per_rank_batches / per_rank_samples
        else:
            samples_per_batch = 1.0
        total_throughput = per_rank_batches * n_gpus * samples_per_batch

        # dataloader throughput (batches/s)
        dl = speed_avg.get("avg_training_dataloader_throughput", float("nan"))

        cases.append({
            "n_gpus": n_gpus,
            "n_runs": n_runs,
            "run_training_batch_ms": fmt(rtb_ms),
            "backward_ms":           fmt(bwd_ms),
            "training_step_ms":      fmt(step_ms),
            "total_throughput":      fmt(total_throughput),
            "dataloader_throughput": fmt(dl),
        })

    headers = [col_header(c["n_gpus"], gpus_per_node, c["n_runs"]) for c in cases]

    print()
    print("**Simple profiler** (complementary to nsys, all values are per-rank averages):\n")
    print(render_markdown_table(cases, headers))


if __name__ == "__main__":
    main()
