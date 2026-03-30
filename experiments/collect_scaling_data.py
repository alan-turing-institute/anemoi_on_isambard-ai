#!/usr/bin/env python3
"""
Collect per-step scaling table data from nsys rank0 report files and emit a
markdown table ready to paste into the report.

Usage:
    python collect_scaling_data.py [--gpus-per-node N] <dir>:<n_gpus> [<dir>:<n_gpus> ...]

    Each <dir>:<n_gpus> pair specifies an experiment directory and its GPU count.
    The directory may contain:
        - numbered subdirectories (1/, 2/, 3/, ...) each with a rank0_report_*.txt
          → metrics are averaged across all subdirectories
        - or directly a rank0_report_*.txt

    The first entry must be the 1-GPU baseline; it is used for all efficiency
    calculations. --gpus-per-node defaults to 4.

Example:
    python collect_scaling_data.py \\
        data/1gpu/baseline_final:1 \\
        data/1node/baseline:4 \\
        data/2nodes/baseline:8 \\
        data/10nodes/baseline:40 \\
        data/50nodes/baseline:200 \\
        data/100_nodes/baseline:400
"""

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_report(report_path: Path) -> dict:
    """
    Parse a rank0 nsys report and return a flat dict of raw metrics (all in ns).

    Keys: step_med, step_min, step_max, step_std,
          backward_med, backward_min, backward_max, backward_std,
          optimizer_med, optimizer_min, optimizer_max, optimizer_std,
          cuda_launch_kernel_med
    """
    text = report_path.read_text()
    result = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # NVTX ranges: lines containing "PushPop  :<phase>"
        for phase in ("step", "backward", "optimizer"):
            if f"PushPop  :{phase}" in stripped:
                parts = stripped.replace(",", "").split()
                # cols: pct total instances avg med min max std Style :name
                try:
                    result[f"{phase}_med"] = float(parts[4])
                    result[f"{phase}_min"] = float(parts[5])
                    result[f"{phase}_max"] = float(parts[6])
                    result[f"{phase}_std"] = float(parts[7])
                except (IndexError, ValueError):
                    pass
                break

        # CUDA API: cudaLaunchKernel (exclude cudaLaunchKernelExC)
        if stripped.endswith("cudaLaunchKernel") and "ExC" not in stripped:
            parts = stripped.replace(",", "").split()
            try:
                result["cuda_launch_kernel_med"] = float(parts[4])
            except (IndexError, ValueError):
                pass

    return result


def find_reports(base_dir: Path) -> list[tuple[str, Path]]:
    """
    Return [(label, report_path), ...] for all experiments under base_dir.
    Looks for numbered subdirectories first; falls back to base_dir itself.
    """
    numbered = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    reports = []
    for d in (numbered if numbered else [base_dir]):
        found = sorted(d.glob("rank0_report*.txt"))
        if found:
            reports.append((d.name, found[0]))
    return reports


def collect_case(base_dir: Path) -> tuple[dict, int]:
    """Load and average all experiments in base_dir. Returns (metrics, n_runs)."""
    reports = find_reports(base_dir)
    if not reports:
        raise FileNotFoundError(f"No rank0_report*.txt found under {base_dir}")

    print(f"  {base_dir} — {len(reports)} run(s): " +
          ", ".join(label for label, _ in reports))

    all_metrics = [parse_report(path) for _, path in reports]
    keys = all_metrics[0].keys()
    avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}
    avg["forward_med"] = avg["step_med"] - avg["backward_med"] - avg["optimizer_med"]
    return avg, len(reports)


# ---------------------------------------------------------------------------
# Markdown table rendering
# ---------------------------------------------------------------------------

def fmt_ms(val_ns: float) -> str:
    return f"{val_ns / 1e6:.1f}"


def fmt_us(val_ns: float) -> str:
    return f"{val_ns / 1e3:.3f}"


def col_header(n_gpus: int, gpus_per_node: int, n_runs: int) -> str:
    runs = f" [{n_runs}]"
    if n_gpus == 1:
        return f"1-GPU{runs}"
    nodes = n_gpus // gpus_per_node
    return f"{n_gpus}-GPU ({nodes} node{'s' if nodes > 1 else ''}){runs}"


def render_markdown_table(cases: list[dict], headers: list[str]) -> str:
    rows = [
        ("Step Med (ms)",               [fmt_ms(c["step_med"])             for c in cases]),
        ("Step Min (ms)",               [fmt_ms(c["step_min"])             for c in cases]),
        ("Step Max (ms)",               [fmt_ms(c["step_max"])             for c in cases]),
        ("Step StdDev (ms)",            [fmt_ms(c["step_std"])             for c in cases]),
        ("Backward Med (ms)",           [fmt_ms(c["backward_med"])         for c in cases]),
        ("Backward Min (ms)",           [fmt_ms(c["backward_min"])         for c in cases]),
        ("Backward Max (ms)",           [fmt_ms(c["backward_max"])         for c in cases]),
        ("Backward StdDev (ms)",        [fmt_ms(c["backward_std"])         for c in cases]),
        ("Optimizer Med (ms)",          [fmt_ms(c["optimizer_med"])        for c in cases]),
        ("Optimizer Min (ms)",          [fmt_ms(c["optimizer_min"])        for c in cases]),
        ("Optimizer Max (ms)",          [fmt_ms(c["optimizer_max"])        for c in cases]),
        ("Optimizer StdDev (ms)",       [fmt_ms(c["optimizer_std"])        for c in cases]),
        ("Forward Med (derived)",       [fmt_ms(c["forward_med"])          for c in cases]),
        ("`cudaLaunchKernel` Med (µs)", [fmt_us(c.get("cuda_launch_kernel_med", 0)) for c in cases]),
        ("**Scaling efficiency**",      [c["efficiency_str"]               for c in cases]),
        ("**Effective GPU count**",     [c["effective_str"]                for c in cases]),
        ("**Wasted GPUs**",             [c["wasted_str"]                   for c in cases]),
        ("**Step overhead vs 1-GPU (ms)**", [c["overhead_str"]            for c in cases]),
        ("**Overhead per node (ms)**",  [c["overhead_per_node_str"]        for c in cases]),
    ]

    # Column widths
    col_w = [max(len(h), max(len(v) for _, vals in rows for v in vals if vals))
             for h in ["Phase"] + headers]
    col_w[0] = max(len("Phase"), max(len(label) for label, _ in rows))
    for i, h in enumerate(headers):
        col_w[i + 1] = max(len(h), max(len(vals[i]) for _, vals in rows))

    def md_row(cells: list[str], aligns: list[str]) -> str:
        parts = []
        for cell, w, align in zip(cells, col_w, aligns):
            if align == "left":
                parts.append(cell.ljust(w))
            else:
                parts.append(cell.rjust(w))
        return "| " + " | ".join(parts) + " |"

    aligns = ["left"] + ["right"] * len(headers)
    sep_cells = [("-" * w) for w in col_w]
    sep_cells[0] = ":" + "-" * (col_w[0] - 1)
    for i in range(1, len(sep_cells)):
        sep_cells[i] = "-" * (col_w[i] - 1) + ":"

    lines = [
        md_row(["Phase"] + headers, aligns),
        "| " + " | ".join(sep_cells) + " |",
    ]
    for label, vals in rows:
        lines.append(md_row([label] + vals, aligns))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Parse dir:n_gpus pairs
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
        avg, n_runs = collect_case(base_dir)
        avg["n_gpus"] = n_gpus
        avg["n_runs"] = n_runs
        cases.append(avg)

    # First entry is the 1-GPU baseline
    baseline_ms = cases[0]["step_med"] / 1e6

    # Compute derived efficiency rows for each case
    for c in cases:
        n = c["n_gpus"]
        step_ms = c["step_med"] / 1e6
        if n == 1:
            c["efficiency_str"] = "100%"
            c["effective_str"] = "1.0"
            c["wasted_str"] = "0"
            c["overhead_str"] = "0"
            c["overhead_per_node_str"] = "—"
        else:
            eff = baseline_ms / step_ms
            effective = n * eff
            wasted = n - effective
            overhead = step_ms - baseline_ms
            n_nodes = n / gpus_per_node
            overhead_per_node = overhead / n_nodes
            c["efficiency_str"] = f"{eff * 100:.1f}%"
            c["effective_str"] = f"{effective:.1f}"
            c["wasted_str"] = f"{wasted:.1f}"
            c["overhead_str"] = f"+{overhead:.1f}"
            c["overhead_per_node_str"] = f"{overhead_per_node:.1f}"

    headers = [col_header(c["n_gpus"], gpus_per_node, c["n_runs"]) for c in cases]

    print()
    print("**Per-step scaling** (Simple profiler, NVTX, nsys profile, rank 0):\n")
    print(render_markdown_table(cases, headers))


if __name__ == "__main__":
    main()
