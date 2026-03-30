#!/usr/bin/env python3
"""
Collect startup overhead data from SLURM .out files containing [STARTUP] lines
and emit a markdown table ready to paste into the report.

Usage:
    python collect_startup_overhead.py [--gpus-per-node N] <dir>:<n_gpus> [<dir>:<n_gpus> ...]

    Same directory conventions as collect_scaling_data.py:
        - numbered subdirectories (1/, 2/, 3/, ...) are averaged
        - or directly a directory containing a .out file

Example:
    python collect_startup_overhead.py \\
        data/1gpu/baseline_final:1 \\
        data/1node/baseline_final:4 \\
        data/2nodes/baseline:8 \\
        data/10nodes/baseline:40 \\
        data/25nodes/baseline:100 \\
        data/50nodes/baseline:200 \\
        data/100nodes/baseline:400
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Phase definitions — order and labels must match the callback output exactly
# ---------------------------------------------------------------------------

PHASES = [
    ("setup",        "setup (model + data ready)",       "T0 → setup (model + data ready)"),
    ("fit_start",    "on_fit_start",                     "setup → on_fit_start (Lightning init)"),
    ("train_start",  "on_train_start",                   "on_fit_start → on_train_start (NCCL init)"),
    ("batch_start",  "first batch start",                "on_train_start → first batch start (bucket alloc)"),
    ("first_batch",  "first batch end (compile done)",   "First batch (NCCL warmup)"),
]

# Internal key → report row label
PHASE_KEYS   = [p[0] for p in PHASES]
PHASE_MATCH  = [p[1] for p in PHASES]
PHASE_LABELS = [p[2] for p in PHASES]

# Regex: captures label text and delta value from a [STARTUP] line.
# Handles optional rank prefix like "0: " or "  0: ".
STARTUP_RE = re.compile(
    r"\[STARTUP\]\s+(.+?)\s{2,}elapsed=\s*[\d.]+s\s+delta=\s*([\d.]+)s"
)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_out_files(base_dir: Path) -> list[tuple[str, Path]]:
    """Return [(label, out_path), ...] from numbered subdirs or base_dir itself."""
    numbered = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    results = []
    for d in (numbered if numbered else [base_dir]):
        found = sorted(d.glob("*.out"))
        if found:
            results.append((d.name, found[0]))
    return results


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_out_file(path: Path) -> dict[str, float]:
    """
    Parse a SLURM .out file and return {phase_key: delta_seconds}.
    Only the first occurrence of each phase label is used (the callback can
    re-fire on_train_batch_start for subsequent batches).
    """
    result = {}
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        m = STARTUP_RE.search(line)
        if not m:
            continue
        label_text = m.group(1).strip()
        delta = float(m.group(2))
        for key, match_str in zip(PHASE_KEYS, PHASE_MATCH):
            if key not in seen and match_str in label_text:
                result[key] = delta
                seen.add(key)
                break
    return result


def collect_case(base_dir: Path) -> tuple[dict[str, float], int]:
    """Load and average all runs in base_dir. Returns (phase_avg, n_runs)."""
    files = find_out_files(base_dir)
    if not files:
        raise FileNotFoundError(f"No .out files found under {base_dir}")

    print(f"  {base_dir} — {len(files)} run(s): " +
          ", ".join(label for label, _ in files))

    all_runs = []
    for label, path in files:
        parsed = parse_out_file(path)
        missing = [k for k in PHASE_KEYS if k not in parsed]
        if missing:
            print(f"    WARNING: {label} missing phases {missing} in {path.name}")
        all_runs.append(parsed)

    # Average only keys present in all runs
    keys = set(all_runs[0].keys())
    for r in all_runs[1:]:
        keys &= set(r.keys())

    avg = {k: sum(r[k] for r in all_runs) / len(all_runs) for k in keys}
    return avg, len(all_runs)


# ---------------------------------------------------------------------------
# Column headers
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

def fmt(v: float) -> str:
    return f"{v:.1f} s"


def render_markdown_table(
    cases: list[dict],
    headers: list[str],
    baseline_total: float,
) -> str:
    # Build rows: phase rows + totals + vs-baseline
    phase_rows = []
    for key, label in zip(PHASE_KEYS, PHASE_LABELS):
        vals = [fmt(c.get(key, float("nan"))) for c in cases]
        phase_rows.append((label, vals))

    # Total = sum of all phase deltas
    totals = []
    for c in cases:
        t = sum(c.get(k, float("nan")) for k in PHASE_KEYS)
        totals.append(fmt(t))

    vs_baseline = []
    for c in cases:
        t = sum(c.get(k, float("nan")) for k in PHASE_KEYS)
        diff = t - baseline_total
        vs_baseline.append("—" if abs(diff) < 0.05 else f"+{diff:.1f} s")

    all_rows = phase_rows + [
        ("**Total**",    [f"**{v}**" for v in totals]),
        ("**vs 1-GPU**", vs_baseline),
    ]

    # Column widths
    col_w = [max(len("Phase"), max(len(label) for label, _ in all_rows))]
    for i, h in enumerate(headers):
        col_w.append(max(len(h), max(len(vals[i]) for _, vals in all_rows)))

    def md_row(cells):
        parts = []
        for j, (cell, w) in enumerate(zip(cells, col_w)):
            parts.append(cell.ljust(w) if j == 0 else cell.rjust(w))
        return "| " + " | ".join(parts) + " |"

    sep = [":" + "-" * (col_w[0] - 1)] + ["-" * (w - 1) + ":" for w in col_w[1:]]

    lines = [
        md_row(["Phase"] + headers),
        "| " + " | ".join(sep) + " |",
    ]
    for label, vals in all_rows:
        lines.append(md_row([label] + vals))

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
        avg["_n_gpus"] = n_gpus
        avg["_n_runs"] = n_runs
        cases.append(avg)

    baseline_total = sum(cases[0].get(k, 0.0) for k in PHASE_KEYS)
    headers = [col_header(c["_n_gpus"], gpus_per_node, c["_n_runs"]) for c in cases]

    print()
    print("**Startup overhead** (wall-clock from T0 to end of first batch, rank 0):\n")
    print(render_markdown_table(cases, headers, baseline_total))


if __name__ == "__main__":
    main()
