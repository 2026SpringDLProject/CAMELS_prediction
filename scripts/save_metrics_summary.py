"""Aggregate per-model validation/test metrics from `make_figures.ipynb` cell
outputs and write a markdown summary. Source of truth: the notebook — re-run
its cells to refresh the numbers."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_DIR / "make_figures.ipynb"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "MODEL_METRICS_SUMMARY.md"

EXT_LINE = re.compile(
    r"^(?P<model>\w+)\s+TEST\s+\|\s+"
    r"RMSE=(?P<rmse>[-\d.]+)\s+MAE=(?P<mae>[-\d.]+)\s+"
    r"Global_NSE=(?P<global_nse>[-\d.]+)\s+"
    r"clipped_mean_basin_NSE=(?P<mean>[-\d.]+)\s+"
    r"clipped_median_basin_NSE=(?P<median>[-\d.]+)\s*$"
)
TFT_LINE = re.compile(
    r"^(?P<split>VAL|TEST)\s+performance\s+\|\s+"
    r"RMSE=(?P<rmse>[-\d.]+)\s+MAE=(?P<mae>[-\d.]+)\s+"
    r"R2=(?P<r2>[-\d.]+)\s+NSE=(?P<global_nse>[-\d.]+)\s+"
    r"clipped_mean_basin_NSE=(?P<mean>[-\d.]+)\s+"
    r"clipped_median_basin_NSE=(?P<median>[-\d.]+)\s*$"
)
CONFIG_RE = re.compile(r"--config\s+config/(\S+\.json)")
EXT_TYPE_RE = re.compile(r"--model-type\s+(\w+)")
EXT_CKPT_RE = re.compile(r"--checkpoint\s+(\S+)")

EXT_LABELS = {"vanilla": "Vanilla Transformer", "itransformer": "iTransformer"}


@dataclass
class SplitMetrics:
    rmse: float
    mae: float
    global_nse: float
    mean_basin_nse: float
    median_basin_nse: float


@dataclass
class ModelRun:
    label: str
    source_cell_index: int
    source_command: str
    splits: Dict[str, SplitMetrics] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _cell_text(cell: Dict) -> str:
    chunks: List[str] = []
    for out in cell.get("outputs", []):
        text = out.get("text")
        if text is None:
            continue
        if isinstance(text, list):
            text = "".join(text)
        chunks.append(text)
    return "\n".join(chunks)


def _label_for_tft_cell(src: str) -> str:
    cfg_match = CONFIG_RE.search(src)
    if not cfg_match:
        return "TFT/LSTM"
    cfg = cfg_match.group(1).lower()
    if "lstm" in cfg:
        return "LSTM baseline"
    return "TFT"


def _label_for_external_cell(src: str) -> str:
    mt = EXT_TYPE_RE.search(src)
    if not mt:
        return "External transformer"
    return EXT_LABELS.get(mt.group(1).lower(), mt.group(1))


def parse_notebook(nb_path: Path) -> List[ModelRun]:
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    runs: List[ModelRun] = []
    for idx, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "visualize_external_transformer_predictions" in src:
            run = ModelRun(
                label=_label_for_external_cell(src),
                source_cell_index=idx,
                source_command=src.strip().splitlines()[0],
            )
            text = _cell_text(cell)
            for line in text.splitlines():
                m = EXT_LINE.match(line.strip())
                if m:
                    run.splits["test"] = SplitMetrics(
                        rmse=float(m.group("rmse")),
                        mae=float(m.group("mae")),
                        global_nse=float(m.group("global_nse")),
                        mean_basin_nse=float(m.group("mean")),
                        median_basin_nse=float(m.group("median")),
                    )
            if "test" not in run.splits:
                run.notes.append(
                    "no metrics line in cell output — re-run the cell to populate"
                )
            runs.append(run)
        elif "visualize_tft_predictions" in src:
            run = ModelRun(
                label=_label_for_tft_cell(src),
                source_cell_index=idx,
                source_command=src.strip().splitlines()[0],
            )
            text = _cell_text(cell)
            for line in text.splitlines():
                m = TFT_LINE.match(line.strip())
                if m:
                    run.splits[m.group("split").lower()] = SplitMetrics(
                        rmse=float(m.group("rmse")),
                        mae=float(m.group("mae")),
                        global_nse=float(m.group("global_nse")),
                        mean_basin_nse=float(m.group("mean")),
                        median_basin_nse=float(m.group("median")),
                    )
            if "val" in run.splits:
                run.notes.append(
                    "val = last CV fold's val windows; the final model was retrained on "
                    "these (treat as in-sample)"
                )
            if not run.splits:
                run.notes.append(
                    "no metrics lines in cell output — re-run the cell to populate"
                )
            runs.append(run)
    return runs


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(runs: List[ModelRun], notebook_path: Path) -> str:
    rel_nb = notebook_path.relative_to(PROJECT_DIR.parent) if notebook_path.is_absolute() else notebook_path
    lines: List[str] = []
    lines.append("# Model Metrics Summary")
    lines.append("")
    lines.append(
        f"Parsed from `{rel_nb}` cell outputs. Per-basin mean / median NSE come from "
        "the visualization scripts' `clipped_*_basin_NSE` print lines (clipped at the "
        "p2 / p98 of finite per-basin NSEs, capped at [-1, 1]). Global NSE is computed "
        "on the flattened (basin × time) test vector against the global mean."
    )
    lines.append("")

    lines.append("## Test set")
    lines.append("")
    lines.append(
        "| Model | Median basin NSE | Mean basin NSE | Global NSE | RMSE | MAE |"
    )
    lines.append("|---|---|---|---|---|---|")
    for run in runs:
        sm = run.splits.get("test")
        if sm is None:
            lines.append(
                f"| {run.label} | n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            lines.append(
                f"| {run.label} "
                f"| {_fmt(sm.median_basin_nse)} "
                f"| {_fmt(sm.mean_basin_nse)} "
                f"| {_fmt(sm.global_nse)} "
                f"| {_fmt(sm.rmse, 3)} "
                f"| {_fmt(sm.mae, 3)} |"
            )
    lines.append("")

    has_val = any("val" in run.splits for run in runs)
    if has_val:
        lines.append("## Validation set (last CV fold)")
        lines.append("")
        lines.append(
            "| Model | Median basin NSE | Mean basin NSE | Global NSE | RMSE | MAE |"
        )
        lines.append("|---|---|---|---|---|---|")
        for run in runs:
            sm = run.splits.get("val")
            if sm is None:
                lines.append(f"| {run.label} | n/a | n/a | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {run.label} "
                    f"| {_fmt(sm.median_basin_nse)} "
                    f"| {_fmt(sm.mean_basin_nse)} "
                    f"| {_fmt(sm.global_nse)} "
                    f"| {_fmt(sm.rmse, 3)} "
                    f"| {_fmt(sm.mae, 3)} |"
                )
        lines.append("")
        lines.append(
            "*'val' is the last expanding-CV fold's val windows. Those windows were part "
            "of `pre_holdout` and therefore seen by the retrained final model — treat "
            "these numbers as in-sample, not held-out.*"
        )
        lines.append("")

    lines.append("## Per-cell sources")
    lines.append("")
    for run in runs:
        lines.append(f"- **{run.label}** (cell {run.source_cell_index}): `{run.source_command}`")
        for note in run.notes:
            lines.append(f"  - _{note}_")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.notebook.exists():
        print(f"Notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)

    runs = parse_notebook(args.notebook)
    if not runs:
        print("No model-runner cells found in the notebook.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(runs, args.notebook) + "\n", encoding="utf-8")
    print(f"Wrote summary: {args.output}", flush=True)
    for run in runs:
        splits = sorted(run.splits.keys()) or ["(none)"]
        print(f"  {run.label}: splits captured = {splits}", flush=True)
        for note in run.notes:
            print(f"    note: {note}", flush=True)


if __name__ == "__main__":
    main()
