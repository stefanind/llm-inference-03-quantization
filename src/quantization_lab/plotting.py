"""Plot saved comparison reports on CPU, without importing the model libraries."""

import csv
import json
from pathlib import Path


MIB = 1024 ** 2
STYLES = {
    "baseline": ("BF16 baseline", "#64748b"),
    "int8_per_tensor": ("INT8 / tensor", "#2563eb"),
    "int8_per_channel": ("INT8 / channel", "#059669"),
    "int4_per_channel_unpacked": ("INT4 / channel (unpacked)", "#d97706"),
    "int4_group128_unpacked": ("INT4 / group128 (unpacked)", "#7c3aed"),
    "int4_group128_packed": ("INT4 / group128 (packed)", "#db2777"),
    "torchao_int4_group128": ("TorchAO INT4 / group128", "#0891b2"),
}
WORKLOAD_METRICS = ("mean_prefill_ms", "decode_tokens_per_second", "peak_gpu_allocated_bytes")


def prepare_results(rows):
    """Separate once-per-variant values from measurements of each workload."""
    quality = {}
    workloads = {}
    for row in rows:
        variant = row["variant"]
        values = {
            "perplexity": float(row["perplexity"]),
            "model_tensor_bytes": int(row["model_tensor_bytes"]),
        }
        # These values repeat beside every workload; they are not extra observations.
        if variant in quality:
            assert quality[variant] == values, "Quality and persistent storage must agree across workloads."
        quality[variant] = values

        key = tuple(int(row[field]) for field in ("batch_size_requested", "prompt_length", "new_tokens"))
        case = workloads.setdefault(key, {})
        assert variant not in case, "Duplicate variant for one workload."
        measurements = {"status": row["status"]}
        for field in WORKLOAD_METRICS:
            measurements[field] = float(row[field]) if row[field] else None
        case[variant] = measurements
    assert quality, "comparison.csv contains no results."
    return quality, dict(sorted(workloads.items()))


def draw_bars(ax, variants, records, metric, label, divisor=1):
    """Draw only available values, and label missing or failed measurements."""
    labels = []
    for index, variant in enumerate(variants):
        name, color = STYLES.get(variant, (variant, "#475569"))
        labels.append(name)
        record = records.get(variant, {})
        status = record.get("status", "ok")
        value = record.get(metric) if status == "ok" else None
        if value is None:
            missing = "N/A" if status == "ok" else status.upper()
            ax.text(0.02, index, missing, transform=ax.get_yaxis_transform(), va="center", color="#64748b")
            continue
        value /= divisor
        ax.barh(index, value, color=color, height=0.65)
        text = f"{value:,.3f}" if metric == "perplexity" else f"{value:,.2f}"
        ax.annotate(text, (value, index), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)

    ax.set_yticks(range(len(variants)), labels)
    ax.set_ylim(len(variants) - 0.5, -0.5)
    ax.set_xlabel(label)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.2)
    ax.margins(x=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_pyplot():
    """Check plotting support before starting a potentially long comparison."""
    try:
        import matplotlib
    except ModuleNotFoundError:
        raise SystemExit("Install the project dependencies with: python -m pip install -e .") from None
    matplotlib.use("Agg")  # Write image files on RunPod without a graphical desktop.
    import matplotlib.pyplot as plt

    return plt


def plot_comparison(comparison_dir):
    """Save four figures beside an existing comparison.csv; return their folder."""
    plt = load_pyplot()
    comparison_dir = Path(comparison_dir)
    with (comparison_dir / "comparison.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    quality, workloads = prepare_results(rows)
    # Stable sorting keeps the baseline first and preserves the CSV's other variants.
    variants = sorted(quality, key=lambda variant: variant != "baseline")
    config = json.loads((comparison_dir / "config.json").read_text(encoding="utf-8"))
    environment_path = comparison_dir / rows[0]["run_dir"] / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    caption = (
        f"{config['model']['id']} | {environment['device_name']} | "
        f"{environment['dtype'].removeprefix('torch.')}"
    )
    output = comparison_dir / "plots"
    output.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
    fig.suptitle(f"Quality and persistent model storage\n{caption}", fontsize=13)
    draw_bars(axes[0], variants, quality, "perplexity", "Perplexity (lower is better)")
    draw_bars(axes[1], variants, quality, "model_tensor_bytes", "Persistent model MiB (lower is better)", MIB)
    for ax in axes:
        ax.set_xlim(left=0)
    fig.savefig(output / "quality_and_storage.png", dpi=150)
    plt.close(fig)

    figures = (
        ("prefill.png", "mean_prefill_ms", "Prefill time", "Milliseconds (lower is better)", 1),
        ("decode.png", "decode_tokens_per_second", "Decode throughput", "Total batch tokens/s (higher is better)", 1),
        ("peak_memory.png", "peak_gpu_allocated_bytes", "Peak GPU allocation during generation", "Allocated MiB (lower is better)", MIB),
    )
    columns = min(2, len(workloads))
    panel_rows = (len(workloads) + columns - 1) // columns
    for filename, metric, title, label, divisor in figures:
        fig, axes = plt.subplots(
            panel_rows, columns, figsize=(7 * columns, 4 * panel_rows + 1),
            squeeze=False, sharex=True, sharey=True, layout="constrained",
        )
        fig.suptitle(f"{title}\n{caption}", fontsize=13)
        for ax, (key, records) in zip(axes.flat, workloads.items()):
            batch, prompt, generated = key
            ax.set_title(f"Batch {batch} | Prompt {prompt} tokens | Generate {generated} tokens", fontsize=10)
            draw_bars(ax, variants, records, metric, label, divisor)
        for ax in list(axes.flat)[len(workloads):]:
            ax.set_visible(False)
        # Set the shared zero origin after every panel has contributed its data range.
        axes[0, 0].set_xlim(left=0)
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    return output
