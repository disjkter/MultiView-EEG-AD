"""Profile model size, approximate MACs, and single-window inference latency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.CEEDNet import CEEDNet1DResNet18  # noqa: E402
from models.EEGConformer import EEGConformer  # noqa: E402
from models.EEGNet import EEGNet  # noqa: E402
from models.Spatial_MSCNN import SpatialMSCNN_DualStream  # noqa: E402
from models.TFCBaseline import TFCClassifier  # noqa: E402


def build_models(num_channels: int, seq_len: int) -> dict[str, nn.Module]:
    """Build the neural models used in the matched CAUEEG comparison."""

    return {
        "EEGNet": EEGNet(num_channels=num_channels, num_classes=2, dropout=0.6),
        "EEG Conformer": EEGConformer(num_channels=num_channels, num_classes=2, dropout=0.6),
        "CEEDNet": CEEDNet1DResNet18(num_channels=num_channels, num_classes=2, dropout=0.6),
        "TF-C": TFCClassifier(
            num_classes=2,
            num_channels=num_channels,
            seq_len=seq_len,
            model_dim=128,
            projection_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.6,
        ),
        "Proposed framework": SpatialMSCNN_DualStream(
            num_classes=2,
            num_channels=num_channels,
            spatial_out_dim=256,
            temporal_out_dim=256,
            spatial_hidden_channels=64,
            spatial_temporal_channels=128,
            classifier_num_layers=2,
            classifier_dropout=0.6,
            use_neuro_prior=False,
        ),
    }


def approximate_macs(model: nn.Module, example: torch.Tensor) -> int:
    """Count Conv1d, Linear, and multihead-attention multiply-accumulates."""

    total = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(
        module: nn.Conv1d | nn.Conv2d,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal total
        kernel_elements = math.prod(module.kernel_size) if isinstance(module.kernel_size, tuple) else int(module.kernel_size)
        operations_per_output = (module.in_channels // module.groups) * kernel_elements
        total += int(output.numel() * operations_per_output)

    def linear_hook(module: nn.Linear, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        total += int(output.numel() * module.in_features)

    def attention_hook(
        module: nn.MultiheadAttention,
        inputs: tuple[torch.Tensor, ...],
        _output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        nonlocal total
        query = inputs[0]
        if module.batch_first:
            batch, tokens, embedding = query.shape
        else:
            tokens, batch, embedding = query.shape
        total += int(4 * batch * tokens * embedding * embedding)
        total += int(2 * batch * tokens * tokens * embedding)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(attention_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    # Keep gradient mode enabled here so older PyTorch versions do not bypass
    # MultiheadAttention modules through an inference-only fused fast path.
    model(example)
    for handle in handles:
        handle.remove()
    return total


def latency_ms(
    model: nn.Module,
    example: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    """Return median and interquartile range of single-window latency."""

    model.eval()
    is_cuda = example.is_cuda
    with torch.inference_mode():
        for _ in range(warmup):
            model(example)
        if is_cuda:
            torch.cuda.synchronize(example.device)
        timings: list[float] = []
        for _ in range(repetitions):
            if is_cuda:
                torch.cuda.synchronize(example.device)
            start = time.perf_counter()
            model(example)
            if is_cuda:
                torch.cuda.synchronize(example.device)
            timings.append((time.perf_counter() - start) * 1000.0)
    quartiles = statistics.quantiles(timings, n=4, method="inclusive")
    return float(statistics.median(timings)), float(quartiles[2] - quartiles[0])


def main() -> None:
    """Profile all models and export JSON, CSV, Markdown, and LaTeX tables."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="outputs/jbhi_revision/model_complexity")
    parser.add_argument("--num-channels", type=int, default=19)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(2026)
    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
    models = build_models(args.num_channels, args.seq_len)
    cpu_example = torch.randn(1, args.num_channels, args.seq_len)
    rows: list[dict[str, object]] = []
    for name, model in models.items():
        model.eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
        macs = approximate_macs(model, cpu_example)
        cpu_median, cpu_iqr = latency_ms(
            model,
            cpu_example,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        gpu_median: float | None = None
        gpu_iqr: float | None = None
        if torch.cuda.is_available():
            gpu_model = model.to("cuda")
            gpu_example = cpu_example.to("cuda")
            gpu_median, gpu_iqr = latency_ms(
                gpu_model,
                gpu_example,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            model.to("cpu")
            torch.cuda.empty_cache()
        rows.append(
            {
                "method": name,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "state_size_mb": state_bytes / (1024**2),
                "approx_macs_million": macs / 1e6,
                "cpu_latency_median_ms": cpu_median,
                "cpu_latency_iqr_ms": cpu_iqr,
                "gpu_latency_median_ms": gpu_median,
                "gpu_latency_iqr_ms": gpu_iqr,
            }
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "input_shape": [1, args.num_channels, args.seq_len],
                "cpu_threads": args.cpu_threads,
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "mac_definition": "Conv1d, Linear, and multihead-attention MACs; FFT, normalization, activation, and pooling operations excluded.",
                "models": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| Method | Parameters | State size (MB) | Approx. MACs (M) | CPU latency (ms) | GPU latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    latex_rows: list[str] = []
    for row in rows:
        gpu = row["gpu_latency_median_ms"]
        gpu_text = "--" if gpu is None else f"{float(gpu):.3f}"
        markdown.append(
            f"| {row['method']} | {int(row['parameters']):,} | {float(row['state_size_mb']):.2f} | "
            f"{float(row['approx_macs_million']):.2f} | {float(row['cpu_latency_median_ms']):.3f} | {gpu_text} |"
        )
        escaped_name = str(row["method"])
        latex_rows.append(
            f"{escaped_name} & {int(row['parameters']) / 1e6:.3f} & "
            f"{float(row['approx_macs_million']):.2f} & "
            f"{float(row['cpu_latency_median_ms']):.3f} & {gpu_text} \\\\"
        )
    prefix.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    prefix.with_suffix(".tex").write_text(
        "\\begin{table*}[t]\n"
        "\\centering\\small\n"
        "\\caption{Model complexity for one $19\\times128$ EEG window. MACs include convolution, linear, and attention operations.}\n"
        "\\label{tab:complexity}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Method & Params (M) & MACs (M) & CPU (ms) & GPU (ms) \\\\\n"
        "\\midrule\n"
        + "\n".join(latex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
