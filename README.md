# Quantization lab

A small project for learning weight quantization by implementing it and measuring
what changes.

The current milestone compares the floating-point Qwen2.5-0.5B baseline with
per-tensor INT8, per-output-channel INT8, and unpacked INT4 with either per-channel
scales or fixed groups of 128. Packed group-128 INT4 stores the same codes in half
as many bytes. A standalone matrix exercise explains the arithmetic first.
One optimized TorchAO group-128 INT4 variant provides a practical reference;
all variants use the same evaluation and benchmark runner.

## Setup: NVIDIA GPU through WSL/Linux

Use Python 3.11 or newer. Create the environment inside the WSL filesystem when
running GPU experiments there:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

This milestone pins **PyTorch 2.9.1 and TorchAO 0.15.0** as a compatible pair.
Install PyTorch's CUDA 12.8 build, then install this project:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA availability should print `True` before running the provided configs. They
explicitly request BF16 on CUDA; the runner does not silently substitute CPU or a
different precision. If BF16 is unavailable, select FP16 explicitly and keep that
baseline dtype consistent across manual comparisons. The optimized TorchAO variant
requires BF16 and an Ampere or newer NVIDIA GPU (SM80+), such as the RTX 3090.

## Setup: RunPod PyTorch image

For a regular GPU Pod, use a RunPod PyTorch image with **Python 3.11+ and PyTorch
2.9.1**, with CUDA 12.8 for this experiment. An installed `2.9.1+cu128` build
satisfies the project pin and is kept by the normal install. Other PyTorch versions
do not satisfy the pin and pip may replace them. Keep the exact image tag and GPU
type consistent across comparisons. TorchAO and plotting are included in
`python -m pip install -e .`.

Put a checkout or copy of this repository at
`/workspace/llm-inference-03-quantization`. Use the image's Python environment
directly; there is no need to create another virtual environment inside the Pod.
Do not copy a local `.venv` to a different machine or image.

```bash
cd /workspace/llm-inference-03-quantization
export HF_HOME=/workspace/.cache/huggingface

python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
python -m pip install -e .

python -m quantization_lab demo
python -m unittest discover -s tests -v
python -m quantization_lab run --config configs/smoke.json
```

`HF_HOME` puts the Hugging Face cache under `/workspace`; the default `results/`
directory is also there because the repository is there. Set `HF_HOME` again in a
new terminal, or add it to the Pod's environment variables. If the GPU reports
`BF16: False`, select `--dtype float16` explicitly for the baseline run.

## Start with the numerical exercise

```bash
python -m quantization_lab demo
python -m unittest discover -s tests -v
```

Both commands work offline on CPU. The demo compares the quantized variants on
the same ordinary and outlier matrices, including INT4 with fixed groups of 128.
Compare weight MSE and linear output MSE. `num_scales`, `scale_min`, and `scale_max`
show how precision is distributed without printing every row's scale. `bits` is
the numerical precision, while `storage_dtype` is the actual code storage type.

The demo uses a 64-by-256 FP32 matrix so every row contains two 128-weight groups.
The source weights use 65,536 bytes, while all unpacked code tensors use 16,384
bytes. `quantized_weight_and_scale_bytes` also includes the FP32 scales:

| Variant | Number of scales | Code and scale bytes |
| --- | --- | --- |
| Per-tensor INT8 | 1 | 16,388 |
| Per-channel INT8 | 64 | 16,640 |
| Unpacked per-channel INT4 | 64 | 16,640 |
| Unpacked group-128 INT4 | 128 | 16,896 |
| Packed group-128 INT4 | 128 | 8,704 |

The extra per-channel scales let unaffected rows keep their precision when
another row contains an outlier. Weights in the outlier's own row still share that
row's larger step size. Grouped scaling further limits that influence to one
128-weight group. Compare both error metrics: lower weight MSE does not guarantee
lower output MSE on the fixed inputs. The demo's FP32 source precision is separate
from the model's BF16 baseline.

Read [the symmetric quantizer](src/quantization_lab/quantizers/symmetric.py) first.
`quantize_weight` accepts `bits=4` or `bits=8`; `NaiveQuantizedLinear` reconstructs
a floating-point matrix on every call. The implementation uses floating-point
matrix multiplication for both precisions. Packed and unpacked group-128 INT4
have identical scales, weight MSE, and output MSE. Only their code storage changes:
16,384 bytes becomes 8,192 bytes, with the same 512 bytes of FP32 scales.

## Compare unpacked and packed group-128 INT4

```bash
python -m quantization_lab run --config configs/smoke.json --variant int4_group128_unpacked
python -m quantization_lab run --config configs/smoke.json --variant int4_group128_packed
```

The earlier `baseline`, `int8_per_tensor`, `int8_per_channel`, and
`int4_per_channel_unpacked` variants remain available. This iteration needs only
the two group-128 runs above, with the same configuration and hardware.

Each invocation loads the original floating-point checkpoint. All quantized variants
replace only linear layers inside Qwen's Transformer blocks. Embeddings,
normalization layers, and the output head remain unchanged, including their weight
sharing. Per-tensor INT8 stores one FP32 scale for the whole weight matrix.
Per-channel INT8 stores scales shaped `[out_features, 1]`, computed from the maximum
absolute weight in each output row. Broadcasting those scales across the input
columns reconstructs the matrix. Unpacked INT4 uses the same per-channel scale
layout, but computes `scale = row_max_abs / 7` and codes in `[-7, 7]` instead of
dividing by 127 and using `[-127, 127]`. This symmetric four-bit convention uses
15 of the 16 available bit patterns.

Group-128 INT4 reshapes each matrix from `[rows, columns]` to `[rows, groups, 128]`
to choose scales shaped `[rows, groups, 1]`. Unpacked codes retain the original
matrix shape. Reconstruction temporarily restores the grouped view, multiplies
by the scales, then returns to the original shape for `F.linear`.

The [packing helpers](src/quantization_lab/quantizers/packing.py) put the first code
in a byte's upper four bits and the second in its lower four bits. Negative codes
use four-bit two's complement: `[-7, 7]` becomes `1001 0111` (`0x97`). Unpacking
extracts the nibbles in order and subtracts 16 from values 8 through 15 to restore
their sign. Packing loses no information and introduces no additional rounding.

The packed layer keeps only a `torch.uint8` code buffer shaped
`[out_features, in_features / 2]`, plus the unchanged FP32 scales and any bias.
It unpacks codes into the original matrix shape on every forward, then follows
the same dequantization and floating-point `F.linear` path. It does not retain an
unpacked backup of the weights.

Only group size 128 is supported; there is no group-size argument or sweep.
Qwen2.5-0.5B's relevant input widths, 896 and 4,864, divide evenly by 128. A
divisibility assertion keeps the code simple, with no padding or partial groups.

All variants reconstruct weights in the input dtype before calling `F.linear`.
Unpacked INT4 keeps `qweight` as `torch.int8`. Per-channel INT4 therefore matches
per-channel INT8 storage, while group-128 INT4 adds more scales. Packing halves
the code bytes, not the scales or unconverted layers. This iteration measures the
storage savings and the execution cost of unpacking on every forward. This is a
storage exercise; there is no optimized integer matrix multiplication kernel.

Numerical checks run during quantization. Forward passes keep only metadata and
shape assertions, avoiding GPU synchronization from repeatedly checking scales.
The `dtype` in the environment report is the floating-point compute dtype; integer
code buffers and FP32 scales coexist with it. Smaller persistent storage does not
imply lower peak memory or faster inference. Unpacking creates temporary tensors
and adds work, so the packed variant may be slower.

The variant defaults to `baseline` if omitted. After the two INT4 smoke runs work,
use `--config configs/comparison.json` if a larger measurement sample is needed.
Keep the model revision, data, image, GPU, and workload settings identical between
the baseline and quantized runs.

## Run all variants together

```bash
python -m pip install -e .
python -m quantization_lab compare --config configs/comparison.json
```

This runs the baseline and all six quantized variants sequentially. Each gets a
fresh Python process, so the previous model and its CUDA allocations are released
before the next variant loads. `--device`, `--dtype`, and `--output` work as they
do for `run`. After all variants finish, the command automatically generates the
four comparison charts. Plotting support is checked first. A separate child process
then checks TorchAO's packed kernels on all four unique Qwen projection shapes,
with prefill and single-token inputs. This happens before downloading weights or
starting any measured experiment. Unsupported setups stop with an error. The full
comparison therefore requires CUDA and BF16; individual manual variants and the
numerical demo remain available on CPU.

Everything goes under one folder:

```text
results/<UTC timestamp>-comparison/
  config.json
  comparison.csv
  plots/
    quality_and_storage.png
    prefill.png
    decode.png
    peak_memory.png
  <UTC timestamp>-baseline/
  <UTC timestamp>-int8_per_tensor/
  <UTC timestamp>-int8_per_channel/
  <UTC timestamp>-int4_per_channel_unpacked/
  <UTC timestamp>-int4_group128_unpacked/
  <UTC timestamp>-int4_group128_packed/
  <UTC timestamp>-torchao_int4_group128/
```

Open `comparison.csv` for the combined view. Each row includes the variant,
perplexity, persistent model tensor bytes, batch size, prompt and generation
lengths, prefill time, decode throughput, peak GPU allocated bytes, and status.
Matching workloads appear together, with the baseline first. Byte counts retain
their original units; divide by 1,048,576 for MiB. Quality and persistent storage
are measured once per variant and repeated beside each workload's timing results.
The original JSON and CSV reports remain in each run folder.

The comparison is updated after each completed variant. Configuration, software,
hardware, resolved model revision, and token-sample metadata must match before a
run's rows are added. An OOM benchmark case retains its status and empty timing
fields. A failed run stops the command and leaves the completed results available.

## Optimized group-128 INT4 reference

To run only the optimized variant:

```bash
python -m quantization_lab run --config configs/comparison.json --variant torchao_int4_group128
```

The adapter uses TorchAO 0.15.0's `Int4WeightOnlyConfig` version 2, group size 128,
`TILE_PACKED_TO_4D` format, and `TINYGEMM` quantization parameters. It converts the
same 168 Transformer projections as the manual variants, preserving embeddings,
normalization, biases, the output head, and weight sharing. It uses the packed
`aten::_weight_int4pack_mm` operation with BF16 activations in eager execution.
It needs no calibration text or model compilation.

TorchAO's format differs from our manual symmetric `[-7, 7]` format: it uses
asymmetric codes in `[0, 15]`, BF16 scales and offsets, and kernel padding. Input
widths 896 and 4,864 are padded to 1,024 and 5,120; logical layer shapes are
preserved. Storage accounting visits the wrapper's physical code and qparam
tensors, including padded entries, rather than counting its logical BF16 shape.
This compares complete implementations, including their numerical choices.

Before evaluation, the optimized run profiles a short prefill and cached decode
outside benchmark timing. It requires one packed operation per converted
projection per forward, finite logits, and a working cache. `environment.json`
records the TorchAO version, packing recipe, padding, and verified operator count
under `quantization.kernel_check`. It also records `execution: "eager"` for all
variants. Smaller storage or faster execution is measured, not assumed.

You can run the initial shape check alone without downloading the model:

```bash
python -m quantization_lab.quantizers.torchao_int4 configs/comparison.json
```

The first comparison to review is BF16 versus manual packed group-128 INT4 versus
TorchAO group-128 INT4, using the existing comparison preset and its plots.

See the [TorchAO compatibility table](https://github.com/pytorch/ao/issues/2919)
and the [pinned packed-kernel implementation](https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quantize_/workflows/int4/int4_tile_packed_to_4d_tensor.py).

The first run downloads Qwen2.5-0.5B and WikiText-2 through Hugging Face. Later runs
can use its cache. A CPU check is possible with explicit overrides, although the
real model can be slow:

```bash
python -m quantization_lab run --config configs/smoke.json --device cpu --dtype float32
```

The runner evaluates quality once, then benchmarks every combination of batch
size, prompt length, and generation length. Change the lists in the config to add
cases; for example, `"new_tokens": [32, 128]`.

| Preset | Batch sizes | Prompt lengths | New tokens | Prompts per length | Evaluation input tokens |
| --- | --- | --- | --- | --- | --- |
| smoke | 1 | 128 | 32 | 4 | 8,192 |
| comparison | 1, 4 | 128, 512 | 32 | 16 | 65,536 |

Each case has one warmup per batch shape and three measured passes over its prompts. Run the
smoke preset during development and the comparison preset at milestones. Small
sample counts are useful engineering checks, not precise tail-latency estimates.

## Plot a completed comparison

`compare` creates the plots automatically. To regenerate them for an existing
comparison, point `plot` at its saved folder:

```bash
python -m quantization_lab plot results/<timestamp>-comparison
```

Replace `<timestamp>-comparison` with your actual folder name. Plotting runs on CPU
and reads existing reports; it does not load the model or repeat the experiments.
It also works in a RunPod terminal without a graphical desktop.

Four PNG files are written under that folder's `plots/` directory:

| File | Contents |
| --- | --- |
| `quality_and_storage.png` | Perplexity and persistent model MiB, once per variant |
| `prefill.png` | Prefill time in milliseconds, for each workload |
| `decode.png` | Total decode tokens/second across the batch, for each workload |
| `peak_memory.png` | Peak GPU allocated MiB during generation, for each workload |

The comparison preset produces four workload panels per timing or peak-memory
figure. Panels distinguish batch size, prompt length, and generation length, and
share a common scale within each figure. Variant colors stay consistent. OOM and
unavailable values are labeled `OOM` and `N/A`, with no measured bar. Quality and
persistent storage are not averaged over the repeated workload rows. Repeating
the plot command regenerates the four images from the current reports.

## Data and evaluation policy

We use **one fixed sample**, not separate training, validation, and calibration
datasets. No text is needed to choose the initial min/max weight scales.

- Source: [`Salesforce/wikitext`, `wikitext-2-raw-v1`](https://huggingface.co/datasets/Salesforce/wikitext), test split.
- Join rows in corpus order with two newlines; add no special tokens.
- Take the configured token prefix. Save a token hash and dataset fingerprint.
- For perplexity, slide a 512-token context by 256 tokens. Overlap supplies context
  but its targets are not scored twice. The first corpus token has no preceding
  context; every other selected token is scored exactly once.
- Accumulate summed FP32 cross-entropy and divide by the actual scored-token count
  before exponentiating. Never average per-window perplexities.
- Select disjoint, equally spaced prompt windows from that same token sample.
  Their IDs are token offsets. Reuse identical inputs across model variants.

This measures relative degradation on one corpus, not general task accuracy.
Windowing is deliberately fixed; see the
[Transformers perplexity guide](https://huggingface.co/docs/transformers/en/perplexity)
for why the context policy matters.

Both configs initially use `main` revisions. A comparison resolves the model
config's Hub cache path to an exact commit in the baseline child, then saves that
commit in its shared config for subsequent variants. Each run loads the model and
tokenizer from the same commit and records it in `environment.json`, including when Transformers leaves its own
`_commit_hash` unset. Individual runs also resolve and save their model revision.
Local model directories have no Hub commit. Before a milestone comparison, pin
the dataset revision to a Hugging Face commit as well. The saved token hash and
fingerprint check that every variant receives identical evaluation inputs.

## Measurement definitions

- **Persistent model tensor bytes:** unique parameter and buffer storage, including
  shared embeddings only once. Quantized tensor wrappers contribute their physical
  packed codes and qparams, including padding. This excludes the KV cache and is
  not file size.
- **Peak GPU allocated bytes:** PyTorch's peak allocation for each measured batch,
  reset after warmup. Includes resident model tensors, inputs, KV cache, and
  temporary tensors. Excludes allocator-reserved slack and non-PyTorch allocations.
  It is `null` on CPU.
- **Prefill milliseconds:** a forward over the entire prompt plus greedy selection
  of the first generated token. This is model timing, not serving TTFT.
- **Decode throughput:** tokens produced by the remaining `N-1` forward calls,
  across the whole batch, divided by their synchronized wall time. Summary rates
  use total tokens / total time, not an average of rates.

Tokenization and input transfer happen outside timing. Every batch starts with a
fresh KV cache. Prompts have exact lengths, so there is no padding. A final partial
batch gets its own warmup and records its actual size. Generation always produces the requested number
of tokens and ignores EOS for controlled work. Generation settings from the model
are not used. Quality evaluation uses no KV cache.

The benchmark uses ordinary eager forwards with PyTorch SDPA attention, not
`torch.compile`, CUDA graphs, or concurrent serving. An out-of-memory workload is
recorded as `oom`; other cases continue. Failures in setup or quality evaluation
stop the run instead of yielding an apparently valid comparison.

## Results and code map

Each run creates `results/<UTC timestamp>-<variant>/`:

```text
config.json           actual settings, including variant and CLI overrides
environment.json      software, device, revision, converted-layer count, tensor bytes, input hash
evaluation.json       perplexity, mean loss, scored tokens, window settings
measurements.jsonl    individual batch measurements and prompt IDs
summary.csv           one row per workload case
```

`models.py` loads the model, converts its linear layers, and counts storage;
`data.py` prepares text and prompts; `evaluate.py` scores quality; `benchmark.py`
times generation; `results.py` writes plain files. `cli.py` connects those functions
with a straightforward loop.

Fourteen focused tests use generated tensors and randomly initialized Qwen
model. Numerical examples check rounding, scales, and zero rows at both precisions.
One group-128 numerical test covers multiple groups, rounding, and a zero group.
One packing test checks known bytes and exact recovery of every emitted INT4 code.
The conversion integration test runs all five quantized variants, checks preserved
layers and tied weights, and exercises cached decoding. It verifies that packing
preserves prefill and decode outputs from the same original weights and saves
exactly half the code bytes. It also checks that per-channel INT4 and INT8 use the
same bytes and grouping adds scales.
One plotting preparation test checks quality deduplication, workload separation,
and preservation of OOM or unavailable measurements. It does not render images.
One TorchAO storage test checks actual wrapper fields, padding, and shared views
on CPU. One CUDA integration test uses a single Qwen block with the real projection
widths, checks preserved layers and biases, and verifies packed dispatch during
prefill and cached decode. This test is skipped when CUDA is unavailable.
Tests download no weights or data; their timings and perplexities are not
model-quality or GPU-performance results.

