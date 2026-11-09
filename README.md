# Quantization lab

A small project for learning weight quantization by implementing it and measuring
what changes.

The current milestone compares the floating-point Qwen2.5-0.5B baseline with
per-tensor and per-output-channel INT8 weights in its Transformer linear layers.
A standalone matrix exercise explains the arithmetic first. There are no advanced
quantization methods or serving frameworks in this implementation yet.

## Setup: NVIDIA GPU through WSL/Linux

Use Python 3.11 or newer. Create the environment inside the WSL filesystem when
running GPU experiments there:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build using the command for your machine from
[PyTorch's installation page](https://pytorch.org/get-started/locally/), then install
this project:

```bash
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA availability should print `True` before running the provided configs. They
explicitly request BF16 on CUDA; the runner does not silently substitute CPU or a
different precision. If BF16 is unavailable, select FP16 explicitly and keep that
baseline dtype consistent across later comparisons.

## Setup: RunPod PyTorch image

For a regular GPU Pod, use a RunPod PyTorch image with **Python 3.11+ and PyTorch
>=2.9,<3**. Older images may cause pip to replace their bundled PyTorch to satisfy
this project's dependencies. Pick a compatible image and keep its exact tag and
GPU type consistent across comparisons.

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

Both commands work offline on CPU. The demo quantizes one matrix with both scale
granularities, then repeats the comparison with a single large outlier. Compare
weight MSE and linear output MSE. `num_scales`, `scale_min`, and `scale_max` show
how the precision is distributed without printing every row's scale.

The 64-by-128 FP32 source matrix uses 32,768 bytes. Its INT8 values use 8,192 bytes:
per-tensor scaling adds one four-byte scale (8,196 bytes total), while per-channel
scaling adds 64 scales (8,448 bytes total). The extra scales let unaffected rows
keep their precision when another row contains an outlier. Weights in the outlier's
own row still share that row's larger step size. The demo's FP32 source precision
is separate from the model's BF16 baseline.

Read [the INT8 implementation](src/quantization_lab/quantizers/int8.py) first.
`NaiveInt8Linear` stores integer weights but reconstructs a floating-point matrix
on every call. This saves persistent storage; it is not an INT8 matrix multiply.

## Compare baseline and INT8

```bash
python -m quantization_lab run --config configs/smoke.json --variant baseline
python -m quantization_lab run --config configs/smoke.json --variant int8_per_tensor
python -m quantization_lab run --config configs/smoke.json --variant int8_per_channel
```

Each invocation loads the original floating-point checkpoint. Both INT8 variants
replaces only linear layers inside Qwen's Transformer blocks. Embeddings,
normalization layers, and the output head remain unchanged, including their weight
sharing. Per-tensor INT8 stores one FP32 scale for the whole weight matrix.
Per-channel INT8 stores scales shaped `[out_features, 1]`, computed from the maximum
absolute weight in each output row. Broadcasting those scales across the input
columns reconstructs the matrix. Both variants reconstruct weights in the input
dtype before calling `F.linear`; neither performs an integer matrix multiply.

Numerical checks run during quantization. Forward passes keep only metadata and
shape assertions, avoiding GPU synchronization from repeatedly checking scales.
The `dtype` in the environment report is the floating-point compute dtype; INT8
weights and FP32 scales coexist with it. Smaller persistent storage does not imply
lower peak memory or faster inference.

The variant defaults to `baseline` if omitted. After all three smoke runs work,
repeat them with `--config configs/comparison.json` for a larger measurement sample.
Keep the model revision, data, image, GPU, and workload settings identical between
the baseline and INT8 runs.

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

Both configs initially use `main` revisions. Before a milestone comparison, replace
the model revision with the resolved commit in `environment.json` and pin the
dataset revision to a Hugging Face commit as well. Keep that config unchanged for
all variants. The saved token hash lets you detect accidental input changes.

## Measurement definitions

- **Persistent model tensor bytes:** unique parameter and buffer storage, including
  shared embeddings only once. This excludes the KV cache and is not file size.
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

Eight focused tests use generated tensors and a tiny randomly initialized Qwen
model. One numerical test checks known per-channel scales, rounding, and a zero
row. The existing conversion integration test runs both granularities and checks
preserved layers and tied weights, INT8 storage, and working cached decoding.
Tests download no weights or data; their timings and perplexities are not
model-quality or GPU-performance results.

