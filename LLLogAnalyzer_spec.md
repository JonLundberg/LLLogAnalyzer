# Spec v2: LLLogAnalyzer

## 1. Purpose

Build a deterministic command-line Python tool for analyzing LM Studio / llama.cpp developer logs, especially logs from GGUF MoE and hybrid models such as `qwen3.6-35b-a3b`.

The tool must extract and summarize:

- Model identity, quantization, context, and parallelism.
- GPU / CPU / CUDA host memory allocations.
- Parsed MoE / expert metadata.
- Inferred MoE / CPU-host placement, clearly marked as inference.
- KV cache size, type, context scaling, and CPU/GPU split.
- Recurrent-state memory.
- Compute-buffer memory.
- GPU layer offload and whole-layer CPU fallback.
- Graph splits.
- Prompt-eval and token-generation performance.
- Context-growth effects on generation speed.
- Deterministic tuning hints based only on parsed facts and fixed heuristics.

This tool must not use an LLM. The same input log and the same threshold configuration must always produce the same output.

## 2. Design Principles

1. **Parsed facts first.** Values directly found in the log must be reported separately from inferred values.
2. **No hidden guessing.** Any inferred field must include its inference method and confidence.
3. **Speed is primary for comparisons.** Memory placement, GPU residency, and graph splits explain speed; they do not override measured speed.
4. **Heuristics are not truth.** Grades are convenience labels relative to configurable thresholds, not objective model quality measurements.
5. **Partial logs are valid.** The tool must produce useful output from truncated or partial logs.
6. **Windows-first, cross-platform where practical.** The base CLI must work on Windows with Python 3.10+ and no required third-party dependencies.
7. **Machine-readable output must be stable.** JSON keys, float formatting, and nullability must be deterministic.

## 3. Target Platform

Primary target:

- Windows 10/11.
- Python 3.10+.
- PowerShell-friendly command line.
- No mandatory third-party dependencies.

Optional dependencies may be supported, but the base tool must work using only the Python standard library.

## 4. Program Name

Recommended script name:

```text
LLLogAnalyzer.py
```

Recommended package/module name:

```text
lm_log_analyzer
```

## 5. Terminology

### 5.1 CUDA device memory

Memory physically resident in GPU VRAM, usually reported as:

```text
CUDA0 model buffer size
CUDA0 KV buffer size
CUDA0 compute buffer size
```

### 5.2 CUDA host memory

Pinned or CUDA-associated host/system RAM. This is not VRAM. It may be used for CPU-MoE expert placement, CPU fallback, or CUDA scheduling.

Reported as:

```text
CUDA_Host model buffer size
CUDA_Host compute buffer size
CUDA_Host output buffer size
```

### 5.3 CPU mapped memory

CPU-side memory-mapped model data, often associated with mmap.

Reported as:

```text
CPU_Mapped model buffer size
```

### 5.4 Whole-layer CPU fallback

When the log reports fewer GPU-offloaded layers than total load units:

```text
offloaded 19/41 layers to GPU
```

The unoffloaded layer/load-unit count is considered whole-layer fallback for reporting purposes.

### 5.5 KV layers

The `N layers` printed in a KV cache line means the number of attention/KV-cache layers, not necessarily the total number of model layers. For hybrid models such as Mamba/attention mixtures, KV layers can be much lower than `n_layer`.

Example:

```text
llama_kv_cache: size = 1373.28 MiB (132352 cells,  10 layers,  1/1 seqs)
```

Do not use KV layer count for layer-offload ratio. Use it only for KV-cache reporting and context scaling.

## 6. Command-Line Interface

Use `argparse`.

```text
usage: LLLogAnalyzer.py [-h]
                                [--file FILE]
                                [--clipboard]
                                [--stdin]
                                [--markdown]
                                [--json]
                                [--out OUT]
                                [--compare FILE [FILE ...]]
                                [--thresholds THRESHOLDS_JSON]
                                [--encoding ENCODING]
                                [--verbose]
                                [input_file]
```

### 6.1 Input selection rules

1. If `--compare` is used, parse all listed files and ignore `--file`, positional input, stdin, and clipboard.
2. If `--clipboard` is used, read clipboard.
3. If `--file` is used, read that file.
4. If positional `input_file` is present, read that file.
5. If stdin is not a TTY, read stdin.
6. Otherwise, show help and return code `2`.

### 6.2 Encoding

Default encoding:

```text
utf-8
```

Use:

```text
--encoding utf-8
--encoding utf-16
--encoding cp1252
```

When reading text:

- Use the requested encoding.
- Use `errors="replace"` rather than failing on malformed text.
- Preserve enough line structure for parsing.

### 6.3 Exit codes

- `0`: success.
- `1`: parsed input but found warnings or no strong timing data.
- `2`: invalid usage, missing file, empty input, or clipboard unavailable.
- `3`: unexpected internal error.

## 7. Input Modes

### 7.1 File input

```powershell
python .\LLLogAnalyzer.py --file ".\Pasted text.txt"
```

Also support positional file input:

```powershell
python .\LLLogAnalyzer.py ".\Pasted text.txt"
```

### 7.2 Standard input

```powershell
Get-Content ".\Pasted text.txt" -Raw | python .\LLLogAnalyzer.py
```

### 7.3 Clipboard input

```powershell
python .\LLLogAnalyzer.py --clipboard
```

Clipboard implementation requirements:

1. First try Python standard-library `tkinter` clipboard access.
2. If that fails on Windows, call PowerShell:

```powershell
powershell.exe -NoProfile -Command "Get-Clipboard -Raw"
```

3. Decode subprocess output using `--encoding`, defaulting to UTF-8 with `errors="replace"`.
4. If PowerShell returns no text, returns non-zero, or cannot be launched, print a clear error and return code `2`.

README must document that the PowerShell clipboard fallback can be affected by locale, encoding, profile, execution environment, or clipboard provider behavior.

## 8. Output Modes

### 8.1 Human-readable report

Default mode:

```powershell
python .\LLLogAnalyzer.py --file log.txt
```

Output must be plain text suitable for terminal use.

### 8.2 Markdown report

```powershell
python .\LLLogAnalyzer.py --file log.txt --markdown
```

Markdown must use tables for memory buffers, timing samples, comparison summaries, and recommendations.

### 8.3 JSON report

```powershell
python .\LLLogAnalyzer.py --file log.txt --json
```

JSON requirements:

- Use `json.dumps(..., sort_keys=True, indent=2)`.
- Round float values to 2 decimal places unless the value is a ratio, in which case use 4 decimal places.
- Use `null` for missing values.
- Preserve stable key names across versions.
- Include a `schema_version` field.

### 8.4 Save report to file

```powershell
python .\LLLogAnalyzer.py --file log.txt --markdown --out report.md
python .\LLLogAnalyzer.py --file log.txt --json --out report.json
```

### 8.5 Compare multiple logs

```powershell
python .\LLLogAnalyzer.py --compare iq3_s.txt q4_k_xl.txt iq3_xxs.txt
```

Comparison mode must:

1. Parse each log independently.
2. Show all comparable timing samples by context band.
3. Choose a speed winner using measured generation speed as the primary metric.
4. Present GPU residency, CUDA host memory, graph splits, quant, and context as explanatory columns.
5. Warn when logs are not directly comparable due to different context bands.

Do not choose a winner solely from placement grade or graph splits.

## 9. Threshold Configuration

Grades and recommendations are based on configurable heuristic thresholds.

### 9.1 Default thresholds

The tool must include a built-in default threshold configuration.

### 9.2 External thresholds file

Support:

```powershell
python .\LLLogAnalyzer.py --file log.txt --thresholds .\thresholds.json
```

The thresholds file must allow overriding:

- Placement grade thresholds.
- Speed grade thresholds by context band.
- Prompt-eval grade thresholds.
- Context bands.
- Small-sample prompt-token threshold.
- Active-context conflict tolerance.
- Graph split warning thresholds.
- CUDA host model buffer warning thresholds.

Example:

```json
{
  "active_context_conflict_tolerance_tokens": 500,
  "small_prompt_sample_tokens": 256,
  "context_bands": {
    "small": [0, 16000],
    "medium": [16000, 40000],
    "large": [40000, 80000],
    "xlarge": [80000, 140000],
    "huge": [140000, null]
  },
  "generation_speed_grades": {
    "small": { "A": 20, "B": 14, "C": 8, "D": 5 },
    "medium": { "A": 20, "B": 14, "C": 8, "D": 5 },
    "large": { "A": 12, "B": 8, "C": 5, "D": 3 },
    "xlarge": { "A": 10, "B": 7, "C": 4, "D": 2 },
    "huge": { "A": 8, "B": 5, "C": 3, "D": 1.5 }
  }
}
```

Invalid threshold files must produce a clear error and return code `2`.

## 10. Data Model

Implement data classes. Every parsed or inferred value that may be uncertain should support source metadata.

### 10.1 SourceValue

Use a generic structure for fields that may be parsed or inferred:

```json
{
  "value": 8,
  "source": "parsed",
  "evidence": "print_info: n_expert_used = 8",
  "confidence": 1.0
}
```

Allowed `source` values:

- `parsed`
- `inferred`
- `derived`
- `default`
- `missing`

If missing:

```json
{
  "value": null,
  "source": "missing",
  "evidence": null,
  "confidence": 0.0
}
```

Use this structure at minimum for:

- MoE expert fields.
- Inferred quant from filename.
- Inferred CPU-MoE placement.
- Active context estimate.
- Derived offload ratio.
- Derived visible memory totals.
- Any recommendation that depends on inference rather than direct parsing.

### 10.2 Top-level report object

JSON top-level shape:

```json
{
  "schema_version": "2.0",
  "source": {},
  "model": {},
  "load": {},
  "hardware": {},
  "moe": {},
  "offload": {},
  "memory": {},
  "graph": {},
  "timings": [],
  "timing_summary": {},
  "analysis": {},
  "recommendations": [],
  "warnings": []
}
```

## 11. Parser Layer

The parser must be line-oriented and robust to:

- Timestamps before log content.
- Wrapped lines.
- Repeated load sections.
- Multiple request/response tasks.
- Partial logs.
- Truncated request bodies.
- Missing model-load sections.
- Missing timing sections.

### 11.1 Regex guidance

Compile regexes once.

Allow:

- Variable spaces.
- Optional commas in numbers.
- Decimal values.
- Units with KiB/MiB/GiB/TiB.
- Extra log prefixes.

Example helper behavior:

```python
def parse_mib(value: str, unit: str) -> Optional[float]:
    ...
```

Required behavior:

- Return `None` on failure.
- Do not raise for malformed input.
- Remove thousands separators from `value`.
- Supported units:
  - `B`
  - `KiB`
  - `MiB`
  - `GiB`
  - `TiB`
  - `KB`, `MB`, `GB`, `TB` may be accepted as base-2 aliases for practical log parsing.
- Convert all values to MiB internally.

### 11.2 Source evidence

For each parsed field, store the original line or a short evidence string where practical. This helps audit ambiguous values.

## 12. Parser: Required Fields

### 12.1 Model and load configuration

Parse:

- Model path.
- Model filename.
- Model display name if present.
- Quant name from filename.
- Architecture.
- Model type.
- Model parameter count.
- File type.
- File size.
- Declared BPW.
- Effective BPW.
- Context requested by LM Studio.
- Context actually constructed by llama.cpp.
- Context trained.
- Parallel count.
- Slot count.
- Batch size.
- Microbatch size.
- Flash attention enabled/disabled.
- KV unified enabled/disabled.
- mmap true/false.
- direct_io true/false.
- Prompt cache enabled and size limit.

Relevant line patterns:

```text
LlamaV4::load called with model path: ...
LlamaV4::load config: n_parallel=1 n_ctx=132107 kv_unified=true
print_info: arch                  = qwen35moe
print_info: model type            = 35B.A3B
print_info: model params          = 34.66 B
print_info: file type             = IQ3_S - 3.4375 bpw
print_info: file size             = 12.73 GiB (3.15 BPW)
print_info: n_ctx_train           = 262144
llama_context: n_ctx              = 132352
llama_context: n_batch            = 512
llama_context: n_ubatch           = 512
llama_context: flash_attn         = enabled
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = false)
srv    load_model: prompt cache is enabled, size limit: 8192 MiB
```

### 12.2 Hardware

Parse:

- Number of CUDA devices.
- Total VRAM.
- GPU name.
- Compute capability.
- VMM support.
- Device VRAM.
- Free VRAM at load start.

Relevant line patterns:

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):
Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB
using device CUDA0 (...) - 15009 MiB free
```

### 12.3 MoE / expert metadata

Parse:

- Expert count.
- Experts used.
- Expert groups.
- Group used.
- Expert feed-forward length.
- Shared expert feed-forward length.
- Override of `qwen35moe.expert_used_count`.

Relevant line patterns:

```text
qwen35moe.expert_count u32              = 256
qwen35moe.expert_used_count u32         = 8
validate_override: Using metadata override (  int) 'qwen35moe.expert_used_count' = 8
load_hparams: ----------------------- n_expert_used = 8 n_expert_groups = 0
print_info: n_expert              = 256
print_info: n_expert_used         = 8
```

Important: LM Studio may not print the exact “Force MoE layers to CPU” setting. If it is not explicitly visible, do not invent it.

Allowed inference:

```text
The allocation pattern is consistent with CPU-MoE / CUDA-host expert placement.
```

This must be represented as an inference:

```json
"cpu_moe_placement": {
  "value": "likely",
  "source": "inferred",
  "evidence": "CUDA_Host model buffer is present and large while model is MoE",
  "confidence": 0.6
}
```

### 12.4 Layer offload

Parse:

```text
load_tensors: offloading output layer to GPU
load_tensors: offloading 30 repeating layers to GPU
load_tensors: offloaded 31/41 layers to GPU
```

Extract:

- Output layer offloaded.
- Repeating layers offloaded.
- Offloaded load units.
- Total load units.

Derived:

- `offload_ratio = offloaded_layers / total_layers`.
- `whole_layer_cpu_fallback_count = total_layers - offloaded_layers`.
- `repeating_layers_cpu_fallback_count = n_layer - repeating_layers_offloaded`.

### 12.5 Model memory buffers

Parse all visible model memory buffers:

```text
load_tensors:          CPU model buffer size =   397.85 MiB
load_tensors:        CUDA0 model buffer size =  9044.30 MiB
load_tensors:    CUDA_Host model buffer size =  3590.51 MiB
load_tensors:   CPU_Mapped model buffer size =  3027.00 MiB
```

Normalize all memory to MiB.

Report:

- CPU model buffer.
- CPU mapped model buffer.
- CUDA model buffer.
- CUDA host model buffer.
- Total visible parsed model memory.
- Percent in CUDA device.
- Percent in CUDA host/system RAM.
- Percent in CPU mapped memory.

### 12.6 KV cache

Parse:

```text
llama_kv_cache:        CPU KV buffer size =   274.66 MiB
llama_kv_cache:      CUDA0 KV buffer size =  1098.62 MiB
llama_kv_cache: size = 1373.28 MiB (132352 cells,  10 layers,  1/1 seqs), K (q8_0):  686.64 MiB, V (q8_0):  686.64 MiB
```

Extract:

- CPU KV buffer MiB.
- CUDA KV buffer MiB.
- Total KV MiB.
- Number of cells.
- KV layers.
- Sequence info.
- K quantization type.
- V quantization type.
- K size.
- V size.

Derived:

- `kv_mib_per_1k_context = total_kv_mib / (ctx_cells / 1000)`.
- Projected KV at 64k, 128k, 196k, and 262k.
- CPU/GPU KV split.

Clarification:

- KV layers are attention-cache layers, not necessarily model layers.
- Do not use KV layers to compute offload ratio.

### 12.7 Recurrent-state memory

Parse:

```text
llama_memory_recurrent:        CPU RS buffer size =    16.75 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =    46.06 MiB
llama_memory_recurrent: size =   62.81 MiB (     1 cells,  40 layers,  1 seqs), R (f32):    2.81 MiB, S (f32):   60.00 MiB
```

Extract:

- CPU RS MiB.
- CUDA RS MiB.
- Total RS MiB.
- R type and size.
- S type and size.

### 12.8 Compute buffers

Parse:

```text
sched_reserve:      CUDA0 compute buffer size =   574.20 MiB
sched_reserve:  CUDA_Host compute buffer size =   278.88 MiB
```

Extract:

- CUDA compute buffer.
- CUDA host compute buffer.

### 12.9 Graph information

Parse:

```text
sched_reserve: graph nodes  = 3849
sched_reserve: graph splits = 220 (with bs=512), 22 (with bs=1)
```

Extract:

- Graph nodes.
- Graph splits per listed batch size.
- Graph splits for bs=1 when present.

Derived:

- `graph_splits_bs1_per_offloaded_layer` if offloaded layer count is known.
- `graph_splits_bs1_per_total_layer` if total layer count is known.

Interpretation:

- `bs=1` splits are more relevant for token generation.
- Large-batch graph splits are more relevant for prompt ingestion.
- Lower is generally better for the same architecture and similar placement.
- Raw graph split counts are not architecture-normalized. In comparison output, show raw splits and normalized splits, and label normalized splits as heuristic.

### 12.10 Timing blocks

Parse every timing block, including task IDs greater than zero:

```text
slot print_timing: id  0 | task 240 |
prompt eval time =     454.20 ms /   322 tokens (    1.41 ms per token,   708.94 tokens per second)
       eval time =    9246.91 ms /   179 tokens (   51.66 ms per token,    19.36 tokens per second)
      total time =    9701.11 ms /   501 tokens
```

Each timing sample must include:

- Slot id if available.
- Task id if available.
- Prompt eval ms.
- Prompt tokens.
- Prompt ms/token.
- Prompt tokens/sec.
- Eval ms.
- Eval tokens.
- Eval ms/token.
- Eval tokens/sec.
- Total ms.
- Total tokens.
- Approximate active context if available.
- Active-context estimation method.

Do not drop tasks with task id > 0.

### 12.11 Active context near timing samples

Associate each timing block with nearby preceding lines for the same slot/task when possible:

```text
slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 132352, n_keep = 11215, task.n_tokens = 63310
slot update_slots: id  0 | task 0 | prompt processing done, n_tokens = 63310, batch.n_tokens = 4
slot      release: id  0 | task 0 | stop processing: n_tokens = 64170, truncated = 0
```

Candidate estimates:

1. `stop_processing_n_tokens`.
2. `prompt_done_n_tokens + eval_tokens`.
3. `task.n_tokens + eval_tokens`.
4. `prompt_tokens + eval_tokens`.
5. `prompt_tokens`.

Default priority order:

1. Use `stop_processing_n_tokens` if present.
2. Else use `prompt_done_n_tokens + eval_tokens`.
3. Else use `task.n_tokens + eval_tokens`.
4. Else use `prompt_tokens + eval_tokens`.
5. Else use `prompt_tokens`.

Conflict rule:

- Compute all available estimates.
- If the selected estimate differs from another available estimate by more than the configured tolerance, default `500` tokens or `5%`, whichever is larger, add a warning to the timing sample.
- Do not change the priority order because of the conflict; instead report the conflict.

JSON field:

```json
"active_context": {
  "value": 64170,
  "source": "derived",
  "method": "stop_processing_n_tokens",
  "candidates": {
    "stop_processing_n_tokens": 64170,
    "prompt_done_plus_eval": 64170,
    "task_tokens_plus_eval": 64170
  },
  "conflict_warning": null
}
```

### 12.12 Prompt cache and checkpoints

Parse:

```text
srv    load_model: prompt cache is enabled, size limit: 8192 MiB
slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 8191, pos_max = 8191, n_tokens = 8192, size = 62.813 MiB)
slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.966 (> 0.100 thold), f_keep = 1.000
slot update_slots: id  0 | task 240 | cache reuse is not supported - ignoring n_cache_reuse = 256
```

Extract:

- Prompt cache enabled.
- Prompt cache limit.
- Checkpoint count.
- Checkpoint size.
- LCP similarity events.
- Cache reuse ignored messages.

Interpretation:

- Prompt-cache presence does not guarantee reuse.
- If the log says cache reuse is not supported, report it clearly.
- Checkpoint overhead may affect small prompt-delta timing samples.

## 13. Analysis Layer

### 13.1 Header disclaimer

Every human-readable and Markdown report must include a short disclaimer:

```text
Grades are deterministic heuristics based on this tool's thresholds, not absolute model-quality or hardware-quality measurements.
```

If a custom thresholds file is used, include its path.

### 13.2 Placement grade

Compute placement grade using configured thresholds.

Default suggested rules:

- A: offload ratio >= 0.75 and CUDA host model buffer <= 5120 MiB and bs=1 graph splits <= 25.
- B: offload ratio >= 0.70 and CUDA host model buffer <= 8192 MiB and bs=1 graph splits <= 35.
- C: offload ratio >= 0.45 and CUDA host model buffer <= 14336 MiB and bs=1 graph splits <= 50.
- D: offload ratio >= 0.30 or CUDA host model buffer <= 20480 MiB.
- F: otherwise.

The report must label this as heuristic.

### 13.3 Speed grade

Evaluate timing samples separately by context band.

Default context bands:

- `small`: < 16k active context.
- `medium`: 16k–40k.
- `large`: 40k–80k.
- `xlarge`: 80k–140k.
- `huge`: > 140k.

Generation-speed grade by band:

For `small` and `medium`:

- A: >= 20 tok/s.
- B: >= 14 tok/s.
- C: >= 8 tok/s.
- D: >= 5 tok/s.
- F: < 5 tok/s.

For `large`:

- A: >= 12 tok/s.
- B: >= 8 tok/s.
- C: >= 5 tok/s.
- D: >= 3 tok/s.
- F: < 3 tok/s.

For `xlarge`:

- A: >= 10 tok/s.
- B: >= 7 tok/s.
- C: >= 4 tok/s.
- D: >= 2 tok/s.
- F: < 2 tok/s.

For `huge`:

- A: >= 8 tok/s.
- B: >= 5 tok/s.
- C: >= 3 tok/s.
- D: >= 1.5 tok/s.
- F: < 1.5 tok/s.

Prompt-eval speed grade:

- A: >= 1500 tok/s.
- B: >= 1000 tok/s.
- C: >= 500 tok/s.
- D: >= 200 tok/s.
- F: < 200 tok/s.

If prompt tokens < configured `small_prompt_sample_tokens`, default `256`, label prompt speed as:

```text
small-sample / overhead-dominated
```

Do not use such prompt samples for best prompt-eval ranking.

### 13.4 Multi-task aggregation

For each log:

- Report all timing samples individually.
- Group timing samples by context band.
- For each band compute:
  - count.
  - best generation tok/s.
  - median generation tok/s.
  - worst generation tok/s.
  - best prompt tok/s, excluding small prompt samples.
  - median prompt tok/s, excluding small prompt samples.
- Overall summary should prefer median over best when more than two samples exist.
- Do not silently drop task ids greater than zero.

### 13.5 Long-context slowdown

If at least two timing samples exist in different context bands, estimate:

- Generation tok/s drop from smallest-context sample to largest-context sample.
- Approximate drop per 10k context tokens.
- Whether the slowdown is mild, moderate, or steep.

Use measured samples only. Do not hard-code model-specific curves.

### 13.6 Fit and headroom estimate

Compute visible parsed allocations:

- Visible CUDA memory = CUDA0 model + CUDA KV + CUDA RS + CUDA compute + CUDA output if available.
- Visible CUDA host memory = CUDA host model + CUDA host compute + CUDA host output.
- Visible CPU memory = CPU model + CPU mapped model + CPU KV + CPU RS.

Do not claim this equals total process memory. Label it:

```text
visible parsed allocations
```

Estimate visible VRAM utilization:

```text
estimated_visible_cuda_mib / reported_total_vram_mib
```

If free VRAM at load start is available, also report it.

## 14. Comparison Layer

### 14.1 Primary ranking

Generation speed is the sole primary ranking metric.

When comparing two runs:

1. Prefer samples in the same context band.
2. If multiple samples exist in the band, compare median generation tok/s.
3. If no overlapping context band exists, state that the runs are not directly comparable and avoid declaring a speed winner.
4. If the user requests an overall winner anyway, choose the run with the best median generation tok/s in the largest common lower-bound context band and label it as approximate.

### 14.2 Informational comparison columns

Comparison table must include:

- Filename.
- Model quant.
- Actual context.
- Context band.
- Median generation tok/s in band.
- Best generation tok/s in band.
- Median prompt eval tok/s.
- Offloaded layers.
- CUDA0 model buffer.
- CUDA host model buffer.
- KV total.
- Graph splits bs=1.
- Normalized graph splits per total layer, if possible.
- Placement grade.
- Speed grade.

### 14.3 Contradiction notes

If the speed winner has worse GPU residency, worse CUDA host buffer, or worse graph splits than another candidate, add a note:

```text
The speed winner has worse GPU residency than another run. This can happen because quantization, context depth, and architecture-specific scheduling affect measured speed. Measured generation speed is treated as primary.
```

### 14.4 Graph split comparison

Raw graph splits are not architecture-normalized.

Comparison mode must show:

- Raw bs=1 graph splits.
- bs=1 graph splits per total layer if total layer count is known.
- A footnote that normalized graph splits are heuristic and most useful when comparing similar model architectures.

## 15. Recommendations Engine

Recommendations must be deterministic and must include the reason and evidence.

### 15.1 Recommendation object

JSON shape:

```json
{
  "title": "Keep Parallel = 1",
  "severity": "info",
  "reason": "Single-slot workload and current placement is strong.",
  "evidence": ["n_parallel=1", "n_slots=1"]
}
```

Severity values:

- `info`
- `suggestion`
- `warning`

### 15.2 Recommendation examples

If `n_parallel > 1`:

```text
Recommendation: Set Parallel / Max Concurrent Predictions to 1.
Reason: n_parallel is currently 4, which increases memory/state overhead for a single interactive workload.
```

If context is low relative to train context and user wants more context:

```text
Recommendation: You can likely try a larger context.
Reason: current n_ctx is 64000 while model train context is 262144, and current KV cache is only 664 MiB.
```

If CUDA host model buffer is large and offload ratio is low:

```text
Recommendation: Prefer a smaller/lower quant or tune CPU-MoE only if it improves measured speed.
Reason: only 19/41 layers are offloaded and CUDA_Host model buffer is 11.7 GiB.
```

If graph splits bs=1 are high:

```text
Recommendation: Reduce cross-device split by improving GPU residency or choosing a lighter quant.
Reason: bs=1 graph splits are 72, which usually hurts generation in similar architectures.
```

If prompt eval is strong but generation is slow:

```text
Recommendation: Do not focus on batch size first; generation is the bottleneck.
Reason: prompt eval is 1290 tok/s, but generation is 7.5 tok/s.
```

If prompt eval is weak and prompt tokens >= 2048:

```text
Recommendation: Try increasing eval batch size from 512 to 1024.
Reason: prompt processing is a major part of runtime and the sample is large enough to be meaningful.
```

### 15.3 Recommendation safety

Bad:

```text
MoE CPU is set to 14.
```

Good:

```text
The log does not explicitly show the MoE CPU layer setting. The allocation pattern is consistent with a CPU-MoE / CUDA-host expert setup.
```

## 16. Renderer Layer

### 16.1 Human-readable report outline

```text
LLLogAnalyzer Report
==================================

Heuristic disclaimer
Source
Model
Load
Hardware
Placement
Memory
KV Cache
Graph
Timing Samples
Timing Summary
Long-Context Behavior
Recommendations
Warnings / Unknowns
```

### 16.2 Markdown report outline

Use headings and tables:

```markdown
# LLLogAnalyzer Report

> Grades are deterministic heuristics...

## Model
## Load
## Memory
## Timing Samples
## Timing Summary
## Recommendations
## Warnings
```

### 16.3 JSON report

Use the data model from section 10.

## 17. Example Human Report

```text
LLLogAnalyzer Report
==================================

Grades are deterministic heuristics based on this tool's thresholds, not absolute model-quality or hardware-quality measurements.

Model
-----
Model file: Qwen3.6-35B-A3B-UD-IQ3_S.gguf
Quant: IQ3_S
File size: 12.73 GiB
BPW: 3.4375 declared, 3.15 effective

Load
----
Context requested: 132107
Context actual: 132352
Context trained: 262144
Parallel: 1
Batch / microbatch: 512 / 512
Flash attention: enabled

Placement
---------
Offloaded: 31/41 layers (75.61%)
Repeating layers on GPU: 30/40
Whole-layer CPU fallback: 10 load units
Placement grade: A (heuristic)

Visible parsed model buffers:
  CUDA0 model:      9044.30 MiB
  CUDA host model:  3590.51 MiB
  CPU model:         397.85 MiB

KV Cache
--------
Total: 1373.28 MiB
CUDA0: 1098.62 MiB
CPU:    274.66 MiB
K/V: q8_0 / q8_0
KV layers: 10 attention-cache layers
KV per 1k context: 10.38 MiB

Graph
-----
Nodes: 3849
Splits: 220 with bs=512, 22 with bs=1

Timing Summary
--------------
Best large-context generation: 12.46 tok/s at ~64k active context
Prompt eval: 1857.04 tok/s over 63310 tokens

Recommendations
---------------
- Keep Parallel = 1.
- This is a strong large-context configuration.
- Test 196k context before 262k.
```

## 18. Tests

Use `unittest` by default.

### 18.1 Unit tests

Create tests for:

- Model path parsing.
- Load config parsing.
- Hardware parsing.
- File size and BPW parsing.
- SourceValue behavior.
- Layer offload parsing.
- Model buffer parsing.
- KV cache parsing.
- KV layer clarification.
- Recurrent-state parsing.
- Compute buffer parsing.
- Graph split parsing.
- Timing block parsing for task ids > 0.
- Active context association and conflict warning.
- Multi-task aggregation.
- Threshold config loading.
- JSON serialization with `sort_keys=True`.
- Float rounding.

### 18.2 Smoke fixtures

Small fixture tests are smoke tests only. They should not be treated as complete coverage.

#### IQ3_S 132k smoke fixture

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):
Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB
LlamaV4::load config: n_parallel=1 n_ctx=132107 kv_unified=true
print_info: file type   = IQ3_S - 3.4375 bpw
print_info: file size   = 12.73 GiB (3.15 BPW)
load_tensors: offloading 30 repeating layers to GPU
load_tensors: offloaded 31/41 layers to GPU
load_tensors:          CPU model buffer size =   397.85 MiB
load_tensors:        CUDA0 model buffer size =  9044.30 MiB
load_tensors:    CUDA_Host model buffer size =  3590.51 MiB
llama_context: n_ctx         = 132352
llama_kv_cache:        CPU KV buffer size =   274.66 MiB
llama_kv_cache:      CUDA0 KV buffer size =  1098.62 MiB
llama_kv_cache: size = 1373.28 MiB (132352 cells,  10 layers,  1/1 seqs), K (q8_0):  686.64 MiB, V (q8_0):  686.64 MiB
llama_memory_recurrent:        CPU RS buffer size =    16.75 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =    46.06 MiB
sched_reserve:      CUDA0 compute buffer size =   574.20 MiB
sched_reserve:  CUDA_Host compute buffer size =   278.88 MiB
sched_reserve: graph splits = 220 (with bs=512), 22 (with bs=1)
srv    load_model: prompt cache is enabled, size limit: 8192 MiB
slot print_timing: id  0 | task 0 |
prompt eval time =   34091.84 ms / 63310 tokens (    0.54 ms per token,  1857.04 tokens per second)
       eval time =   69005.05 ms /   860 tokens (   80.24 ms per token,    12.46 tokens per second)
      total time =  103096.89 ms / 64170 tokens
```

Expected assertions:

- Quant = `IQ3_S`.
- Actual context = `132352`.
- Offloaded layers = `31`.
- Total layers = `41`.
- CUDA model = `9044.30`.
- CUDA host model = `3590.51`.
- KV total = `1373.28`.
- KV layers = `10`.
- Graph splits bs=1 = `22`.
- Eval tok/s = `12.46`.
- Prompt cache enabled = true.

#### Q4_K_XL 64k smoke fixture

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):
Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB
LlamaV4::load config: n_parallel=1 n_ctx=64000 kv_unified=true
LlamaV4::load called with model path: C:\models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
print_info: file type   = Q4_K - Medium
print_info: file size   = 20.81 GiB (5.16 BPW)
load_tensors: offloading 18 repeating layers to GPU
load_tensors: offloaded 19/41 layers to GPU
load_tensors:        CUDA0 model buffer size =  9651.39 MiB
load_tensors:    CUDA_Host model buffer size = 11662.73 MiB
llama_context: n_ctx         = 64000
llama_kv_cache: size =  664.06 MiB ( 64000 cells,  10 layers,  1/1 seqs), K (q8_0):  332.03 MiB, V (q8_0):  332.03 MiB
sched_reserve: graph splits = 466 (with bs=512), 36 (with bs=1)
slot print_timing: id  0 | task 3933 |
prompt eval time =   34633.09 ms / 44710 tokens (    0.77 ms per token,  1290.96 tokens per second)
       eval time =  102848.98 ms /   773 tokens (  133.05 ms per token,     7.52 tokens per second)
      total time =  137482.06 ms / 45483 tokens
```

Expected assertions:

- Filename-derived quant = `Q4_K_XL`.
- Parsed file type = `Q4_K - Medium`.
- Offloaded = `19/41`.
- CUDA host model > CUDA0 model.
- Graph splits bs=1 = `36`.
- Eval tok/s = `7.52`.

### 18.3 Integration tests

Add integration tests with larger real-world snippets that include:

- Multiple timing blocks.
- Task IDs other than 0.
- Cache checkpoints.
- Cache reuse ignored messages.
- Partial/truncated request bodies.
- Missing load section.
- Missing timing section.

## 19. README Requirements

The implementation must include `README.md` with:

1. Project purpose.
2. Installation / prerequisites.
3. Quickstart:
   - File input.
   - Stdin input.
   - Clipboard input.
   - Markdown output.
   - JSON output.
   - Compare mode.
4. Example terminal output.
5. Explanation of:
   - CUDA0 memory.
   - CUDA host memory.
   - CPU mapped memory.
   - KV cache.
   - Graph splits.
   - Prompt eval vs generation speed.
6. Heuristic-grading disclaimer.
7. Threshold customization instructions.
8. Known limitations:
   - Windows-first clipboard behavior.
   - Partial/truncated log behavior.
   - LM Studio may not expose CPU-MoE layer count in logs.
   - Raw graph splits are not architecture-normalized.
   - Visible parsed memory is not total process memory.
9. Testing instructions:

```powershell
python -m unittest discover
```

## 20. Implementation Deliverables

The agent must produce:

1. `LLLogAnalyzer.py`
2. `README.md`
3. `tests/test_LLLogAnalyzer.py`
4. Optional `examples/` folder with sample snippets.
5. Optional `thresholds.default.json` if the defaults are externalized.

The tool must run with:

```powershell
python .\LLLogAnalyzer.py --clipboard
python .\LLLogAnalyzer.py --file ".\Pasted text.txt"
Get-Content ".\Pasted text.txt" -Raw | python .\LLLogAnalyzer.py
python .\LLLogAnalyzer.py --compare ".\iq3s.txt" ".\q4.txt"
python .\LLLogAnalyzer.py --file ".\iq3s.txt" --thresholds ".\thresholds.json"
python -m unittest discover
```

## 21. Acceptance Criteria

The implementation is complete when:

- It runs on Windows with Python 3.10+.
- It can read from file, stdin, and clipboard.
- It supports `--encoding`.
- It supports human, Markdown, and JSON output.
- JSON output is stable with sorted keys and rounded floats.
- It supports configurable thresholds via JSON.
- It extracts all major memory allocations from representative LM Studio logs.
- It extracts all timing blocks, not just the first one.
- It handles task IDs greater than 0.
- It reports prompt eval and generation speed separately.
- It associates timing samples with approximate active context when possible.
- It reports the active-context estimation method.
- It warns on active-context estimate conflicts.
- It identifies whole-layer CPU fallback from `offloaded N/M` lines.
- It reports graph splits and normalized graph-split heuristics.
- It emits deterministic recommendations with evidence.
- It can compare multiple logs using generation speed as the primary metric.
- It clearly separates parsed facts from inferred values.
- It never uses nondeterministic LLM logic.
- It handles partial/truncated logs gracefully.
- Tests pass.

## 22. Implementation Order

Implement incrementally in this order:

1. CLI input reading.
2. Data classes and `SourceValue`.
3. Basic model/load parser.
4. Memory parser.
5. Timing parser.
6. Active context association.
7. Analysis and summaries.
8. Renderers.
9. Compare mode.
10. Threshold config.
11. Clipboard fallback hardening.
12. Tests and README.
