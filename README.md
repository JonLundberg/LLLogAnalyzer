# LLLogAnalyzer

LLLogAnalyzer is a deterministic command-line analyzer for LM Studio and llama.cpp developer logs. It extracts model identity, memory placement, KV cache, graph splits, timing samples, context estimates, comparison summaries, and fixed heuristic recommendations without using an LLM.

## Requirements

- Python 3.10 or newer
- No required third-party packages
- Windows-first behavior, with standard stdin/file support on other platforms

## Installation

Run directly from a checkout:

```powershell
python .\LLLogAnalyzer.py --help
```

Or install the local package in editable mode:

```powershell
python -m pip install -e .
llloganalyzer --help
```

## Quickstart

Analyze a file:

```powershell
python .\LLLogAnalyzer.py --file ".\log.txt"
python .\LLLogAnalyzer.py ".\log.txt"
llloganalyzer --file ".\log.txt"
```

Analyze stdin:

```powershell
Get-Content ".\log.txt" -Raw | python .\LLLogAnalyzer.py
```

Analyze clipboard:

```powershell
python .\LLLogAnalyzer.py --clipboard
```

Show expanded timing diagnostics:

```powershell
python .\LLLogAnalyzer.py --clipboard --time
python .\LLLogAnalyzer.py ".\log.txt" -t
```

Show the full detailed text report:

```powershell
python .\LLLogAnalyzer.py --clipboard --verbose
python .\LLLogAnalyzer.py ".\log.txt" -v
```

Show speed by context size:

```powershell
python .\LLLogAnalyzer.py --file ".\log.txt" --context-table
python .\LLLogAnalyzer.py --file ".\log.txt" --speed-graph
```

Markdown and JSON output:

```powershell
python .\LLLogAnalyzer.py --file ".\log.txt" --markdown --out report.md
python .\LLLogAnalyzer.py --file ".\log.txt" --json --out report.json
```

Compare logs:

```powershell
python .\LLLogAnalyzer.py --compare ".\iq3s.txt" ".\q4.txt"
```

Use custom thresholds:

```powershell
python .\LLLogAnalyzer.py --file ".\log.txt" --thresholds ".\thresholds.json"
```

## Example Output

```text
Model: Qwen3.6-35B-A3B-UD-IQ3_S.gguf | Quant: IQ3_S | Context: 132352
Median generation: 12.46 tok/s (1 sample)
Average generation: 12.46 tok/s
High generation: 12.46 tok/s
Low generation: 12.46 tok/s
Prompt eval: 1857.04 tok/s
```

## Concepts

- `CUDA0` memory is parsed GPU device memory, usually VRAM.
- `CUDA_Host` memory is pinned or CUDA-associated host/system RAM, not VRAM.
- `CPU_Mapped` memory is CPU-side mapped model data, often from mmap.
- KV cache memory stores attention keys and values. KV layers are attention-cache layers, not necessarily total model layers.
- Graph splits describe scheduling partitions. Lower is generally better for similar architectures, but raw split counts are not architecture-normalized.
- Prompt eval speed measures prompt ingestion. Generation speed measures new-token decoding and is the primary metric in compare mode.
- Visible parsed memory is not total process memory; it is only the allocations explicitly visible in the log.

## Heuristics

Grades are deterministic convenience labels based on built-in or user-provided thresholds. They are not absolute model-quality or hardware-quality measurements. Parsed facts, derived values, and inferred values are separated in JSON output. Inferred fields include evidence and confidence.

Threshold JSON files can override placement grades, generation-speed grades by context band, prompt-eval grades, context bands, active-context conflict tolerance, graph split warnings, and CUDA host model-buffer warnings.

## Clipboard Notes

Clipboard mode first tries Python `tkinter`. On Windows it falls back to:

```powershell
powershell.exe -NoProfile -Command "Get-Clipboard -Raw"
```

That fallback can be affected by locale, encoding, profile, execution environment, or clipboard provider behavior.

## Privacy Notes

LM Studio developer logs can include local filesystem paths, usernames, prompts, request bodies, tool calls, and assistant responses. Review and sanitize logs before sharing them in issues, pull requests, or fixtures. Public tests in this repository use compact sanitized fixtures under `tests/fixtures/`.

## Known Limitations

- Partial or truncated logs produce partial reports and warnings.
- LM Studio may not expose the exact CPU-MoE layer count in logs. The analyzer reports CPU-MoE placement only as an inference when allocation evidence supports it.
- Raw graph splits are not architecture-normalized.
- Visible parsed allocations are not total process memory.
- Recommendations are fixed heuristics and should be validated with measured speed.

## Testing

```powershell
python -m unittest discover -v
```

## License

MIT. See [LICENSE](LICENSE).
