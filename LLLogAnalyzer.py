#!/usr/bin/env python3
"""LLLogAnalyzer: deterministic LM Studio / llama.cpp log analyzer.

This script intentionally uses only Python's standard library so it can run on
Windows with a stock Python 3.10+ installation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "2.0"


@dataclass
class SourceValue:
    value: Any = None
    source: str = "missing"
    evidence: Optional[str] = None
    confidence: float = 0.0

    @classmethod
    def parsed(cls, value: Any, evidence: str) -> "SourceValue":
        return cls(value=value, source="parsed", evidence=evidence.strip(), confidence=1.0)

    @classmethod
    def inferred(cls, value: Any, evidence: str, confidence: float) -> "SourceValue":
        return cls(value=value, source="inferred", evidence=evidence.strip(), confidence=confidence)

    @classmethod
    def derived(cls, value: Any, evidence: str, confidence: float = 1.0) -> "SourceValue":
        return cls(value=value, source="derived", evidence=evidence.strip(), confidence=confidence)

    @classmethod
    def default(cls, value: Any, evidence: str) -> "SourceValue":
        return cls(value=value, source="default", evidence=evidence.strip(), confidence=1.0)

    @classmethod
    def missing(cls) -> "SourceValue":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


class UsageError(Exception):
    """Expected CLI/input error."""


def source_dict(value: Any = None, source: str = "missing", evidence: Optional[str] = None, confidence: float = 0.0) -> Dict[str, Any]:
    return SourceValue(value=value, source=source, evidence=evidence, confidence=confidence).to_dict()


def missing_source() -> Dict[str, Any]:
    return SourceValue.missing().to_dict()


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "active_context_conflict_tolerance_tokens": 500,
    "small_prompt_sample_tokens": 256,
    "context_bands": {
        "small": [0, 16000],
        "medium": [16000, 40000],
        "large": [40000, 80000],
        "xlarge": [80000, 140000],
        "huge": [140000, None],
    },
    "generation_speed_grades": {
        "small": {"A": 20, "B": 14, "C": 8, "D": 5},
        "medium": {"A": 20, "B": 14, "C": 8, "D": 5},
        "large": {"A": 12, "B": 8, "C": 5, "D": 3},
        "xlarge": {"A": 10, "B": 7, "C": 4, "D": 2},
        "huge": {"A": 8, "B": 5, "C": 3, "D": 1.5},
    },
    "prompt_eval_speed_grades": {"A": 1500, "B": 1000, "C": 500, "D": 200},
    "placement_grade_thresholds": {
        "A": {"offload_ratio": 0.75, "cuda_host_model_mib": 5120, "graph_splits_bs1": 25},
        "B": {"offload_ratio": 0.70, "cuda_host_model_mib": 8192, "graph_splits_bs1": 35},
        "C": {"offload_ratio": 0.45, "cuda_host_model_mib": 14336, "graph_splits_bs1": 50},
        "D": {"offload_ratio": 0.30, "cuda_host_model_mib": 20480},
    },
    "graph_split_warning_threshold_bs1": 35,
    "graph_split_high_threshold_bs1": 50,
    "cuda_host_model_warning_mib": 8192,
}


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_thresholds(path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not path:
        return json.loads(json.dumps(DEFAULT_THRESHOLDS)), None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            overrides = json.load(handle)
    except OSError as exc:
        raise UsageError(f"Unable to read thresholds file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(f"Invalid thresholds JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise UsageError("Invalid thresholds JSON: top-level value must be an object")
    thresholds = deep_merge(DEFAULT_THRESHOLDS, overrides)
    validate_thresholds(thresholds)
    return thresholds, path


def validate_thresholds(thresholds: Dict[str, Any]) -> None:
    required = [
        "active_context_conflict_tolerance_tokens",
        "small_prompt_sample_tokens",
        "context_bands",
        "generation_speed_grades",
        "prompt_eval_speed_grades",
        "placement_grade_thresholds",
    ]
    for key in required:
        if key not in thresholds:
            raise UsageError(f"Invalid thresholds JSON: missing {key}")
    if not isinstance(thresholds["context_bands"], dict):
        raise UsageError("Invalid thresholds JSON: context_bands must be an object")
    for name, bounds in thresholds["context_bands"].items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise UsageError(f"Invalid thresholds JSON: context band {name!r} must be [lower, upper]")
        lower, upper = bounds
        if not isinstance(lower, (int, float)) or (upper is not None and not isinstance(upper, (int, float))):
            raise UsageError(f"Invalid thresholds JSON: context band {name!r} has invalid bounds")


def parse_number(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", "").strip())
    except (AttributeError, ValueError):
        return None


def parse_int(value: str) -> Optional[int]:
    number = parse_number(value)
    if number is None:
        return None
    return int(number)


def parse_mib(value: str, unit: str) -> Optional[float]:
    number = parse_number(value)
    if number is None:
        return None
    unit_key = unit.strip()
    factors = {
        "B": 1 / (1024 * 1024),
        "KiB": 1 / 1024,
        "KB": 1 / 1024,
        "MiB": 1,
        "MB": 1,
        "GiB": 1024,
        "GB": 1024,
        "TiB": 1024 * 1024,
        "TB": 1024 * 1024,
    }
    factor = factors.get(unit_key)
    if factor is None:
        return None
    return number * factor


def bool_from_text(value: str) -> Optional[bool]:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "enabled", "1"}:
        return True
    if lowered in {"false", "no", "disabled", "0"}:
        return False
    return None


def filename_from_path(path: str) -> str:
    return re.split(r"[\\/]", path.rstrip())[ -1]


def infer_quant_from_filename(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    stem = re.sub(r"\.gguf$", "", filename, flags=re.IGNORECASE)
    matches = re.findall(r"(IQ\d+_[A-Z0-9_]+|Q\d+_K(?:_[A-Z0-9]+)?|Q\d+_[A-Z0-9_]+)", stem.upper())
    if not matches:
        return None
    # Prefer the last quant-looking token because model names often contain other numeric tokens.
    return matches[-1].rstrip("_")


def quant_from_file_type(file_type: Optional[str]) -> Optional[str]:
    if not file_type:
        return None
    match = re.search(r"\b(IQ\d+_[A-Z0-9_]+|Q\d+_K(?:_[A-Z0-9]+)?|Q\d+_[A-Z0-9_]+)\b", file_type.upper())
    if match:
        return match.group(1).rstrip("_")
    return None


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def median(values: Iterable[float]) -> Optional[float]:
    items = [v for v in values if v is not None]
    if not items:
        return None
    return float(statistics.median(items))


class Regexes:
    LOAD_PATH = re.compile(r"LlamaV\d+::load called with model path:\s*(?P<path>.+)$")
    LOAD_CONFIG = re.compile(r"LlamaV\d+::load config:\s*(?P<config>.+)$")
    # raw llama.cpp (no LM Studio wrapper) fallbacks
    LOAD_MODEL_PATH = re.compile(r"load_model:\s*loading model\s*'(?P<path>[^']+)'")
    NEW_SLOT_CTX = re.compile(r"new slot,\s*n_ctx\s*=\s*(?P<ctx>[0-9,]+)", re.IGNORECASE)
    N_CTX_SEQ = re.compile(r"n_ctx_seq\s*\(\s*(?P<ctx>[0-9,]+)\s*\)", re.IGNORECASE)
    N_CTX_TRAIN = re.compile(r"n_ctx_train\s*\(\s*(?P<train>[0-9,]+)\s*\)", re.IGNORECASE)
    CONFIG_KV = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")
    PRINT_INFO = re.compile(r"print_info:\s*(?P<key>[^=]+?)\s*=\s*(?P<value>.+)$")
    FILE_SIZE = re.compile(
        r"(?P<size>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)"
        r"(?:\s*\((?P<bpw>[0-9.]+)\s*BPW\))?",
        re.IGNORECASE,
    )
    DECLARED_BPW = re.compile(r"(?P<bpw>[0-9.]+)\s*bpw", re.IGNORECASE)
    CUDA_INIT = re.compile(
        r"ggml_cuda_init:\s*found\s*(?P<count>\d+)\s*CUDA devices\s*"
        r"\(Total VRAM:\s*(?P<vram>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)\)",
        re.IGNORECASE,
    )
    DEVICE = re.compile(
        r"Device\s*(?P<index>\d+):\s*(?P<name>.*?),\s*compute capability\s*(?P<cc>[0-9.]+),\s*"
        r"VMM:\s*(?P<vmm>yes|no),\s*VRAM:\s*(?P<vram>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)",
        re.IGNORECASE,
    )
    USING_DEVICE = re.compile(
        r"using device\s*(?P<device>CUDA\d+).*?-\s*(?P<free>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)\s*free",
        re.IGNORECASE,
    )
    META_FIELD = re.compile(r"(?P<name>[A-Za-z0-9_.]+)\s+\w+\s*=\s*(?P<value>[0-9,]+)")
    OVERRIDE = re.compile(r"validate_override:.*?'(?P<name>[^']+)'\s*=\s*(?P<value>[0-9,]+)")
    LOAD_HPARAMS_EXPERT = re.compile(r"\b(?P<name>n_expert(?:_used|_groups)?)\s*=\s*(?P<value>[0-9,]+)")
    OUTPUT_OFFLOAD = re.compile(r"offloading output layer to GPU", re.IGNORECASE)
    REPEATING_OFFLOAD = re.compile(r"offloading\s*(?P<count>[0-9,]+)\s*repeating layers to GPU", re.IGNORECASE)
    OFFLOADED = re.compile(r"offloaded\s*(?P<off>[0-9,]+)\s*/\s*(?P<total>[0-9,]+)\s*layers to GPU", re.IGNORECASE)
    BUFFER = re.compile(
        r"(?P<area>load_tensors|llama_kv_cache|llama_memory_recurrent|sched_reserve):\s*"
        r"(?P<where>[A-Za-z0-9_]+)\s+(?P<kind>model|KV|RS|compute|output)\s+buffer size\s*=\s*"
        r"(?P<size>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)",
        re.IGNORECASE,
    )
    KV_SIZE = re.compile(
        r"llama_kv_cache:\s*size\s*=\s*(?P<total>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)\s*"
        r"\(\s*(?P<cells>[0-9,]+)\s*cells,\s*(?P<layers>[0-9,]+)\s*layers,\s*(?P<seqs>[^)]+?)\s*\),\s*"
        r"K\s*\((?P<ktype>[^)]+)\):\s*(?P<k_size>[0-9.,]+)\s*(?P<k_unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB),\s*"
        r"V\s*\((?P<vtype>[^)]+)\):\s*(?P<v_size>[0-9.,]+)\s*(?P<v_unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)",
        re.IGNORECASE,
    )
    RS_SIZE = re.compile(
        r"llama_memory_recurrent:\s*size\s*=\s*(?P<total>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)\s*"
        r"\(\s*(?P<cells>[0-9,]+)\s*cells,\s*(?P<layers>[0-9,]+)\s*layers,\s*(?P<seqs>[0-9,]+)\s*seqs?\s*\),\s*"
        r"R\s*\((?P<rtype>[^)]+)\):\s*(?P<r_size>[0-9.,]+)\s*(?P<r_unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB),\s*"
        r"S\s*\((?P<stype>[^)]+)\):\s*(?P<s_size>[0-9.,]+)\s*(?P<s_unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)",
        re.IGNORECASE,
    )
    GRAPH_NODES = re.compile(r"graph nodes\s*=\s*(?P<nodes>[0-9,]+)", re.IGNORECASE)
    GRAPH_SPLITS = re.compile(r"graph splits\s*=\s*(?P<splits>.+)$", re.IGNORECASE)
    GRAPH_SPLIT_ITEM = re.compile(r"(?P<count>[0-9,]+)\s*\(with bs=(?P<bs>[0-9,]+)\)", re.IGNORECASE)
    PROMPT_CACHE = re.compile(
        r"prompt cache is (?P<state>enabled|disabled)(?:,\s*size limit:\s*(?P<size>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB))?",
        re.IGNORECASE,
    )
    CHECKPOINT = re.compile(
        r"created context checkpoint\s*(?P<index>[0-9,]+)\s*of\s*(?P<count>[0-9,]+).*?"
        r"n_tokens\s*=\s*(?P<tokens>[0-9,]+),\s*size\s*=\s*(?P<size>[0-9.,]+)\s*(?P<unit>B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)",
        re.IGNORECASE,
    )
    LCP = re.compile(r"selected slot by LCP similarity,\s*sim_best\s*=\s*(?P<sim>[0-9.]+)", re.IGNORECASE)
    CACHE_REUSE_IGNORED = re.compile(r"cache reuse is not supported.*?n_cache_reuse\s*=\s*(?P<reuse>[0-9,]+)", re.IGNORECASE)
    SLOT_TASK = re.compile(r"slot\s+\w+:\s*id\s*(?P<slot>-?[0-9,]+)\s*\|\s*task\s*(?P<task>-?[0-9,]+)\s*\|", re.IGNORECASE)
    TASK_TOKENS = re.compile(r"task\.n_tokens\s*=\s*(?P<tokens>[0-9,]+)", re.IGNORECASE)
    PROMPT_DONE = re.compile(r"prompt processing done,\s*n_tokens\s*=\s*(?P<tokens>[0-9,]+)", re.IGNORECASE)
    STOP_PROCESSING = re.compile(r"stop processing:\s*n_tokens\s*=\s*(?P<tokens>[0-9,]+)", re.IGNORECASE)
    TIMING_HEADER = re.compile(r"slot\s+print_timing:\s*id\s*(?P<slot>-?[0-9,]+)\s*\|\s*task\s*(?P<task>-?[0-9,]+)\s*\|", re.IGNORECASE)
    PROMPT_EVAL = re.compile(
        r"prompt eval time\s*=\s*(?P<ms>[0-9.,]+)\s*ms\s*/\s*(?P<tokens>[0-9,]+)\s*tokens\s*"
        r"\(\s*(?P<mspt>[0-9.,]+)\s*ms per token,\s*(?P<tps>[0-9.,]+)\s*tokens per second\)",
        re.IGNORECASE,
    )
    EVAL = re.compile(
        r"\beval time\s*=\s*(?P<ms>[0-9.,]+)\s*ms\s*/\s*(?P<tokens>[0-9,]+)\s*tokens\s*"
        r"\(\s*(?P<mspt>[0-9.,]+)\s*ms per token,\s*(?P<tps>[0-9.,]+)\s*tokens per second\)",
        re.IGNORECASE,
    )
    TOTAL = re.compile(r"total time\s*=\s*(?P<ms>[0-9.,]+)\s*ms\s*/\s*(?P<tokens>[0-9,]+)\s*tokens", re.IGNORECASE)


def empty_report(source_name: Optional[str], thresholds_path: Optional[str]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "name": source_name,
            "thresholds_path": thresholds_path,
        },
        "model": {
            "path": None,
            "filename": None,
            "display_name": None,
            "quant": missing_source(),
            "architecture": None,
            "model_type": None,
            "model_params": None,
            "file_type": None,
            "file_size_mib": None,
            "file_size_display": None,
            "declared_bpw": None,
            "effective_bpw": None,
            "n_layer": None,
            "context_trained": None,
        },
        "load": {
            "context_requested": None,
            "context_actual": None,
            "parallel": None,
            "slots": None,
            "batch": None,
            "microbatch": None,
            "flash_attention": None,
            "kv_unified": None,
            "mmap": None,
            "direct_io": None,
            "prompt_cache": {
                "enabled": None,
                "limit_mib": None,
                "checkpoints": [],
                "lcp_similarity_events": [],
                "cache_reuse_ignored": [],
            },
        },
        "hardware": {
            "cuda_device_count": None,
            "total_vram_mib": None,
            "devices": [],
            "free_vram_at_load_mib": None,
        },
        "moe": {
            "expert_count": missing_source(),
            "experts_used": missing_source(),
            "expert_groups": missing_source(),
            "group_used": missing_source(),
            "expert_ffn_length": missing_source(),
            "expert_shared_ffn_length": missing_source(),
            "expert_used_override": missing_source(),
            "cpu_moe_placement": missing_source(),
        },
        "offload": {
            "output_layer_offloaded": None,
            "repeating_layers_offloaded": None,
            "offloaded_layers": None,
            "total_layers": None,
            "offload_ratio": missing_source(),
            "whole_layer_cpu_fallback_count": None,
            "repeating_layers_cpu_fallback_count": None,
        },
        "memory": {
            "model_buffers_mib": {},
            "kv_buffers_mib": {},
            "rs_buffers_mib": {},
            "compute_buffers_mib": {},
            "output_buffers_mib": {},
            "visible_model_total_mib": missing_source(),
            "visible_cuda_model_percent": None,
            "visible_cuda_host_model_percent": None,
            "visible_cpu_mapped_model_percent": None,
            "visible_allocations_mib": {
                "cuda": missing_source(),
                "cuda_host": missing_source(),
                "cpu": missing_source(),
            },
            "visible_cuda_vram_ratio": missing_source(),
        },
        "kv_cache": {
            "total_mib": None,
            "cells": None,
            "layers": None,
            "seqs": None,
            "k_type": None,
            "v_type": None,
            "k_mib": None,
            "v_mib": None,
            "mib_per_1k_context": None,
            "projected_mib": {},
            "cpu_gpu_split": {},
            "note": "KV layers are attention-cache layers, not total model layers.",
        },
        "recurrent_state": {
            "total_mib": None,
            "cells": None,
            "layers": None,
            "seqs": None,
            "r_type": None,
            "s_type": None,
            "r_mib": None,
            "s_mib": None,
        },
        "graph": {
            "nodes": None,
            "splits": {},
            "splits_bs1": None,
            "splits_bs1_per_offloaded_layer": None,
            "splits_bs1_per_total_layer": None,
            "note": "Raw graph splits are not architecture-normalized; normalized splits are heuristic.",
        },
        "timings": [],
        "timing_summary": {},
        "analysis": {
            "placement_grade": None,
            "placement_grade_note": "heuristic",
            "long_context_behavior": None,
            "fit_and_headroom": {},
        },
        "recommendations": [],
        "warnings": [],
    }


class LogParser:
    def __init__(self, text: str, source_name: Optional[str], thresholds: Dict[str, Any], thresholds_path: Optional[str] = None) -> None:
        self.lines = text.splitlines()
        self.thresholds = thresholds
        self.report = empty_report(source_name, thresholds_path)
        self.context_by_slot_task: Dict[Tuple[int, int], Dict[str, int]] = {}
        self.context_by_slot: Dict[int, Dict[str, int]] = {}

    def parse(self) -> Dict[str, Any]:
        self._timings_by_key: Dict[Tuple[Optional[int], Optional[int]], Dict[str, Any]] = {}
        self._timing_order: List[Tuple[Optional[int], Optional[int]]] = []
        self._last_timing_key: Optional[Tuple[Optional[int], Optional[int]]] = None
        for line in self.lines:
            self._parse_line(line)
            header = Regexes.TIMING_HEADER.search(line)
            if header:
                # New llama.cpp / llama-server format prints a print_timing header on
                # every timing line, including the metric lines themselves. Accumulate
                # metrics per (slot, task) rather than scanning a fixed block window.
                slot = parse_int(header.group("slot"))
                task = parse_int(header.group("task"))
                key = (slot, task)
                self._last_timing_key = key
                self._apply_timing_metrics(self._timing_for_key(key), line)
                continue
            if self._is_timing_metric_line(line):
                # Old format: metric lines follow a headerless block header. Attach them
                # to the most recently seen timing block.
                key = self._last_timing_key or (None, None)
                self._apply_timing_metrics(self._timing_for_key(key), line)
        self.report["timings"] = [
            timing
            for timing in (self._timings_by_key[key] for key in self._timing_order)
            if timing["prompt_eval_ms"] is not None
            or timing["eval_ms"] is not None
            or timing["total_ms"] is not None
        ]
        self._finalize()
        return self.report

    def _is_timing_metric_line(self, line: str) -> bool:
        return bool(Regexes.PROMPT_EVAL.search(line) or Regexes.EVAL.search(line) or Regexes.TOTAL.search(line))

    def _timing_for_key(self, key: Tuple[Optional[int], Optional[int]]) -> Dict[str, Any]:
        timing = self._timings_by_key.get(key)
        if timing is None:
            timing = self._blank_timing(key[0], key[1])
            self._timings_by_key[key] = timing
            self._timing_order.append(key)
        return timing

    @staticmethod
    def _blank_timing(slot: Optional[int], task: Optional[int]) -> Dict[str, Any]:
        return {
            "slot_id": slot,
            "task_id": task,
            "prompt_eval_ms": None,
            "prompt_tokens": None,
            "prompt_ms_per_token": None,
            "prompt_tokens_per_second": None,
            "eval_ms": None,
            "eval_tokens": None,
            "eval_ms_per_token": None,
            "eval_tokens_per_second": None,
            "total_ms": None,
            "total_tokens": None,
            "active_context": {},
            "context_band": None,
            "generation_speed_grade": None,
            "prompt_eval_grade": None,
            "warnings": [],
        }

    def _apply_timing_metrics(self, timing: Dict[str, Any], line: str) -> None:
        # "prompt eval time" also contains "eval time", so check the prompt form first.
        prompt = Regexes.PROMPT_EVAL.search(line)
        if prompt:
            timing["prompt_eval_ms"] = parse_number(prompt.group("ms"))
            timing["prompt_tokens"] = parse_int(prompt.group("tokens"))
            timing["prompt_ms_per_token"] = parse_number(prompt.group("mspt"))
            timing["prompt_tokens_per_second"] = parse_number(prompt.group("tps"))
            return
        eval_match = Regexes.EVAL.search(line)
        if eval_match:
            timing["eval_ms"] = parse_number(eval_match.group("ms"))
            timing["eval_tokens"] = parse_int(eval_match.group("tokens"))
            timing["eval_ms_per_token"] = parse_number(eval_match.group("mspt"))
            timing["eval_tokens_per_second"] = parse_number(eval_match.group("tps"))
            return
        total = Regexes.TOTAL.search(line)
        if total:
            timing["total_ms"] = parse_number(total.group("ms"))
            timing["total_tokens"] = parse_int(total.group("tokens"))

    def _parse_line(self, line: str) -> None:
        self._parse_model_and_load(line)
        self._parse_hardware(line)
        self._parse_moe(line)
        self._parse_offload(line)
        self._parse_buffers(line)
        self._parse_kv_size(line)
        self._parse_rs_size(line)
        self._parse_graph(line)
        self._parse_prompt_cache(line)
        self._parse_context_event(line)

    def _parse_model_and_load(self, line: str) -> None:
        match = Regexes.LOAD_PATH.search(line)
        if match:
            path = match.group("path").strip()
            self.report["model"]["path"] = path
            self.report["model"]["filename"] = filename_from_path(path)
            return

        match = Regexes.LOAD_MODEL_PATH.search(line)
        if match and self.report["model"]["path"] is None:
            path = match.group("path").strip()
            self.report["model"]["path"] = path
            self.report["model"]["filename"] = filename_from_path(path)
            return

        match = Regexes.LOAD_CONFIG.search(line)
        if match:
            for kv in Regexes.CONFIG_KV.finditer(match.group("config")):
                key = kv.group("key")
                value = kv.group("value")
                if key == "n_parallel":
                    self.report["load"]["parallel"] = parse_int(value)
                elif key == "n_ctx":
                    self.report["load"]["context_requested"] = parse_int(value)
                elif key == "kv_unified":
                    self.report["load"]["kv_unified"] = bool_from_text(value)
                elif key in {"n_slots", "slots"}:
                    self.report["load"]["slots"] = parse_int(value)
            return

        match = Regexes.PRINT_INFO.search(line)
        if match:
            key = normalize_key(match.group("key"))
            value = match.group("value").strip()
            model = self.report["model"]
            load = self.report["load"]
            if key == "arch":
                model["architecture"] = value
            elif key == "model_type":
                model["model_type"] = value
            elif key == "model_params":
                model["model_params"] = value
            elif key == "file_type":
                model["file_type"] = value
                bpw = Regexes.DECLARED_BPW.search(value)
                if bpw:
                    model["declared_bpw"] = parse_number(bpw.group("bpw"))
            elif key == "file_size":
                file_match = Regexes.FILE_SIZE.search(value)
                if file_match:
                    model["file_size_mib"] = parse_mib(file_match.group("size"), normalized_unit(file_match.group("unit")))
                    model["file_size_display"] = f"{file_match.group('size')} {file_match.group('unit')}"
                    if file_match.group("bpw"):
                        model["effective_bpw"] = parse_number(file_match.group("bpw"))
            elif key == "n_ctx_train":
                model["context_trained"] = parse_int(value)
            elif key == "n_layer":
                model["n_layer"] = parse_int(value)
            elif key == "n_expert":
                self.report["moe"]["expert_count"] = SourceValue.parsed(parse_int(value), line).to_dict()
            elif key == "n_expert_used":
                self.report["moe"]["experts_used"] = SourceValue.parsed(parse_int(value), line).to_dict()
            elif key == "n_expert_groups":
                self.report["moe"]["expert_groups"] = SourceValue.parsed(parse_int(value), line).to_dict()
            elif key == "n_group_used":
                self.report["moe"]["group_used"] = SourceValue.parsed(parse_int(value), line).to_dict()
            elif key in {"n_batch", "batch"}:
                load["batch"] = parse_int(value)
            elif key in {"n_ubatch", "microbatch"}:
                load["microbatch"] = parse_int(value)
            elif key == "flash_attn":
                load["flash_attention"] = bool_from_text(value)
            elif key in {"name", "general_name"}:
                model["display_name"] = value
            return

        if "llama_context:" in line:
            ctx_match = re.search(r"n_ctx\s*=\s*([0-9,]+)", line)
            seq_match = Regexes.N_CTX_SEQ.search(line)
            train_match = Regexes.N_CTX_TRAIN.search(line)
            batch_match = re.search(r"n_batch\s*=\s*([0-9,]+)", line)
            ubatch_match = re.search(r"n_ubatch\s*=\s*([0-9,]+)", line)
            flash_match = re.search(r"flash_attn\s*=\s*(enabled|disabled|true|false)", line, re.IGNORECASE)
            if ctx_match:
                self.report["load"]["context_actual"] = parse_int(ctx_match.group(1))
            elif seq_match and self.report["load"]["context_actual"] is None:
                # raw llama.cpp prints "n_ctx_seq (N)" instead of "n_ctx = N"
                self.report["load"]["context_actual"] = parse_int(seq_match.group("ctx"))
            if train_match and self.report["model"]["context_trained"] is None:
                self.report["model"]["context_trained"] = parse_int(train_match.group("train"))
            if batch_match:
                self.report["load"]["batch"] = parse_int(batch_match.group(1))
            if ubatch_match:
                self.report["load"]["microbatch"] = parse_int(ubatch_match.group(1))
            if flash_match:
                self.report["load"]["flash_attention"] = bool_from_text(flash_match.group(1))

        # raw llama.cpp: "new slot, n_ctx = N" is the authoritative allocated context
        slot_ctx = Regexes.NEW_SLOT_CTX.search(line)
        if slot_ctx and self.report["load"]["context_actual"] is None:
            self.report["load"]["context_actual"] = parse_int(slot_ctx.group("ctx"))

        # llama-server: "initializing, n_slots = N, n_ctx_slot = N, kv_unified = 'true'"
        if "n_ctx_slot" in line:
            ctx_slot = re.search(r"n_ctx_slot\s*=\s*([0-9,]+)", line)
            if ctx_slot and self.report["load"]["context_actual"] is None:
                self.report["load"]["context_actual"] = parse_int(ctx_slot.group(1))
            n_slots = re.search(r"n_slots\s*=\s*([0-9,]+)", line)
            if n_slots and self.report["load"]["slots"] is None:
                self.report["load"]["slots"] = parse_int(n_slots.group(1))
            kv_unified = re.search(r"kv_unified\s*=\s*'?(true|false|yes|no)'?", line, re.IGNORECASE)
            if kv_unified and self.report["load"]["kv_unified"] is None:
                self.report["load"]["kv_unified"] = bool_from_text(kv_unified.group(1))

        if "load_tensors:" in line and "mmap" in line:
            mmap_match = re.search(r"mmap\s*=\s*(true|false|yes|no)", line, re.IGNORECASE)
            direct_match = re.search(r"direct_io\s*=\s*(true|false|yes|no)", line, re.IGNORECASE)
            if mmap_match:
                self.report["load"]["mmap"] = bool_from_text(mmap_match.group(1))
            if direct_match:
                self.report["load"]["direct_io"] = bool_from_text(direct_match.group(1))

    def _parse_hardware(self, line: str) -> None:
        match = Regexes.CUDA_INIT.search(line)
        if match:
            self.report["hardware"]["cuda_device_count"] = parse_int(match.group("count"))
            self.report["hardware"]["total_vram_mib"] = parse_mib(match.group("vram"), normalized_unit(match.group("unit")))
            return

        match = Regexes.DEVICE.search(line)
        if match:
            self.report["hardware"]["devices"].append(
                {
                    "index": parse_int(match.group("index")),
                    "name": match.group("name").strip(),
                    "compute_capability": match.group("cc"),
                    "vmm": bool_from_text(match.group("vmm")),
                    "vram_mib": parse_mib(match.group("vram"), normalized_unit(match.group("unit"))),
                }
            )
            return

        match = Regexes.USING_DEVICE.search(line)
        if match:
            self.report["hardware"]["free_vram_at_load_mib"] = parse_mib(match.group("free"), normalized_unit(match.group("unit")))

    def _parse_moe(self, line: str) -> None:
        match = Regexes.META_FIELD.search(line)
        if match:
            name = match.group("name")
            value = parse_int(match.group("value"))
            moe = self.report["moe"]
            if name.endswith(".expert_count"):
                moe["expert_count"] = SourceValue.parsed(value, line).to_dict()
            elif name.endswith(".expert_used_count"):
                moe["experts_used"] = SourceValue.parsed(value, line).to_dict()
            elif name.endswith(".expert_feed_forward_length"):
                moe["expert_ffn_length"] = SourceValue.parsed(value, line).to_dict()
            elif name.endswith(".expert_shared_feed_forward_length"):
                moe["expert_shared_ffn_length"] = SourceValue.parsed(value, line).to_dict()

        match = Regexes.OVERRIDE.search(line)
        if match and match.group("name").endswith("expert_used_count"):
            self.report["moe"]["expert_used_override"] = SourceValue.parsed(parse_int(match.group("value")), line).to_dict()

        if "load_hparams:" in line:
            for match in Regexes.LOAD_HPARAMS_EXPERT.finditer(line):
                name = match.group("name")
                value = parse_int(match.group("value"))
                if name == "n_expert":
                    self.report["moe"]["expert_count"] = SourceValue.parsed(value, line).to_dict()
                elif name == "n_expert_used":
                    self.report["moe"]["experts_used"] = SourceValue.parsed(value, line).to_dict()
                elif name == "n_expert_groups":
                    self.report["moe"]["expert_groups"] = SourceValue.parsed(value, line).to_dict()

    def _parse_offload(self, line: str) -> None:
        if Regexes.OUTPUT_OFFLOAD.search(line):
            self.report["offload"]["output_layer_offloaded"] = True
        match = Regexes.REPEATING_OFFLOAD.search(line)
        if match:
            self.report["offload"]["repeating_layers_offloaded"] = parse_int(match.group("count"))
        match = Regexes.OFFLOADED.search(line)
        if match:
            self.report["offload"]["offloaded_layers"] = parse_int(match.group("off"))
            self.report["offload"]["total_layers"] = parse_int(match.group("total"))

    def _parse_buffers(self, line: str) -> None:
        match = Regexes.BUFFER.search(line)
        if not match:
            return
        where = match.group("where")
        kind = match.group("kind").lower()
        mib = parse_mib(match.group("size"), normalized_unit(match.group("unit")))
        if mib is None:
            return
        if kind == "model":
            self.report["memory"]["model_buffers_mib"][where] = mib
        elif kind == "kv":
            self.report["memory"]["kv_buffers_mib"][where] = mib
        elif kind == "rs":
            self.report["memory"]["rs_buffers_mib"][where] = mib
        elif kind == "compute":
            self.report["memory"]["compute_buffers_mib"][where] = mib
        elif kind == "output":
            self.report["memory"]["output_buffers_mib"][where] = mib

    def _parse_kv_size(self, line: str) -> None:
        match = Regexes.KV_SIZE.search(line)
        if not match:
            return
        total = parse_mib(match.group("total"), normalized_unit(match.group("unit")))
        cells = parse_int(match.group("cells"))
        kv_cache = self.report["kv_cache"]
        kv_cache.update(
            {
                "total_mib": total,
                "cells": cells,
                "layers": parse_int(match.group("layers")),
                "seqs": match.group("seqs").strip(),
                "k_type": match.group("ktype").strip(),
                "v_type": match.group("vtype").strip(),
                "k_mib": parse_mib(match.group("k_size"), normalized_unit(match.group("k_unit"))),
                "v_mib": parse_mib(match.group("v_size"), normalized_unit(match.group("v_unit"))),
            }
        )
        if total is not None and cells:
            kv_cache["mib_per_1k_context"] = total / (cells / 1000)
            kv_cache["projected_mib"] = {
                "64000": kv_cache["mib_per_1k_context"] * 64,
                "128000": kv_cache["mib_per_1k_context"] * 128,
                "196000": kv_cache["mib_per_1k_context"] * 196,
                "262000": kv_cache["mib_per_1k_context"] * 262,
            }

    def _parse_rs_size(self, line: str) -> None:
        match = Regexes.RS_SIZE.search(line)
        if not match:
            return
        self.report["recurrent_state"].update(
            {
                "total_mib": parse_mib(match.group("total"), normalized_unit(match.group("unit"))),
                "cells": parse_int(match.group("cells")),
                "layers": parse_int(match.group("layers")),
                "seqs": parse_int(match.group("seqs")),
                "r_type": match.group("rtype").strip(),
                "s_type": match.group("stype").strip(),
                "r_mib": parse_mib(match.group("r_size"), normalized_unit(match.group("r_unit"))),
                "s_mib": parse_mib(match.group("s_size"), normalized_unit(match.group("s_unit"))),
            }
        )

    def _parse_graph(self, line: str) -> None:
        match = Regexes.GRAPH_NODES.search(line)
        if match:
            self.report["graph"]["nodes"] = parse_int(match.group("nodes"))
        match = Regexes.GRAPH_SPLITS.search(line)
        if match:
            for split in Regexes.GRAPH_SPLIT_ITEM.finditer(match.group("splits")):
                bs = str(parse_int(split.group("bs")))
                count = parse_int(split.group("count"))
                self.report["graph"]["splits"][bs] = count
                if bs == "1":
                    self.report["graph"]["splits_bs1"] = count

    def _parse_prompt_cache(self, line: str) -> None:
        cache = self.report["load"]["prompt_cache"]
        match = Regexes.PROMPT_CACHE.search(line)
        if match:
            cache["enabled"] = bool_from_text(match.group("state"))
            if match.group("size"):
                cache["limit_mib"] = parse_mib(match.group("size"), normalized_unit(match.group("unit")))
            return

        match = Regexes.CHECKPOINT.search(line)
        if match:
            cache["checkpoints"].append(
                {
                    "index": parse_int(match.group("index")),
                    "count": parse_int(match.group("count")),
                    "tokens": parse_int(match.group("tokens")),
                    "size_mib": parse_mib(match.group("size"), normalized_unit(match.group("unit"))),
                }
            )
            return

        match = Regexes.LCP.search(line)
        if match:
            cache["lcp_similarity_events"].append({"similarity": parse_number(match.group("sim"))})
            return

        match = Regexes.CACHE_REUSE_IGNORED.search(line)
        if match:
            cache["cache_reuse_ignored"].append({"requested_tokens": parse_int(match.group("reuse")), "evidence": line.strip()})

    def _parse_context_event(self, line: str) -> None:
        slot_task = Regexes.SLOT_TASK.search(line)
        if not slot_task:
            return
        slot = parse_int(slot_task.group("slot"))
        task = parse_int(slot_task.group("task"))
        if slot is None or task is None:
            return
        state = self.context_by_slot_task.setdefault((slot, task), {})
        slot_state = self.context_by_slot.setdefault(slot, {})
        task_tokens = Regexes.TASK_TOKENS.search(line)
        prompt_done = Regexes.PROMPT_DONE.search(line)
        stop_processing = Regexes.STOP_PROCESSING.search(line)
        if task_tokens:
            value = parse_int(task_tokens.group("tokens"))
            if value is not None:
                state["task_tokens"] = value
                slot_state["task_tokens"] = value
        if prompt_done:
            value = parse_int(prompt_done.group("tokens"))
            if value is not None:
                state["prompt_done_n_tokens"] = value
                slot_state["prompt_done_n_tokens"] = value
        if stop_processing:
            value = parse_int(stop_processing.group("tokens"))
            if value is not None:
                state["stop_processing_n_tokens"] = value
                slot_state["stop_processing_n_tokens"] = value

    def _derive_active_context(self, slot: Optional[int], task: Optional[int], timing: Dict[str, Any]) -> Dict[str, Any]:
        context: Dict[str, int] = {}
        if slot is not None and task is not None:
            context.update(self.context_by_slot_task.get((slot, task), {}))
        if slot is not None:
            for key, value in self.context_by_slot.get(slot, {}).items():
                context.setdefault(key, value)
        eval_tokens = timing.get("eval_tokens") or 0
        prompt_tokens = timing.get("prompt_tokens")
        candidates: Dict[str, int] = {}
        if "stop_processing_n_tokens" in context:
            candidates["stop_processing_n_tokens"] = context["stop_processing_n_tokens"]
        if "prompt_done_n_tokens" in context:
            candidates["prompt_done_plus_eval"] = context["prompt_done_n_tokens"] + eval_tokens
        if "task_tokens" in context:
            candidates["task_tokens_plus_eval"] = context["task_tokens"] + eval_tokens
        if prompt_tokens is not None:
            candidates["prompt_tokens_plus_eval"] = prompt_tokens + eval_tokens
            candidates["prompt_tokens"] = prompt_tokens
        priority = [
            "stop_processing_n_tokens",
            "prompt_done_plus_eval",
            "task_tokens_plus_eval",
            "prompt_tokens_plus_eval",
            "prompt_tokens",
        ]
        selected_method = None
        selected_value = None
        for method in priority:
            if method in candidates:
                selected_method = method
                selected_value = candidates[method]
                break
        conflict_warning = None
        if selected_value is not None:
            tolerance_tokens = self.thresholds.get("active_context_conflict_tolerance_tokens", 500)
            tolerance = max(float(tolerance_tokens), selected_value * 0.05)
            conflicts = {
                key: value for key, value in candidates.items()
                if abs(value - selected_value) > tolerance
            }
            if conflicts:
                conflict_warning = (
                    f"Active-context estimates for slot {slot} task {task} differ by more than "
                    f"{int(tolerance)} tokens; selected {selected_method}={selected_value}."
                )
        return {
            "value": selected_value,
            "source": "derived" if selected_value is not None else "missing",
            "method": selected_method,
            "candidates": candidates,
            "conflict_warning": conflict_warning,
        }

    def _finalize(self) -> None:
        self._refresh_timing_contexts()
        self._finalize_quant()
        self._finalize_offload()
        self._finalize_memory()
        self._finalize_kv_split()
        self._finalize_graph()
        self._finalize_moe_inference()
        self.report["analysis"]["placement_grade"] = placement_grade(self.report, self.thresholds)
        self.report["timing_summary"] = timing_summary(self.report["timings"], self.thresholds)
        self.report["analysis"]["long_context_behavior"] = long_context_behavior(self.report["timings"])
        self.report["analysis"]["fit_and_headroom"] = fit_and_headroom(self.report)
        self.report["recommendations"] = recommendations(self.report, self.thresholds)
        self._finalize_warnings()

    def _refresh_timing_contexts(self) -> None:
        active_context_warnings = []
        self.report["warnings"] = [
            warning for warning in self.report["warnings"]
            if not warning.startswith("Active-context estimates ")
        ]
        for timing in self.report["timings"]:
            timing["warnings"] = [
                warning for warning in timing.get("warnings", [])
                if not warning.startswith("Active-context estimates ")
            ]
            timing["active_context"] = self._derive_active_context(timing.get("slot_id"), timing.get("task_id"), timing)
            timing["context_band"] = context_band(timing["active_context"].get("value"), self.thresholds)
            timing["generation_speed_grade"] = speed_grade(timing["eval_tokens_per_second"], timing["context_band"], self.thresholds)
            timing["prompt_eval_grade"] = prompt_grade(timing["prompt_tokens_per_second"], timing["prompt_tokens"], self.thresholds)
            warning = timing["active_context"].get("conflict_warning")
            if warning:
                timing["warnings"].append(warning)
                active_context_warnings.append(warning)
        self.report["warnings"].extend(active_context_warnings)

    def _finalize_quant(self) -> None:
        model = self.report["model"]
        filename_quant = infer_quant_from_filename(model.get("filename"))
        if filename_quant:
            model["quant"] = SourceValue.inferred(filename_quant, f"filename: {model.get('filename')}", 0.9).to_dict()
            return
        file_type_quant = quant_from_file_type(model.get("file_type"))
        if file_type_quant:
            model["quant"] = SourceValue.parsed(file_type_quant, f"file type: {model.get('file_type')}").to_dict()

    def _finalize_offload(self) -> None:
        offload = self.report["offload"]
        off = offload.get("offloaded_layers")
        total = offload.get("total_layers")
        if off is not None and total:
            ratio = off / total
            offload["offload_ratio"] = SourceValue.derived(ratio, f"offloaded_layers / total_layers = {off}/{total}").to_dict()
            offload["whole_layer_cpu_fallback_count"] = total - off
        n_layer = self.report["model"].get("n_layer")
        repeating = offload.get("repeating_layers_offloaded")
        if n_layer is not None and repeating is not None:
            offload["repeating_layers_cpu_fallback_count"] = max(n_layer - repeating, 0)

    def _finalize_memory(self) -> None:
        memory = self.report["memory"]
        model_buffers = memory["model_buffers_mib"]
        visible_model_total = sum(model_buffers.values()) if model_buffers else None
        if visible_model_total is not None:
            memory["visible_model_total_mib"] = SourceValue.derived(visible_model_total, "sum of visible model buffers").to_dict()
            cuda_model = sum(value for key, value in model_buffers.items() if key.startswith("CUDA") and key != "CUDA_Host")
            cuda_host_model = model_buffers.get("CUDA_Host", 0.0)
            cpu_mapped = model_buffers.get("CPU_Mapped", 0.0)
            if visible_model_total > 0:
                memory["visible_cuda_model_percent"] = cuda_model / visible_model_total * 100
                memory["visible_cuda_host_model_percent"] = cuda_host_model / visible_model_total * 100
                memory["visible_cpu_mapped_model_percent"] = cpu_mapped / visible_model_total * 100

        cuda_total = 0.0
        cuda_host_total = 0.0
        cpu_total = 0.0
        for bucket_name in ["model_buffers_mib", "kv_buffers_mib", "rs_buffers_mib", "compute_buffers_mib", "output_buffers_mib"]:
            for key, value in memory[bucket_name].items():
                if key == "CUDA_Host":
                    cuda_host_total += value
                elif key.startswith("CUDA"):
                    cuda_total += value
                elif key.startswith("CPU"):
                    cpu_total += value
        if any([cuda_total, cuda_host_total, cpu_total]):
            memory["visible_allocations_mib"]["cuda"] = SourceValue.derived(cuda_total, "sum of visible CUDA device allocations").to_dict()
            memory["visible_allocations_mib"]["cuda_host"] = SourceValue.derived(cuda_host_total, "sum of visible CUDA host allocations").to_dict()
            memory["visible_allocations_mib"]["cpu"] = SourceValue.derived(cpu_total, "sum of visible CPU allocations").to_dict()
        total_vram = self.report["hardware"].get("total_vram_mib")
        if total_vram and cuda_total:
            memory["visible_cuda_vram_ratio"] = SourceValue.derived(cuda_total / total_vram, "visible CUDA allocations / reported total VRAM").to_dict()

    def _finalize_kv_split(self) -> None:
        kv_buffers = self.report["memory"]["kv_buffers_mib"]
        total = self.report["kv_cache"].get("total_mib") or sum(kv_buffers.values())
        if total:
            split = {}
            for key, value in kv_buffers.items():
                split[key] = {
                    "mib": value,
                    "percent": value / total * 100,
                }
            self.report["kv_cache"]["cpu_gpu_split"] = split

    def _finalize_graph(self) -> None:
        graph = self.report["graph"]
        bs1 = graph.get("splits_bs1")
        offloaded = self.report["offload"].get("offloaded_layers")
        total = self.report["offload"].get("total_layers")
        if bs1 is not None and offloaded:
            graph["splits_bs1_per_offloaded_layer"] = bs1 / offloaded
        if bs1 is not None and total:
            graph["splits_bs1_per_total_layer"] = bs1 / total

    def _finalize_moe_inference(self) -> None:
        moe = self.report["moe"]
        is_moe = any(
            [
                moe["expert_count"].get("value"),
                moe["experts_used"].get("value"),
                "moe" in (self.report["model"].get("architecture") or "").lower(),
                "a3b" in (self.report["model"].get("model_type") or "").lower(),
            ]
        )
        cuda_host_model = self.report["memory"]["model_buffers_mib"].get("CUDA_Host")
        if is_moe and cuda_host_model and cuda_host_model >= 1024:
            moe["cpu_moe_placement"] = SourceValue.inferred(
                "likely",
                "CUDA_Host model buffer is present and large while model metadata indicates MoE",
                0.6,
            ).to_dict()
        elif is_moe:
            moe["cpu_moe_placement"] = SourceValue.inferred(
                "unknown",
                "MoE metadata is present, but the log does not show enough allocation evidence for CPU-MoE placement",
                0.3,
            ).to_dict()

    def _finalize_warnings(self) -> None:
        if not self.report["timings"]:
            self.report["warnings"].append("No timing blocks were found; speed analysis is incomplete.")
        if self.report["load"]["prompt_cache"]["cache_reuse_ignored"]:
            self.report["warnings"].append("The log reports that prompt cache reuse is not supported for at least one request.")
        if self.report["model"]["path"] is None and self.report["model"]["filename"] is None:
            self.report["warnings"].append("Model path was not found; filename-derived fields may be missing.")


def normalized_unit(unit: str) -> str:
    exact = {
        "b": "B",
        "kib": "KiB",
        "mib": "MiB",
        "gib": "GiB",
        "tib": "TiB",
        "kb": "KB",
        "mb": "MB",
        "gb": "GB",
        "tb": "TB",
    }
    return exact.get(unit.lower(), unit)


def context_band(active_context: Optional[int], thresholds: Dict[str, Any]) -> Optional[str]:
    if active_context is None:
        return None
    for name, bounds in thresholds.get("context_bands", {}).items():
        lower, upper = bounds
        if active_context >= lower and (upper is None or active_context < upper):
            return name
    return None


def grade_from_thresholds(value: Optional[float], grades: Dict[str, float]) -> Optional[str]:
    if value is None:
        return None
    for grade in ["A", "B", "C", "D"]:
        threshold = grades.get(grade)
        if threshold is not None and value >= threshold:
            return grade
    return "F"


def speed_grade(value: Optional[float], band: Optional[str], thresholds: Dict[str, Any]) -> Optional[str]:
    if value is None or band is None:
        return None
    grades = thresholds.get("generation_speed_grades", {}).get(band)
    if not grades:
        return None
    return grade_from_thresholds(value, grades)


def prompt_grade(value: Optional[float], prompt_tokens: Optional[int], thresholds: Dict[str, Any]) -> Optional[str]:
    if value is None:
        return None
    if prompt_tokens is not None and prompt_tokens < thresholds.get("small_prompt_sample_tokens", 256):
        return "small-sample / overhead-dominated"
    return grade_from_thresholds(value, thresholds.get("prompt_eval_speed_grades", {}))


def placement_grade(report: Dict[str, Any], thresholds: Dict[str, Any]) -> Optional[str]:
    ratio = report["offload"]["offload_ratio"].get("value")
    cuda_host = report["memory"]["model_buffers_mib"].get("CUDA_Host")
    bs1 = report["graph"].get("splits_bs1")
    placement = thresholds.get("placement_grade_thresholds", {})

    def satisfies(grade: str) -> bool:
        rule = placement.get(grade, {})
        min_ratio = rule.get("offload_ratio")
        max_host = rule.get("cuda_host_model_mib")
        max_splits = rule.get("graph_splits_bs1")
        if min_ratio is not None and (ratio is None or ratio < min_ratio):
            return False
        if max_host is not None and cuda_host is not None and cuda_host > max_host:
            return False
        if max_splits is not None and bs1 is not None and bs1 > max_splits:
            return False
        return True

    for grade in ["A", "B", "C"]:
        if satisfies(grade):
            return grade
    d_rule = placement.get("D", {})
    if ratio is not None and ratio >= d_rule.get("offload_ratio", 0.30):
        return "D"
    if cuda_host is not None and cuda_host <= d_rule.get("cuda_host_model_mib", 20480):
        return "D"
    if ratio is None and cuda_host is None and bs1 is None:
        return None
    return "F"


def timing_summary(timings: List[Dict[str, Any]], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    by_band: Dict[str, Dict[str, Any]] = {}
    for timing in timings:
        band = timing.get("context_band") or "unknown"
        by_band.setdefault(
            band,
            {
                "count": 0,
                "generation_sample_count": 0,
                "best_generation_tps": None,
                "average_generation_tps": None,
                "median_generation_tps": None,
                "worst_generation_tps": None,
                "best_prompt_tps": None,
                "median_prompt_tps": None,
                "_generation_values": [],
                "_prompt_values": [],
            },
        )
        entry = by_band[band]
        entry["count"] += 1
        if timing.get("eval_tokens_per_second") is not None:
            entry["_generation_values"].append(timing["eval_tokens_per_second"])
        prompt_tokens = timing.get("prompt_tokens")
        if (
            timing.get("prompt_tokens_per_second") is not None
            and prompt_tokens is not None
            and prompt_tokens >= thresholds.get("small_prompt_sample_tokens", 256)
        ):
            entry["_prompt_values"].append(timing["prompt_tokens_per_second"])
    for entry in by_band.values():
        gen_values = entry.pop("_generation_values")
        prompt_values = entry.pop("_prompt_values")
        entry["generation_sample_count"] = len(gen_values)
        if gen_values:
            entry["best_generation_tps"] = max(gen_values)
            entry["average_generation_tps"] = sum(gen_values) / len(gen_values)
            entry["median_generation_tps"] = median(gen_values)
            entry["worst_generation_tps"] = min(gen_values)
        if prompt_values:
            entry["best_prompt_tps"] = max(prompt_values)
            entry["median_prompt_tps"] = median(prompt_values)
    overall_values = [timing["eval_tokens_per_second"] for timing in timings if timing.get("eval_tokens_per_second") is not None]
    preferred = median(overall_values) if overall_values else None
    preferred_method = "median_generation_tps" if overall_values else None
    average_generation_tps = sum(overall_values) / len(overall_values) if overall_values else None
    return {
        "by_context_band": by_band,
        "overall_generation_tps": preferred,
        "average_generation_tps": average_generation_tps,
        "overall_generation_method": preferred_method,
        "sample_count": len(overall_values),
        "timing_sample_count": len(timings),
    }


def long_context_behavior(timings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    samples = [
        (timing.get("active_context", {}).get("value"), timing.get("eval_tokens_per_second"))
        for timing in timings
        if timing.get("active_context", {}).get("value") is not None and timing.get("eval_tokens_per_second") is not None
    ]
    if len(samples) < 2:
        return None
    samples.sort(key=lambda item: item[0])
    smallest_ctx, smallest_speed = samples[0]
    largest_ctx, largest_speed = samples[-1]
    if largest_ctx == smallest_ctx or smallest_speed == 0:
        return None
    drop = smallest_speed - largest_speed
    drop_percent = drop / smallest_speed * 100
    per_10k = drop / ((largest_ctx - smallest_ctx) / 10000)
    severity = "mild"
    if drop_percent >= 40:
        severity = "steep"
    elif drop_percent >= 20:
        severity = "moderate"
    return {
        "smallest_context": smallest_ctx,
        "smallest_context_generation_tps": smallest_speed,
        "largest_context": largest_ctx,
        "largest_context_generation_tps": largest_speed,
        "drop_tps": drop,
        "drop_percent": drop_percent,
        "drop_tps_per_10k_context": per_10k,
        "severity": severity,
    }


def fit_and_headroom(report: Dict[str, Any]) -> Dict[str, Any]:
    visible_cuda = report["memory"]["visible_allocations_mib"]["cuda"].get("value")
    total_vram = report["hardware"].get("total_vram_mib")
    result = {
        "label": "visible parsed allocations",
        "visible_cuda_mib": visible_cuda,
        "reported_total_vram_mib": total_vram,
        "visible_cuda_vram_ratio": report["memory"]["visible_cuda_vram_ratio"].get("value"),
        "free_vram_at_load_mib": report["hardware"].get("free_vram_at_load_mib"),
    }
    if visible_cuda is not None and total_vram:
        result["visible_cuda_headroom_mib"] = total_vram - visible_cuda
    else:
        result["visible_cuda_headroom_mib"] = None
    return result


def recommendations(report: Dict[str, Any], thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    parallel = report["load"].get("parallel")
    slots = report["load"].get("slots")
    if parallel is not None and parallel > 1:
        recs.append(
            {
                "title": "Set Parallel to 1",
                "severity": "suggestion",
                "reason": f"n_parallel is currently {parallel}, which increases memory and state overhead for a single interactive workload.",
                "evidence": [f"n_parallel={parallel}"],
            }
        )
    elif parallel == 1 and (slots in {None, 1}):
        evidence = ["n_parallel=1"]
        if slots is not None:
            evidence.append(f"n_slots={slots}")
        recs.append(
            {
                "title": "Keep Parallel = 1",
                "severity": "info",
                "reason": "Single-slot workload and current parallelism avoid extra memory and state overhead.",
                "evidence": evidence,
            }
        )

    actual_ctx = report["load"].get("context_actual")
    train_ctx = report["model"].get("context_trained")
    kv_total = report["kv_cache"].get("total_mib")
    if actual_ctx and train_ctx and actual_ctx < train_ctx * 0.75 and kv_total and kv_total < 4096:
        recs.append(
            {
                "title": "Test a larger context before maxing it out",
                "severity": "suggestion",
                "reason": "The constructed context is below the trained context and parsed KV memory is still modest.",
                "evidence": [f"n_ctx={actual_ctx}", f"n_ctx_train={train_ctx}", f"KV total={kv_total:.2f} MiB"],
            }
        )

    offloaded = report["offload"].get("offloaded_layers")
    total = report["offload"].get("total_layers")
    ratio = report["offload"]["offload_ratio"].get("value")
    cuda_host = report["memory"]["model_buffers_mib"].get("CUDA_Host")
    if cuda_host is not None and (cuda_host >= thresholds.get("cuda_host_model_warning_mib", 8192) or (ratio is not None and ratio < 0.5)):
        evidence = [f"CUDA_Host model buffer={cuda_host:.2f} MiB"]
        if offloaded is not None and total is not None:
            evidence.append(f"offloaded={offloaded}/{total}")
        recs.append(
            {
                "title": "Prefer measured speed when tuning CPU-MoE or quant",
                "severity": "suggestion",
                "reason": "A large CUDA host model buffer or low offload ratio can explain slower generation, but measured generation speed remains primary.",
                "evidence": evidence,
            }
        )

    bs1 = report["graph"].get("splits_bs1")
    if bs1 is not None and bs1 >= thresholds.get("graph_split_warning_threshold_bs1", 35):
        recs.append(
            {
                "title": "Reduce cross-device graph splits if possible",
                "severity": "suggestion",
                "reason": "High bs=1 graph split counts usually hurt token generation in similar architectures.",
                "evidence": [f"graph splits bs=1={bs1}"],
            }
        )

    for timing in report["timings"]:
        prompt_tps = timing.get("prompt_tokens_per_second")
        gen_tps = timing.get("eval_tokens_per_second")
        prompt_tokens = timing.get("prompt_tokens")
        if prompt_tps is not None and gen_tps is not None and prompt_tps >= 1000 and gen_tps < 8:
            recs.append(
                {
                    "title": "Treat generation as the bottleneck",
                    "severity": "info",
                    "reason": "Prompt eval is much faster than token generation in a measured timing sample.",
                    "evidence": [f"prompt eval={prompt_tps:.2f} tok/s", f"generation={gen_tps:.2f} tok/s"],
                }
            )
            break
        if prompt_tps is not None and prompt_tokens is not None and prompt_tokens >= 2048 and prompt_tps < 500:
            recs.append(
                {
                    "title": "Consider increasing eval batch size",
                    "severity": "suggestion",
                    "reason": "Prompt processing is weak on a large enough prompt sample to be meaningful.",
                    "evidence": [f"prompt tokens={prompt_tokens}", f"prompt eval={prompt_tps:.2f} tok/s"],
                }
            )
            break
    return dedupe_recommendations(recs)


def dedupe_recommendations(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for rec in recs:
        title = rec["title"]
        if title in seen:
            continue
        seen.add(title)
        output.append(rec)
    return output


def parse_log(text: str, source_name: Optional[str] = None, thresholds: Optional[Dict[str, Any]] = None, thresholds_path: Optional[str] = None) -> Dict[str, Any]:
    parser = LogParser(text, source_name, thresholds or json.loads(json.dumps(DEFAULT_THRESHOLDS)), thresholds_path)
    return parser.parse()


def json_ready(value: Any, path: Tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item, path + (str(key),)) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item, path) for item in value]
    if isinstance(value, float):
        key = path[-1] if path else ""
        ratio_keys = {"offload_ratio", "visible_cuda_vram_ratio", "splits_bs1_per_total_layer", "splits_bs1_per_offloaded_layer"}
        if "ratio" in key or any(part in ratio_keys or "ratio" in part for part in path):
            return round(value, 4)
        return round(value, 2)
    return value


def render_json(report: Dict[str, Any]) -> str:
    return json.dumps(json_ready(report), sort_keys=True, indent=2)


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def source_value_text(source_value: Dict[str, Any]) -> str:
    value = source_value.get("value")
    if value is None:
        return "unknown"
    return str(value)


def generation_speed_values(report: Dict[str, Any]) -> List[float]:
    return [
        timing["eval_tokens_per_second"]
        for timing in report["timings"]
        if timing.get("eval_tokens_per_second") is not None
    ]


def render_human(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    model = report["model"]
    load = report["load"]
    timing_summary_data = report["timing_summary"]

    lines.append(
        f"Model: {model.get('filename') or 'unknown'} | "
        f"Quant: {source_value_text(model.get('quant', missing_source()))} | "
        f"Context: {fmt(load.get('context_actual'))}"
    )
    overall_tps = timing_summary_data.get("overall_generation_tps")
    sample_count = timing_summary_data.get("sample_count")
    sample_label = "sample" if sample_count == 1 else "samples"
    lines.append(f"Median generation: {fmt(overall_tps, ' tok/s')} ({fmt(sample_count)} {sample_label})")
    lines.append(f"Average generation: {fmt(timing_summary_data.get('average_generation_tps'), ' tok/s')}")
    generation_values = generation_speed_values(report)
    lines.append(f"High generation: {fmt(max(generation_values) if generation_values else None, ' tok/s')}")
    lines.append(f"Low generation: {fmt(min(generation_values) if generation_values else None, ' tok/s')}")
    prompt_values = [timing.get("prompt_tokens_per_second") for timing in report["timings"] if timing.get("prompt_tokens_per_second") is not None]
    lines.append(f"Prompt eval: {fmt(max(prompt_values) if prompt_values else None, ' tok/s')}")
    return "\n".join(lines)


def render_time_human(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    model = report["model"]
    load = report["load"]
    offload = report["offload"]
    memory = report["memory"]
    kv = report["kv_cache"]
    graph = report["graph"]
    timing_summary_data = report["timing_summary"]

    lines.append("LLLogAnalyzer Tuning Summary")
    lines.append("========================")
    lines.append("")
    lines.append(
        f"Model: {model.get('filename') or 'unknown'} | "
        f"Quant: {source_value_text(model.get('quant', missing_source()))} | "
        f"Context: {fmt(load.get('context_actual'))}"
    )
    lines.append("")

    lines.append("Speed")
    lines.append("-----")
    overall_tps = timing_summary_data.get("overall_generation_tps")
    sample_count = timing_summary_data.get("sample_count")
    sample_label = "sample" if sample_count == 1 else "samples"
    lines.append(f"Median generation: {fmt(overall_tps, ' tok/s')} ({fmt(sample_count)} {sample_label})")
    lines.append(f"Average generation: {fmt(timing_summary_data.get('average_generation_tps'), ' tok/s')}")
    generation_values = generation_speed_values(report)
    lines.append(f"High generation: {fmt(max(generation_values) if generation_values else None, ' tok/s')}")
    lines.append(f"Low generation: {fmt(min(generation_values) if generation_values else None, ' tok/s')}")
    prompt_values = [timing.get("prompt_tokens_per_second") for timing in report["timings"] if timing.get("prompt_tokens_per_second") is not None]
    lines.append(f"Prompt eval: {fmt(max(prompt_values) if prompt_values else None, ' tok/s')}")
    if report["timings"]:
        lines.append("Samples:")
        for timing in report["timings"]:
            active = timing.get("active_context", {}).get("value")
            lines.append(
                f"- {fmt(timing.get('eval_tokens_per_second'), ' tok/s')} gen, "
                f"{fmt(timing.get('prompt_tokens_per_second'), ' tok/s')} prompt, "
                f"ctx {fmt(active)}, {timing.get('context_band') or 'unknown'} band"
            )
    else:
        lines.append("Samples: none")
    lines.append("")

    lines.append("Diagnostics")
    lines.append("-----------")
    ratio = offload.get("offload_ratio", {}).get("value")
    if offload.get("offloaded_layers") is not None and offload.get("total_layers") is not None:
        offload_text = f"{offload['offloaded_layers']}/{offload['total_layers']} layers ({fmt(ratio * 100 if ratio is not None else None, '%')})"
    else:
        offload_text = "unknown"
    lines.append(
        f"Offload: {offload_text}; CPU fallback {fmt(offload.get('whole_layer_cpu_fallback_count'))}; "
        f"placement {fmt(report['analysis'].get('placement_grade'))}"
    )
    visible_cuda = memory["visible_allocations_mib"]["cuda"].get("value")
    visible_host = memory["visible_allocations_mib"]["cuda_host"].get("value")
    lines.append(
        f"Memory: CUDA {fmt(visible_cuda, ' MiB')}; CUDA host {fmt(visible_host, ' MiB')}; "
        f"KV {fmt(kv.get('total_mib'), ' MiB')}"
    )
    if graph.get("splits"):
        split_text = ", ".join(f"bs={bs}: {count}" for bs, count in sorted(graph["splits"].items(), key=lambda item: int(item[0])))
    else:
        split_text = "unknown"
    lines.append(
        f"Graph: nodes {fmt(graph.get('nodes'))}; splits {split_text}; "
        f"bs=1/layer {fmt(graph.get('splits_bs1_per_total_layer'))}"
    )
    lines.append(
        f"Load: batch {fmt(load.get('batch'))}/{fmt(load.get('microbatch'))}; "
        f"flash {fmt(load.get('flash_attention'))}; KV unified {fmt(load.get('kv_unified'))}; "
        f"prompt cache {fmt(load['prompt_cache'].get('enabled'))}"
    )

    actionable_recommendations = [rec for rec in report["recommendations"] if rec.get("severity") != "info"]
    if actionable_recommendations:
        lines.append("")
        lines.append("Recommendations")
        lines.append("---------------")
        for rec in actionable_recommendations[:3]:
            lines.append(f"- {rec['title']}")
    if report["warnings"]:
        lines.append("")
        lines.append("Warnings")
        lines.append("--------")
        for warning in report["warnings"][:3]:
            lines.append(f"- {warning}")
        if len(report["warnings"]) > 3:
            lines.append(f"- {len(report['warnings']) - 3} more warnings; rerun with --verbose for details.")
    return "\n".join(lines)


def context_range_text(bounds: Any) -> str:
    if not isinstance(bounds, list) or len(bounds) != 2:
        return "unknown"
    lower, upper = bounds
    if not isinstance(lower, (int, float)) or (upper is not None and not isinstance(upper, (int, float))):
        return "unknown"
    lower_text = f"{int(lower):,}" if float(lower).is_integer() else fmt(lower)
    if upper is None:
        return f">= {lower_text}"
    if isinstance(upper, (int, float)) and float(upper).is_integer() and upper > lower:
        return f"{lower_text}-{int(upper) - 1:,}"
    upper_text = f"{int(upper):,}" if float(upper).is_integer() else fmt(upper)
    return f"{lower_text}-<{upper_text}"


def context_band_sort_key(item: Tuple[str, Any]) -> Tuple[float, str]:
    name, bounds = item
    if isinstance(bounds, list) and bounds and isinstance(bounds[0], (int, float)):
        return float(bounds[0]), name
    return float("inf"), name


def render_context_speed_table(report: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
    lines = ["Average Generation by Context Range", "-----------------------------------"]
    by_band = report["timing_summary"].get("by_context_band", {})
    context_bands = thresholds.get("context_bands", DEFAULT_THRESHOLDS["context_bands"])
    rows: List[Tuple[str, str, str, str]] = []
    seen_bands = set()

    for band, bounds in sorted(context_bands.items(), key=context_band_sort_key):
        summary = by_band.get(band, {})
        rows.append((
            band,
            context_range_text(bounds),
            fmt(summary.get("generation_sample_count", 0)),
            fmt(summary.get("average_generation_tps"), " tok/s"),
        ))
        seen_bands.add(band)

    for band in sorted(name for name in by_band if name not in seen_bands):
        summary = by_band[band]
        rows.append((
            band,
            "unknown",
            fmt(summary.get("generation_sample_count", 0)),
            fmt(summary.get("average_generation_tps"), " tok/s"),
        ))

    if not rows:
        rows = [("unknown", "unknown", "0", "unknown")]

    headers = ("Band", "Context range", "Samples", "Avg gen")
    widths = [
        max(len(headers[0]), *(len(row[0]) for row in rows)),
        max(len(headers[1]), *(len(row[1]) for row in rows)),
        max(len(headers[2]), *(len(row[2]) for row in rows)),
        max(len(headers[3]), *(len(row[3]) for row in rows)),
    ]
    lines.append(f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  {headers[2]:>{widths[2]}}  {headers[3]:>{widths[3]}}")
    lines.append(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * widths[3]}")
    for band, context_range, samples, average in rows:
        lines.append(f"{band:<{widths[0]}}  {context_range:<{widths[1]}}  {samples:>{widths[2]}}  {average:>{widths[3]}}")
    return "\n".join(lines)


def context_speed_samples(report: Dict[str, Any]) -> List[Tuple[int, float]]:
    samples = []
    for timing in report["timings"]:
        active = timing.get("active_context", {}).get("value")
        speed = timing.get("eval_tokens_per_second")
        if active is not None and speed is not None:
            samples.append((active, speed))
    return sorted(samples, key=lambda item: item[0])


def render_speed_graph(report: Dict[str, Any]) -> str:
    samples = context_speed_samples(report)
    lines = ["Generation Speed vs Context", "---------------------------"]
    if not samples:
        lines.append("No generation timing samples with context size found.")
        return "\n".join(lines)

    width = 60
    height = 12
    contexts = [sample[0] for sample in samples]
    speeds = [sample[1] for sample in samples]
    x_min = min(contexts)
    x_max = max(contexts)
    y_min = 0.0
    y_max = max(speeds)
    if y_max <= y_min:
        y_max = 1.0

    grid = [[" "] * width for _ in range(height)]
    for context, speed in samples:
        if x_max == x_min:
            col = width // 2
        else:
            col = int(round((context - x_min) / (x_max - x_min) * (width - 1)))
        row = height - 1 - int(round((speed - y_min) / (y_max - y_min) * (height - 1)))
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        grid[row][col] = "#" if grid[row][col] != " " else "*"

    for index, row_cells in enumerate(grid):
        y_value = y_max - ((y_max - y_min) * index / (height - 1))
        lines.append(f"{y_value:>8.2f} |{''.join(row_cells)}")
    lines.append(f"{'':>8} +{'-' * width}")
    if x_max == x_min:
        lines.append(f"{'Context':>8}: {x_min:,} tokens")
    else:
        x_mid = int((x_min + x_max) / 2)
        lines.append(f"{'Context':>8}: {x_min:,} -> {x_mid:,} -> {x_max:,} tokens")
    lines.append(
        f"Samples: {len(samples)} | Speed range: {fmt(min(speeds), ' tok/s')} to {fmt(max(speeds), ' tok/s')}"
    )
    if any(cell == "#" for row_cells in grid for cell in row_cells):
        lines.append("Legend: # = multiple samples in the same plot cell")
    return "\n".join(lines)


def render_verbose_human(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("LLLogAnalyzer Report")
    lines.append("==================================")
    lines.append("")
    disclaimer = "Grades are deterministic heuristics based on this tool's thresholds, not absolute model-quality or hardware-quality measurements."
    if report["source"].get("thresholds_path"):
        disclaimer += f" Thresholds: {report['source']['thresholds_path']}."
    lines.append(disclaimer)
    lines.append("")
    model = report["model"]
    load = report["load"]
    offload = report["offload"]
    memory = report["memory"]
    kv = report["kv_cache"]
    graph = report["graph"]

    lines.append("Source")
    lines.append("------")
    lines.append(f"Input: {report['source'].get('name') or 'stdin/clipboard'}")
    lines.append("")

    lines.append("Model")
    lines.append("-----")
    lines.append(f"Model file: {model.get('filename') or 'unknown'}")
    lines.append(f"Quant: {source_value_text(model.get('quant', missing_source()))}")
    lines.append(f"Architecture: {model.get('architecture') or 'unknown'}")
    lines.append(f"Model type: {model.get('model_type') or 'unknown'}")
    lines.append(f"File type: {model.get('file_type') or 'unknown'}")
    lines.append(f"File size: {fmt(model.get('file_size_mib'), ' MiB')}")
    if model.get("declared_bpw") is not None or model.get("effective_bpw") is not None:
        lines.append(f"BPW: {fmt(model.get('declared_bpw'))} declared, {fmt(model.get('effective_bpw'))} effective")
    lines.append("")

    lines.append("Load")
    lines.append("----")
    lines.append(f"Context requested: {fmt(load.get('context_requested'))}")
    lines.append(f"Context actual: {fmt(load.get('context_actual'))}")
    lines.append(f"Context trained: {fmt(model.get('context_trained'))}")
    lines.append(f"Parallel: {fmt(load.get('parallel'))}")
    lines.append(f"Batch / microbatch: {fmt(load.get('batch'))} / {fmt(load.get('microbatch'))}")
    lines.append(f"Flash attention: {fmt(load.get('flash_attention'))}")
    lines.append(f"KV unified: {fmt(load.get('kv_unified'))}")
    lines.append(f"Prompt cache: {fmt(load['prompt_cache'].get('enabled'))}, limit {fmt(load['prompt_cache'].get('limit_mib'), ' MiB')}")
    lines.append("")

    lines.append("Hardware")
    lines.append("--------")
    lines.append(f"CUDA devices: {fmt(report['hardware'].get('cuda_device_count'))}")
    lines.append(f"Total VRAM: {fmt(report['hardware'].get('total_vram_mib'), ' MiB')}")
    for device in report["hardware"].get("devices", []):
        lines.append(
            f"Device {device.get('index')}: {device.get('name')} "
            f"(cc {device.get('compute_capability')}, VMM {device.get('vmm')}, {fmt(device.get('vram_mib'), ' MiB')})"
        )
    lines.append("")

    lines.append("Placement")
    lines.append("---------")
    ratio = offload.get("offload_ratio", {}).get("value")
    if offload.get("offloaded_layers") is not None and offload.get("total_layers") is not None:
        lines.append(f"Offloaded: {offload['offloaded_layers']}/{offload['total_layers']} layers ({fmt(ratio * 100 if ratio is not None else None, '%')})")
    else:
        lines.append("Offloaded: unknown")
    lines.append(f"Repeating layers on GPU: {fmt(offload.get('repeating_layers_offloaded'))}")
    lines.append(f"Whole-layer CPU fallback: {fmt(offload.get('whole_layer_cpu_fallback_count'))}")
    lines.append(f"CPU-MoE placement: {source_value_text(report['moe'].get('cpu_moe_placement', missing_source()))}")
    lines.append(f"Placement grade: {fmt(report['analysis'].get('placement_grade'))} (heuristic)")
    lines.append("")

    lines.append("Memory")
    lines.append("------")
    lines.append("Visible parsed model buffers:")
    for key in sorted(memory["model_buffers_mib"]):
        lines.append(f"  {key}: {fmt(memory['model_buffers_mib'][key], ' MiB')}")
    visible = memory.get("visible_model_total_mib", {}).get("value")
    lines.append(f"Visible model total: {fmt(visible, ' MiB')}")
    visible_cuda = memory["visible_allocations_mib"]["cuda"].get("value")
    visible_host = memory["visible_allocations_mib"]["cuda_host"].get("value")
    visible_cpu = memory["visible_allocations_mib"]["cpu"].get("value")
    lines.append(f"Visible CUDA allocations: {fmt(visible_cuda, ' MiB')}")
    lines.append(f"Visible CUDA host allocations: {fmt(visible_host, ' MiB')}")
    lines.append(f"Visible CPU allocations: {fmt(visible_cpu, ' MiB')}")
    lines.append("")

    lines.append("KV Cache")
    lines.append("--------")
    lines.append(f"Total: {fmt(kv.get('total_mib'), ' MiB')}")
    for key in sorted(memory["kv_buffers_mib"]):
        lines.append(f"{key}: {fmt(memory['kv_buffers_mib'][key], ' MiB')}")
    lines.append(f"K/V: {fmt(kv.get('k_type'))} / {fmt(kv.get('v_type'))}")
    lines.append(f"KV layers: {fmt(kv.get('layers'))} attention-cache layers")
    lines.append(f"KV per 1k context: {fmt(kv.get('mib_per_1k_context'), ' MiB')}")
    lines.append("")

    lines.append("Graph")
    lines.append("-----")
    lines.append(f"Nodes: {fmt(graph.get('nodes'))}")
    if graph.get("splits"):
        split_text = ", ".join(f"{count} with bs={bs}" for bs, count in sorted(graph["splits"].items(), key=lambda item: int(item[0])))
        lines.append(f"Splits: {split_text}")
    else:
        lines.append("Splits: unknown")
    lines.append(f"bs=1 splits per total layer: {fmt(graph.get('splits_bs1_per_total_layer'))} (heuristic)")
    lines.append("")

    lines.append("Timing Samples")
    lines.append("--------------")
    if report["timings"]:
        for timing in report["timings"]:
            active = timing.get("active_context", {}).get("value")
            method = timing.get("active_context", {}).get("method")
            lines.append(
                f"slot {fmt(timing.get('slot_id'))}, task {fmt(timing.get('task_id'))}: "
                f"prompt {fmt(timing.get('prompt_tokens_per_second'), ' tok/s')} over {fmt(timing.get('prompt_tokens'))} tokens; "
                f"generation {fmt(timing.get('eval_tokens_per_second'), ' tok/s')} over {fmt(timing.get('eval_tokens'))} tokens; "
                f"active context {fmt(active)} ({method or 'unknown'}), band {timing.get('context_band') or 'unknown'}"
            )
    else:
        lines.append("No timing samples found.")
    lines.append("")

    lines.append("Timing Summary")
    lines.append("--------------")
    by_band = report["timing_summary"].get("by_context_band", {})
    for band, summary in by_band.items():
        lines.append(
            f"{band}: count {summary.get('count')}, median generation {fmt(summary.get('median_generation_tps'), ' tok/s')}, "
            f"best generation {fmt(summary.get('best_generation_tps'), ' tok/s')}, median prompt {fmt(summary.get('median_prompt_tps'), ' tok/s')}"
        )
    lines.append("")

    lines.append("Long-Context Behavior")
    lines.append("---------------------")
    behavior = report["analysis"].get("long_context_behavior")
    if behavior:
        lines.append(
            f"{behavior['severity']}: {fmt(behavior.get('smallest_context_generation_tps'), ' tok/s')} at "
            f"{behavior.get('smallest_context')} tokens to {fmt(behavior.get('largest_context_generation_tps'), ' tok/s')} "
            f"at {behavior.get('largest_context')} tokens."
        )
    else:
        lines.append("Not enough measured samples across context depths.")
    lines.append("")

    lines.append("Recommendations")
    lines.append("---------------")
    if report["recommendations"]:
        for rec in report["recommendations"]:
            lines.append(f"- {rec['title']}: {rec['reason']}")
    else:
        lines.append("- No deterministic recommendations were triggered.")
    lines.append("")

    lines.append("Warnings / Unknowns")
    lines.append("-------------------")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    return "\n".join(lines)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# LLLogAnalyzer Report")
    lines.append("")
    disclaimer = "Grades are deterministic heuristics based on this tool's thresholds, not absolute model-quality or hardware-quality measurements."
    if report["source"].get("thresholds_path"):
        disclaimer += f" Thresholds: `{report['source']['thresholds_path']}`."
    lines.append(f"> {disclaimer}")
    lines.append("")
    model = report["model"]
    load = report["load"]
    lines.append("## Model")
    lines.extend(
        markdown_table(
            ["Field", "Value"],
            [
                ["Source", report["source"].get("name") or "stdin/clipboard"],
                ["Model file", model.get("filename") or "unknown"],
                ["Quant", source_value_text(model.get("quant", missing_source()))],
                ["Architecture", model.get("architecture") or "unknown"],
                ["Model type", model.get("model_type") or "unknown"],
                ["File type", model.get("file_type") or "unknown"],
                ["File size MiB", fmt(model.get("file_size_mib"))],
                ["Declared BPW", fmt(model.get("declared_bpw"))],
                ["Effective BPW", fmt(model.get("effective_bpw"))],
            ],
        )
    )
    lines.append("")
    lines.append("## Load")
    lines.extend(
        markdown_table(
            ["Field", "Value"],
            [
                ["Context requested", fmt(load.get("context_requested"))],
                ["Context actual", fmt(load.get("context_actual"))],
                ["Context trained", fmt(model.get("context_trained"))],
                ["Parallel", fmt(load.get("parallel"))],
                ["Batch", fmt(load.get("batch"))],
                ["Microbatch", fmt(load.get("microbatch"))],
                ["Flash attention", fmt(load.get("flash_attention"))],
                ["KV unified", fmt(load.get("kv_unified"))],
                ["Prompt cache", fmt(load["prompt_cache"].get("enabled"))],
                ["Prompt cache limit MiB", fmt(load["prompt_cache"].get("limit_mib"))],
            ],
        )
    )
    lines.append("")
    lines.append("## Memory")
    memory_rows = []
    for key, value in sorted(report["memory"]["model_buffers_mib"].items()):
        memory_rows.append(["model", key, fmt(value)])
    for key, value in sorted(report["memory"]["kv_buffers_mib"].items()):
        memory_rows.append(["KV", key, fmt(value)])
    for key, value in sorted(report["memory"]["rs_buffers_mib"].items()):
        memory_rows.append(["recurrent-state", key, fmt(value)])
    for key, value in sorted(report["memory"]["compute_buffers_mib"].items()):
        memory_rows.append(["compute", key, fmt(value)])
    lines.extend(markdown_table(["Type", "Location", "MiB"], memory_rows or [["unknown", "unknown", "unknown"]]))
    lines.append("")
    lines.append("## Placement")
    offload = report["offload"]
    lines.extend(
        markdown_table(
            ["Field", "Value"],
            [
                ["Offloaded layers", f"{fmt(offload.get('offloaded_layers'))}/{fmt(offload.get('total_layers'))}"],
                ["Offload ratio", fmt(offload.get("offload_ratio", {}).get("value"))],
                ["Whole-layer CPU fallback", fmt(offload.get("whole_layer_cpu_fallback_count"))],
                ["CPU-MoE placement", source_value_text(report["moe"].get("cpu_moe_placement", missing_source()))],
                ["Placement grade", f"{fmt(report['analysis'].get('placement_grade'))} (heuristic)"],
            ],
        )
    )
    lines.append("")
    lines.append("## KV Cache")
    kv = report["kv_cache"]
    lines.extend(
        markdown_table(
            ["Field", "Value"],
            [
                ["Total MiB", fmt(kv.get("total_mib"))],
                ["Cells", fmt(kv.get("cells"))],
                ["KV layers", fmt(kv.get("layers"))],
                ["K/V type", f"{fmt(kv.get('k_type'))} / {fmt(kv.get('v_type'))}"],
                ["MiB per 1k context", fmt(kv.get("mib_per_1k_context"))],
            ],
        )
    )
    lines.append("")
    lines.append("## Graph")
    graph = report["graph"]
    split_text = ", ".join(f"{count} with bs={bs}" for bs, count in sorted(graph["splits"].items(), key=lambda item: int(item[0]))) or "unknown"
    lines.extend(
        markdown_table(
            ["Field", "Value"],
            [
                ["Nodes", fmt(graph.get("nodes"))],
                ["Splits", split_text],
                ["bs=1 splits per total layer", fmt(graph.get("splits_bs1_per_total_layer"))],
            ],
        )
    )
    lines.append("")
    lines.append("## Timing Samples")
    timing_rows = []
    for timing in report["timings"]:
        active = timing.get("active_context", {}).get("value")
        timing_rows.append(
            [
                fmt(timing.get("slot_id")),
                fmt(timing.get("task_id")),
                fmt(active),
                timing.get("context_band") or "unknown",
                fmt(timing.get("prompt_tokens")),
                fmt(timing.get("prompt_tokens_per_second")),
                fmt(timing.get("eval_tokens")),
                fmt(timing.get("eval_tokens_per_second")),
                timing.get("generation_speed_grade") or "unknown",
            ]
        )
    lines.extend(
        markdown_table(
            ["Slot", "Task", "Active context", "Band", "Prompt tokens", "Prompt tok/s", "Eval tokens", "Eval tok/s", "Speed grade"],
            timing_rows or [["unknown"] * 9],
        )
    )
    lines.append("")
    lines.append("## Timing Summary")
    summary_rows = []
    for band, summary in report["timing_summary"].get("by_context_band", {}).items():
        summary_rows.append(
            [
                band,
                summary.get("count"),
                fmt(summary.get("median_generation_tps")),
                fmt(summary.get("best_generation_tps")),
                fmt(summary.get("worst_generation_tps")),
                fmt(summary.get("median_prompt_tps")),
            ]
        )
    lines.extend(markdown_table(["Band", "Count", "Median gen tok/s", "Best gen tok/s", "Worst gen tok/s", "Median prompt tok/s"], summary_rows or [["unknown"] * 6]))
    lines.append("")
    lines.append("## Recommendations")
    if report["recommendations"]:
        for rec in report["recommendations"]:
            lines.append(f"- **{rec['title']}** ({rec['severity']}): {rec['reason']}")
    else:
        lines.append("- No deterministic recommendations were triggered.")
    lines.append("")
    lines.append("## Warnings")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    return "\n".join(lines)


def comparable_bands(reports: List[Dict[str, Any]]) -> List[str]:
    band_sets = []
    for report in reports:
        bands = {
            timing.get("context_band")
            for timing in report.get("timings", [])
            if timing.get("context_band") and timing.get("eval_tokens_per_second") is not None
        }
        band_sets.append(bands)
    if not band_sets:
        return []
    common = set.intersection(*band_sets) if all(band_sets) else set()
    order = list(DEFAULT_THRESHOLDS["context_bands"].keys())
    return sorted(common, key=lambda band: order.index(band) if band in order else len(order))


def comparison_data(reports: List[Dict[str, Any]], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    common = comparable_bands(reports)
    winner = None
    winner_band = None
    warnings: List[str] = []
    if common:
        band_order = thresholds.get("context_bands", {})
        common_sorted = sorted(common, key=lambda band: band_order.get(band, [0, None])[0])
        winner_band = common_sorted[-1]
        scores = []
        for idx, report in enumerate(reports):
            band_summary = report["timing_summary"]["by_context_band"].get(winner_band, {})
            scores.append((band_summary.get("median_generation_tps"), idx))
        scores = [score for score in scores if score[0] is not None]
        if scores:
            scores.sort(reverse=True)
            winner = reports[scores[0][1]]["source"].get("name")
    else:
        warnings.append("Logs are not directly comparable because no common context band with generation timing was found.")

    rows = []
    for report in reports:
        for band, summary in report["timing_summary"].get("by_context_band", {}).items():
            rows.append(
                {
                    "filename": report["source"].get("name"),
                    "model_quant": report["model"]["quant"].get("value"),
                    "actual_context": report["load"].get("context_actual"),
                    "context_band": band,
                    "median_generation_tps": summary.get("median_generation_tps"),
                    "best_generation_tps": summary.get("best_generation_tps"),
                    "median_prompt_eval_tps": summary.get("median_prompt_tps"),
                    "offloaded_layers": report["offload"].get("offloaded_layers"),
                    "total_layers": report["offload"].get("total_layers"),
                    "cuda0_model_mib": report["memory"]["model_buffers_mib"].get("CUDA0"),
                    "cuda_host_model_mib": report["memory"]["model_buffers_mib"].get("CUDA_Host"),
                    "kv_total_mib": report["kv_cache"].get("total_mib"),
                    "graph_splits_bs1": report["graph"].get("splits_bs1"),
                    "graph_splits_bs1_per_total_layer": report["graph"].get("splits_bs1_per_total_layer"),
                    "placement_grade": report["analysis"].get("placement_grade"),
                    "speed_grade": speed_grade(summary.get("median_generation_tps"), None if band == "unknown" else band, thresholds),
                }
            )

    notes = []
    if winner:
        winner_report = next((report for report in reports if report["source"].get("name") == winner), None)
        if winner_report:
            winner_ratio = winner_report["offload"]["offload_ratio"].get("value")
            winner_host = winner_report["memory"]["model_buffers_mib"].get("CUDA_Host")
            winner_splits = winner_report["graph"].get("splits_bs1")
            for report in reports:
                if report is winner_report:
                    continue
                other_ratio = report["offload"]["offload_ratio"].get("value")
                other_host = report["memory"]["model_buffers_mib"].get("CUDA_Host")
                other_splits = report["graph"].get("splits_bs1")
                if other_ratio is not None and winner_ratio is not None and winner_ratio < other_ratio:
                    notes.append("The speed winner has worse GPU residency than another run. Measured generation speed is treated as primary.")
                    break
                if other_host is not None and winner_host is not None and winner_host > other_host:
                    notes.append("The speed winner has a larger CUDA host model buffer than another run. Measured generation speed is treated as primary.")
                    break
                if other_splits is not None and winner_splits is not None and winner_splits > other_splits:
                    notes.append("The speed winner has more bs=1 graph splits than another run. Measured generation speed is treated as primary.")
                    break

    return {
        "schema_version": SCHEMA_VERSION,
        "winner": winner,
        "winner_context_band": winner_band,
        "rows": rows,
        "notes": notes,
        "warnings": warnings,
        "reports": reports,
    }


def render_compare_human(data: Dict[str, Any]) -> str:
    lines = [
        "LLLogAnalyzer Comparison",
        "====================================",
        "",
        "Grades are deterministic heuristics based on this tool's thresholds, not absolute model-quality or hardware-quality measurements.",
        "",
    ]
    if data.get("winner"):
        lines.append(f"Speed winner: {data['winner']} in {data.get('winner_context_band')} context band")
    else:
        lines.append("Speed winner: not declared")
    lines.append("")
    rows = [
        [
            row.get("filename"),
            row.get("model_quant") or "unknown",
            fmt(row.get("actual_context")),
            row.get("context_band") or "unknown",
            fmt(row.get("median_generation_tps")),
            fmt(row.get("best_generation_tps")),
            fmt(row.get("median_prompt_eval_tps")),
            f"{fmt(row.get('offloaded_layers'))}/{fmt(row.get('total_layers'))}",
            fmt(row.get("cuda0_model_mib")),
            fmt(row.get("cuda_host_model_mib")),
            fmt(row.get("kv_total_mib")),
            fmt(row.get("graph_splits_bs1")),
            fmt(row.get("graph_splits_bs1_per_total_layer")),
            row.get("placement_grade") or "unknown",
            row.get("speed_grade") or "unknown",
        ]
        for row in data.get("rows", [])
    ]
    lines.extend(
        markdown_table(
            [
                "File",
                "Quant",
                "Actual ctx",
                "Band",
                "Median gen",
                "Best gen",
                "Median prompt",
                "Offload",
                "CUDA0 model",
                "CUDA host model",
                "KV total",
                "bs=1 splits",
                "splits/layer",
                "Placement",
                "Speed",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("- Normalized graph splits are heuristic and most useful when comparing similar model architectures.")
    for note in data.get("notes", []):
        lines.append(f"- {note}")
    for warning in data.get("warnings", []):
        lines.append(f"- {warning}")
    return "\n".join(lines)


def render_compare_markdown(data: Dict[str, Any]) -> str:
    lines = ["# LLLogAnalyzer Comparison", ""]
    lines.append("> Grades are deterministic heuristics based on this tool's thresholds, not absolute model-quality or hardware-quality measurements.")
    lines.append("")
    if data.get("winner"):
        lines.append(f"**Speed winner:** `{data['winner']}` in `{data.get('winner_context_band')}` context band")
    else:
        lines.append("**Speed winner:** not declared")
    lines.append("")
    rows = []
    for row in data.get("rows", []):
        rows.append(
            [
                row.get("filename"),
                row.get("model_quant") or "unknown",
                fmt(row.get("actual_context")),
                row.get("context_band") or "unknown",
                fmt(row.get("median_generation_tps")),
                fmt(row.get("best_generation_tps")),
                fmt(row.get("median_prompt_eval_tps")),
                f"{fmt(row.get('offloaded_layers'))}/{fmt(row.get('total_layers'))}",
                fmt(row.get("cuda0_model_mib")),
                fmt(row.get("cuda_host_model_mib")),
                fmt(row.get("kv_total_mib")),
                fmt(row.get("graph_splits_bs1")),
                fmt(row.get("graph_splits_bs1_per_total_layer")),
                row.get("placement_grade") or "unknown",
                row.get("speed_grade") or "unknown",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "File",
                "Quant",
                "Actual ctx",
                "Band",
                "Median gen",
                "Best gen",
                "Median prompt",
                "Offload",
                "CUDA0 model",
                "CUDA host model",
                "KV total",
                "bs=1 splits",
                "splits/layer",
                "Placement",
                "Speed",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Normalized graph splits are heuristic and most useful when comparing similar model architectures.")
    for note in data.get("notes", []):
        lines.append(f"- {note}")
    for warning in data.get("warnings", []):
        lines.append(f"- {warning}")
    return "\n".join(lines)


def read_file(path: str, encoding: str) -> str:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        raise UsageError(f"Unable to read input file {path!r}: {exc}") from exc


def read_stdin() -> str:
    return sys.stdin.read()


def read_clipboard(encoding: str) -> str:
    try:
        import tkinter  # type: ignore

        root = tkinter.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
        finally:
            root.destroy()
        if text:
            return text
    except Exception:
        pass

    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise UsageError(f"Clipboard unavailable: unable to launch PowerShell fallback: {exc}") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode(encoding, errors="replace").strip()
            raise UsageError(f"Clipboard unavailable: PowerShell Get-Clipboard failed. {stderr}")
        text = completed.stdout.decode(encoding, errors="replace")
        if text:
            return text
    raise UsageError("Clipboard unavailable or empty")


def select_input(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    if args.clipboard:
        return read_clipboard(args.encoding), "clipboard"
    if args.file:
        return read_file(args.file, args.encoding), args.file
    if args.input_file:
        return read_file(args.input_file, args.encoding), args.input_file
    if args.stdin or not sys.stdin.isatty():
        return read_stdin(), "stdin"
    raise UsageError("No input provided")


def render_report(report: Dict[str, Any], args: argparse.Namespace, thresholds: Optional[Dict[str, Any]] = None) -> str:
    if args.json:
        return render_json(report)
    if args.markdown:
        return render_markdown(report)
    if args.verbose:
        base = render_verbose_human(report)
    elif args.time:
        base = render_time_human(report)
    else:
        base = render_human(report)
    sections = [base]
    if args.context_table:
        sections.append(render_context_speed_table(report, thresholds or DEFAULT_THRESHOLDS))
    if args.speed_graph:
        sections.append(render_speed_graph(report))
    return "\n\n".join(sections)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLLogAnalyzer.py")
    parser.add_argument("input_file", nargs="?")
    parser.add_argument("--file")
    parser.add_argument("--clipboard", action="store_true")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs="+")
    parser.add_argument("--thresholds")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("-t", "--time", action="store_true")
    parser.add_argument("--context-table", "--speed-table", dest="context_table", action="store_true")
    parser.add_argument("--speed-graph", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def write_output(text: str, out_path: Optional[str], encoding: str) -> None:
    if out_path:
        with open(out_path, "w", encoding=encoding, errors="replace") as handle:
            handle.write(text)
            handle.write("\n")
    else:
        print(text)


def run(args: argparse.Namespace) -> int:
    if args.json and args.markdown:
        raise UsageError("Choose either --json or --markdown, not both")
    thresholds, thresholds_path = load_thresholds(args.thresholds)
    if args.compare:
        reports = []
        for path in args.compare:
            text = read_file(path, args.encoding)
            if not text.strip():
                raise UsageError(f"Input file {path!r} is empty")
            reports.append(parse_log(text, source_name=path, thresholds=thresholds, thresholds_path=thresholds_path))
        data = comparison_data(reports, thresholds)
        if args.json:
            output = render_json(data)
        elif args.markdown:
            output = render_compare_markdown(data)
        else:
            output = render_compare_human(data)
        write_output(output, args.out, args.encoding)
        return 1 if data.get("warnings") else 0

    text, source_name = select_input(args)
    if not text.strip():
        raise UsageError("Input is empty")
    report = parse_log(text, source_name=source_name, thresholds=thresholds, thresholds_path=thresholds_path)
    write_output(render_report(report, args, thresholds), args.out, args.encoding)
    no_strong_timing = not any(timing.get("eval_tokens_per_second") for timing in report["timings"])
    return 1 if report["warnings"] or no_strong_timing else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        try:
            return run(args)
        except UsageError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if str(exc) == "No input provided":
                parser.print_help(sys.stderr)
            return 2
        except Exception as exc:
            print(f"internal error: {exc}", file=sys.stderr)
            if getattr(args, "verbose", False):
                raise
            return 3
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


if __name__ == "__main__":
    sys.exit(main())
