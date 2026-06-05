import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "LLLogAnalyzer.py"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

spec = importlib.util.spec_from_file_location("LLLogAnalyzer", MODULE_PATH)
LLLogAnalyzer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = LLLogAnalyzer
spec.loader.exec_module(LLLogAnalyzer)


def read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


class ParserTests(unittest.TestCase):
    def test_iq3_fixture_parses_core_fields(self) -> None:
        report = LLLogAnalyzer.parse_log(read_fixture("minimal_iq3_s_132k_complete.log"), source_name="iq3.log")

        self.assertEqual(report["model"]["filename"], "Qwen3.6-35B-A3B-UD-IQ3_S.gguf")
        self.assertEqual(report["model"]["quant"]["value"], "IQ3_S")
        self.assertEqual(report["load"]["context_requested"], 132107)
        self.assertEqual(report["load"]["context_actual"], 132352)
        self.assertTrue(report["load"]["flash_attention"])
        self.assertEqual(report["offload"]["offloaded_layers"], 31)
        self.assertEqual(report["offload"]["total_layers"], 41)
        self.assertAlmostEqual(report["offload"]["offload_ratio"]["value"], 0.7561, places=4)
        self.assertEqual(report["memory"]["model_buffers_mib"]["CUDA0"], 9044.30)
        self.assertEqual(report["memory"]["model_buffers_mib"]["CUDA_Host"], 3590.51)
        self.assertEqual(report["graph"]["splits_bs1"], 22)
        self.assertEqual(report["analysis"]["placement_grade"], "A")
        self.assertEqual(report["warnings"], [])

    def test_iq3_fixture_parses_timing_summary(self) -> None:
        report = LLLogAnalyzer.parse_log(read_fixture("minimal_iq3_s_132k_complete.log"), source_name="iq3.log")

        self.assertEqual(len(report["timings"]), 1)
        timing = report["timings"][0]
        self.assertEqual(timing["task_id"], 0)
        self.assertEqual(timing["prompt_tokens"], 63310)
        self.assertEqual(timing["eval_tokens"], 860)
        self.assertEqual(timing["total_tokens"], 64170)
        self.assertEqual(timing["active_context"]["value"], 64170)
        self.assertEqual(timing["eval_tokens_per_second"], 12.46)
        self.assertEqual(report["timing_summary"]["overall_generation_tps"], 12.46)

    def test_q4_fixture_prefers_filename_quant_over_file_type(self) -> None:
        report = LLLogAnalyzer.parse_log(read_fixture("minimal_q4_k_xl_64k_complete.log"), source_name="q4.log")

        self.assertEqual(report["model"]["filename"], "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        self.assertEqual(report["model"]["file_type"], "Q4_K - Medium")
        self.assertEqual(report["model"]["quant"]["value"], "Q4_K_XL")
        self.assertEqual(report["offload"]["offloaded_layers"], 19)
        self.assertEqual(report["memory"]["model_buffers_mib"]["CUDA_Host"], 11662.73)
        self.assertEqual(report["graph"]["splits_bs1"], 36)
        self.assertEqual(report["timing_summary"]["overall_generation_tps"], 7.52)

    def test_json_output_is_stable_and_rounded(self) -> None:
        report = LLLogAnalyzer.parse_log(read_fixture("minimal_iq3_s_132k_complete.log"), source_name="iq3.log")

        first = LLLogAnalyzer.render_json(report)
        second = LLLogAnalyzer.render_json(report)
        data = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(data["offload"]["offload_ratio"]["value"], 0.7561)
        self.assertEqual(data["memory"]["model_buffers_mib"]["CUDA0"], 9044.3)

    def test_compare_uses_generation_speed_for_common_context_band(self) -> None:
        thresholds = json.loads(json.dumps(LLLogAnalyzer.DEFAULT_THRESHOLDS))
        iq3 = LLLogAnalyzer.parse_log(read_fixture("minimal_iq3_s_132k_complete.log"), source_name="iq3.log", thresholds=thresholds)
        q4 = LLLogAnalyzer.parse_log(read_fixture("minimal_q4_k_xl_64k_complete.log"), source_name="q4.log", thresholds=thresholds)

        comparison = LLLogAnalyzer.comparison_data([q4, iq3], thresholds)

        self.assertEqual(comparison["winner"], "iq3.log")
        self.assertEqual(comparison["winner_context_band"], "large")
        self.assertEqual(comparison["warnings"], [])

    def test_active_context_conflicts_are_auditable(self) -> None:
        snippet = """
slot update_slots: id  0 | task 7 | new prompt, n_ctx_slot = 64000, n_keep = 0, task.n_tokens = 10000
slot update_slots: id  0 | task 7 | prompt processing done, n_tokens = 10000, batch.n_tokens = 4
slot print_timing: id  0 | task 7 |
prompt eval time = 1000.00 ms / 10000 tokens (0.10 ms per token, 10000.00 tokens per second)
       eval time = 1000.00 ms / 100 tokens (10.00 ms per token, 100.00 tokens per second)
      total time = 2000.00 ms / 10100 tokens
slot      release: id  0 | task 7 | stop processing: n_tokens = 12000, truncated = 0
"""
        report = LLLogAnalyzer.parse_log(snippet, source_name="synthetic.log")

        timing = report["timings"][0]
        self.assertEqual(timing["active_context"]["method"], "stop_processing_n_tokens")
        self.assertEqual(timing["active_context"]["value"], 12000)
        self.assertEqual(timing["active_context"]["candidates"]["prompt_done_plus_eval"], 10100)
        self.assertIsNotNone(timing["active_context"]["conflict_warning"])


class CliTests(unittest.TestCase):
    def test_json_cli_output(self) -> None:
        fixture = FIXTURE_DIR / "minimal_iq3_s_132k_complete.log"
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--file", str(fixture), "--json"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertEqual(data["model"]["quant"]["value"], "IQ3_S")


class PublicFixtureHygieneTests(unittest.TestCase):
    def test_public_fixtures_do_not_include_local_user_paths_or_request_bodies(self) -> None:
        forbidden = [
            "C:" + "\\Users\\",
            "Received " + "request:",
            '"' + "messages" + '"',
            "reasoning" + "_content",
        ]
        for fixture in FIXTURE_DIR.glob("*.log"):
            text = fixture.read_text(encoding="utf-8", errors="replace")
            with self.subTest(fixture=fixture.name):
                for marker in forbidden:
                    self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
