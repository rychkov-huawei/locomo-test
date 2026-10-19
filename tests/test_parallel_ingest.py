"""Parallel ingest: completion, worker limits, and oGMemory token accounting."""

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from locomo_test.config import Config, OgmemEnv, SessionConfig, SessionPolicy
from locomo_test.eval import load_memory_token_totals, run_ingest


def make_sample(sample_id, n_sessions=3):
    conv = {"speaker_a": "Alice", "speaker_b": "Bob"}
    for i in range(1, n_sessions + 1):
        conv[f"session_{i}"] = [{"speaker": "Alice", "text": f"message {i}"}]
        conv[f"session_{i}_date_time"] = f"{i} pm on {i} May, 2023"
    return {"sample_id": sample_id, "conversation": conv, "qa": []}


class FakeOgmemCounter:
    """Cumulative token counter shared by all concurrent ingest workers."""

    def __init__(self, per_session=100, delay=0.05):
        self.state = {"llm": {"input_tokens": 1000, "output_tokens": 500,
                              "total_tokens": 1500, "calls": 0},
                      "embedding": {"total_tokens": 700, "calls": 0}}
        self.lock = threading.Lock()
        self.per_session = per_session
        self.delay = delay

    def snapshot(self, *_args, **_kwargs):
        with self.lock:
            return copy.deepcopy(self.state)

    def ingest(self, *_args, **_kwargs):
        time.sleep(self.delay)  # force concurrent sessions to overlap
        with self.lock:
            self.state["llm"]["total_tokens"] += self.per_session
            self.state["llm"]["input_tokens"] += self.per_session
            self.state["llm"]["calls"] += 1
            self.state["embedding"]["total_tokens"] += 10
            self.state["embedding"]["calls"] += 1
        return {"protocol": "after-turn-wait-v1", "fingerprint": "x",
                "chunk_count": 1, "archive_ids": ["a"]}


class ParallelIngestTests(unittest.TestCase):
    def setUp(self):
        self.samples = [make_sample(0, 4), make_sample(1, 2)]

    def _records(self, directory):
        return json.loads(Path(directory, ".ingest_record.json").read_text())

    def test_parallel_gateway_ingest_completes_all_sessions(self):
        seen = []
        lock = threading.Lock()
        active = [0]
        peak = [0]

        def fake_send(*_args, **_kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                seen.append(threading.current_thread().name)
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return "OK", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        cfg = Config(name="unit", memory_mode="none", parallel=4)
        with tempfile.TemporaryDirectory() as directory, patch(
            "locomo_test.eval.load_locomo_data", return_value=self.samples
        ), patch(
            "locomo_test.eval.send_message_with_retry", side_effect=fake_send
        ), patch(
            "locomo_test.eval.get_session_id_from_key", return_value=None
        ):
            results, memory_totals, token_totals = run_ingest(cfg, directory)

            self.assertEqual(len(results), 6)
            self.assertGreater(peak[0], 1)  # sessions really ran concurrently
            records = self._records(directory)
            self.assertEqual(len(records), 6)
            self.assertTrue(all(r["success"] for r in records.values()))
            self.assertEqual(token_totals["total_tokens"], 6 * 15)
            self.assertFalse(
                Path(directory, ".ingest_failures.json").exists()
                and json.loads(Path(directory, ".ingest_failures.json").read_text())
            )

    def test_parallel_ogmem_reports_global_token_delta(self):
        counter = FakeOgmemCounter(per_session=100, delay=0.05)
        cfg = Config(name="unit", memory_mode="ogmem", parallel=4, ogmem=OgmemEnv())
        with tempfile.TemporaryDirectory() as directory, patch(
            "locomo_test.eval.load_locomo_data", return_value=self.samples
        ), patch(
            "locomo_test.eval.query_ogmem_token_stats", side_effect=counter.snapshot
        ), patch(
            "locomo_test.eval.ingest_chunks", side_effect=counter.ingest
        ), patch(
            "locomo_test.eval.wait_for_index", return_value={"idle": True}
        ):
            _results, memory_totals, _tokens = run_ingest(cfg, directory)

            # Six sessions overlap pairwise; summing per-session before/after
            # windows would count the same concurrent work several times over.
            self.assertEqual(memory_totals["llm_total"], 6 * 100)
            self.assertEqual(memory_totals["embedding"], 6 * 10)
            persisted = load_memory_token_totals(directory)
            self.assertEqual(persisted["llm_total"], 6 * 100)
            for entry in self._records(directory).values():
                # Per-session deltas are not recorded under parallel ingest.
                self.assertEqual(entry["meta"]["memory_token_usage"], {})

    def test_serial_ogmem_still_records_per_session_deltas(self):
        counter = FakeOgmemCounter(per_session=100, delay=0)
        cfg = Config(name="unit", memory_mode="ogmem", parallel=1, ogmem=OgmemEnv())
        with tempfile.TemporaryDirectory() as directory, patch(
            "locomo_test.eval.load_locomo_data", return_value=self.samples
        ), patch(
            "locomo_test.eval.query_ogmem_token_stats", side_effect=counter.snapshot
        ), patch(
            "locomo_test.eval.ingest_chunks", side_effect=counter.ingest
        ), patch(
            "locomo_test.eval.wait_for_index", return_value={"idle": True}
        ):
            _results, memory_totals, _tokens = run_ingest(cfg, directory)

            self.assertEqual(memory_totals["llm_total"], 6 * 100)
            entries = list(self._records(directory).values())
            self.assertTrue(all(e["meta"]["memory_token_usage"].get("llm_total") == 100
                                for e in entries))
            self.assertIsNone(load_memory_token_totals(directory))

    def test_shared_policy_gateway_ingest_stays_serial(self):
        active = [0]
        peak = [0]
        lock = threading.Lock()

        def fake_send(*_args, **_kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return "OK", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        cfg = Config(name="unit", memory_mode="none", parallel=4,
                     session=SessionConfig(policy=SessionPolicy.SHARED))
        with tempfile.TemporaryDirectory() as directory, patch(
            "locomo_test.eval.load_locomo_data", return_value=self.samples
        ), patch(
            "locomo_test.eval.send_message_with_retry", side_effect=fake_send
        ), patch(
            "locomo_test.eval.get_session_id_from_key", return_value=None
        ):
            results, _memory, _tokens = run_ingest(cfg, directory)

            self.assertEqual(len(results), 6)
            self.assertEqual(peak[0], 1)  # shared session is ingested in order


if __name__ == "__main__":
    unittest.main()
