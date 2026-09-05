#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# (c) 2026 Jacopo Nardiello. See LICENSE and CREDITS.md.
"""CPU-only tests for the adaptive-k policy (no vLLM needed).

    python3 scripts/node/patches/test_adaptive_k_policy.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_k_scheduler import (  # noqa: E402
    AdaptiveKConfig, AdaptiveKPolicy, DraftedRing, assign_lengths, filter_candidates,
    placeholder_len, should_observe,
)


def policy(**kw) -> AdaptiveKPolicy:
    return AdaptiveKPolicy(AdaptiveKConfig(**kw))


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = AdaptiveKConfig()
        self.assertEqual((c.k_lo, c.k_hi, c.up, c.down, c.alpha, c.seed, c.mode, c.signal, c.log_every),
                         (3, 5, 0.58, 0.42, 0.15, 1.0, "per-request", "pos", 200))
        self.assertTrue(c.enabled)

    def test_env_parsing(self):
        env = {
            "VLLM_ADAPTIVE_K_ENABLE": "1", "VLLM_ADAPTIVE_K_LO": "2", "VLLM_ADAPTIVE_K_HI": "7",
            "VLLM_ADAPTIVE_K_UP": "0.7", "VLLM_ADAPTIVE_K_DOWN": "0.3", "VLLM_ADAPTIVE_K_ALPHA": "0.5",
            "VLLM_ADAPTIVE_K_SEED": "0.0", "VLLM_ADAPTIVE_K_MODE": "batch-uniform",
            "VLLM_ADAPTIVE_K_SIGNAL": "mean", "VLLM_ADAPTIVE_K_LOG_EVERY": "0",
        }
        c = AdaptiveKConfig.from_env(env)
        self.assertEqual((c.k_lo, c.k_hi, c.up, c.down, c.alpha, c.seed, c.mode, c.signal, c.log_every),
                         (2, 7, 0.7, 0.3, 0.5, 0.0, "batch-uniform", "mean", 0))
        self.assertFalse(AdaptiveKConfig.from_env({"VLLM_ADAPTIVE_K_ENABLE": "0"}).enabled)
        # empty values fall back to defaults; os.environ is restored afterwards
        c2 = AdaptiveKConfig.from_env({"VLLM_ADAPTIVE_K_LO": ""})
        self.assertEqual(c2.k_lo, 3)
        self.assertNotIn("VLLM_ADAPTIVE_K_LO", os.environ)

    def test_validation(self):
        with self.assertRaises(ValueError):
            AdaptiveKConfig(k_lo=5, k_hi=3)
        with self.assertRaises(ValueError):
            AdaptiveKConfig(up=0.4, down=0.5)
        with self.assertRaises(ValueError):
            AdaptiveKConfig(mode="random")
        with self.assertRaises(ValueError):
            AdaptiveKConfig(signal="median")
        with self.assertRaises(ValueError):
            AdaptiveKConfig(alpha=0.0)
        with self.assertRaises(ValueError):
            AdaptiveKConfig.from_env({"VLLM_ADAPTIVE_K_SEED": "1.5"})


class TestSignal(unittest.TestCase):
    def test_pos_signal_is_the_k_lo_prefix_indicator(self):
        p = policy()  # default signal "pos"
        self.assertEqual(p.signal(0, 5), 0.0)
        self.assertEqual(p.signal(2, 5), 0.0)   # positions 1-2 accepted, 3 rejected
        self.assertEqual(p.signal(3, 5), 1.0)
        self.assertEqual(p.signal(5, 5), 1.0)
        self.assertEqual(p.signal(3, 3), 1.0)   # same value whatever k was verified
        self.assertEqual(p.signal(2, 2), 1.0)   # short draft: its own length
        self.assertEqual(p.signal(1, 2), 0.0)

    def test_mean_signal_uses_first_k_lo_positions(self):
        p = policy(signal="mean")
        self.assertEqual(p.signal(0, 5), 0.0)
        self.assertAlmostEqual(p.signal(1, 5), 1 / 3)
        self.assertAlmostEqual(p.signal(3, 5), 1.0)
        self.assertAlmostEqual(p.signal(5, 5), 1.0)  # positions beyond k_lo do not count
        self.assertAlmostEqual(p.signal(3, 3), 1.0)  # same value whatever k was verified
        self.assertAlmostEqual(p.signal(1, 2), 0.5)  # short draft scaled to its own length
        self.assertEqual(p.signal(-4, 5), 0.0)


class TestPolicy(unittest.TestCase):
    def test_seed_optimistic_starts_high(self):
        p = policy()
        self.assertEqual(p.decide("a"), 5)
        self.assertEqual(p.ema("a"), 1.0)

    def test_seed_low_starts_low(self):
        p = policy(seed=0.0)
        self.assertEqual(p.decide("a"), 3)

    def test_seed_in_band_starts_high(self):
        p = policy(seed=0.5)
        self.assertEqual(p.decide("a"), 5)

    def test_ema_update(self):
        p = policy(alpha=0.5, seed=1.0)
        self.assertAlmostEqual(p.observe("a", 0, 5), 0.5)
        self.assertAlmostEqual(p.observe("a", 0, 5), 0.25)
        self.assertAlmostEqual(p.observe("a", 3, 5), 0.625)
        self.assertEqual(p.counters["observations"], 3)

    def test_zero_drafts_is_a_no_op(self):
        p = policy(alpha=0.5)
        p.observe("a", 0, 0)
        self.assertEqual(p.ema("a"), 1.0)
        self.assertEqual(p.counters["observations"], 0)

    def test_prose_like_request_drops_to_k_lo_and_stays(self):
        p = policy()  # defaults: pos signal, alpha 0.15, band 0.42-0.58
        r = "prose"
        for _ in range(6):  # six steps where position 3 was rejected
            p.observe(r, 1, 5)
        self.assertLessEqual(p.ema(r), 0.42)   # 0.85**6 = 0.377
        self.assertEqual(p.decide(r), 3)
        # two good steps land inside the hysteresis band: stays low (0.471, 0.550)
        p.observe(r, 3, 5)
        self.assertEqual(p.decide(r), 3)
        p.observe(r, 3, 5)
        self.assertTrue(0.42 < p.ema(r) < 0.58)
        self.assertEqual(p.decide(r), 3)
        # a third one crosses `up`: back to k_hi (0.617)
        p.observe(r, 3, 5)
        self.assertGreaterEqual(p.ema(r), 0.58)
        self.assertEqual(p.decide(r), 5)
        self.assertEqual(p.counters["switches"], 2)

    def test_structured_like_request_stays_high(self):
        p = policy()
        r = "json"
        for _ in range(20):
            p.observe(r, 5, 5)
        self.assertEqual(p.decide(r), 5)
        self.assertEqual(p.counters["switches"], 0)

    def test_hysteresis_band_keeps_current_k(self):
        p = policy(alpha=1.0, signal="mean")  # EMA == last signal, easy to steer
        r = "x"
        p.observe(r, 0, 5)              # ema 0 -> low
        self.assertEqual(p.decide(r), 3)
        p.observe(r, 1, 2)              # signal 0.5, inside (0.42, 0.58): stays low
        self.assertEqual(p.decide(r), 3)
        p.observe(r, 3, 5)              # 1.0 -> high
        self.assertEqual(p.decide(r), 5)
        p.observe(r, 1, 2)              # 0.5 again: stays high this time
        self.assertEqual(p.decide(r), 5)
        self.assertEqual(p.counters["switches"], 2)

    def test_batch_uniform_needs_every_request_high(self):
        p = policy(alpha=1.0)
        p.observe("hi", 3, 5)
        p.observe("lo", 0, 5)
        self.assertEqual(p.decide_batch(["hi", "lo"]), 3)
        self.assertEqual(p.decide_batch(["hi"]), 5)
        self.assertEqual(p.decide_batch([]), 3)
        self.assertEqual(p.counters["steps_uniform_lo"], 2)
        self.assertEqual(p.counters["steps_uniform_hi"], 1)

    def test_per_request_independent(self):
        p = policy(alpha=1.0)
        p.observe("hi", 3, 5)
        p.observe("lo", 0, 5)
        self.assertEqual(p.decide("hi"), 5)
        self.assertEqual(p.decide("lo"), 3)

    def test_eviction(self):
        p = policy(alpha=1.0)
        p.observe("a", 0, 5)
        self.assertEqual(p.decide("a"), 3)
        p.evict("a")
        self.assertEqual(p.tracked(), 0)
        self.assertEqual(p.decide("a"), 5)  # re-seeded, optimistic again
        p.evict("never-seen")  # no error

    def test_counters_line(self):
        p = policy()
        p.decide("a")
        line = p.counters_line()
        self.assertIn("tracked=1", line)
        self.assertIn("decisions lo/hi=0/1", line)

    def test_disabled_config_is_representable(self):
        c = AdaptiveKConfig(enabled=False)
        self.assertIn("enabled=0", c.describe())


class TestObserveGating(unittest.TestCase):
    """The pure gate the scheduler subclass applies before feeding the EMA."""

    def test_real_drafts_and_output_are_observed(self):
        self.assertTrue(should_observe("a", {"a"}, 5, True, 1))

    def test_resumed_request_placeholders_are_not_observed(self):
        # a request promoted out of the waiting queue gets [-1]*k placeholders and is
        # not in the drafted set of that step (scheduler.py:1085-1088)
        self.assertFalse(should_observe("resumed", {"a"}, 5, True, 1))

    def test_no_drafts_not_observed(self):
        self.assertFalse(should_observe("a", {"a"}, 0, True, 1))

    def test_empty_output_not_observed_unless_zero_sampled(self):
        self.assertFalse(should_observe("a", {"a"}, 5, False, 1))
        self.assertTrue(should_observe("a", {"a"}, 5, False, 0))

    def test_stale_output_not_observed(self):
        self.assertFalse(should_observe("a", {"a"}, 5, True, 1, is_stale=True))
        self.assertFalse(should_observe("a", {"a"}, 5, True, 1, is_stale=True, drop_stale=True))

    def test_kv_load_failure_step_not_observed(self):
        self.assertFalse(should_observe("a", {"a"}, 5, True, 1, kv_load_failed=True))

    def test_drafted_set_is_the_ring_union(self):
        # what the subclass does: the union of the last three hand-outs is the drafted set
        ring = DraftedRing(3)
        ring.push({"a", "b"})
        self.assertTrue(should_observe("b", ring.union(), 3, True, 1))
        for _ in range(3):
            ring.push(set())
        self.assertFalse(should_observe("b", ring.union(), 3, True, 1))


class TestAsyncPlaceholderSizing(unittest.TestCase):
    """What _update_after_schedule does with the policy's k on the async path."""

    def test_clamp_to_engine_drafts(self):
        self.assertEqual(placeholder_len(5, 5), 5)
        self.assertEqual(placeholder_len(7, 5), 5)   # never more than the engine drafts
        self.assertEqual(placeholder_len(3, 5), 3)
        self.assertEqual(placeholder_len(3, 2), 2)   # engine drafting fewer than k_lo
        self.assertEqual(placeholder_len(-1, 5), 0)

    def test_per_request_placeholder_lengths(self):
        p = policy(alpha=1.0)
        p.observe("code", 3, 5)   # first three accepted -> high
        p.observe("prose", 0, 5)  # nothing accepted -> low
        lens = {r: placeholder_len(p.decide(r), 5) for r in ("code", "prose", "new")}
        self.assertEqual(lens, {"code": 5, "prose": 3, "new": 5})

    def test_batch_uniform_placeholder_length(self):
        p = policy(alpha=1.0)
        p.observe("code", 3, 5)
        p.observe("prose", 0, 5)
        self.assertEqual(placeholder_len(p.decide_batch(["code", "prose"]), 5), 3)
        self.assertEqual(placeholder_len(p.decide_batch(["code", "new"]), 5), 5)

    def test_table_k_is_neutralised(self):
        # the dynamic-SD table would make the stock AsyncScheduler use 3 drafts at batch 2-6;
        # the override sizes placeholders from the policy alone, only clamped to engine_k
        p = policy(alpha=1.0)
        for r in ("a", "b", "c"):
            p.observe(r, 3, 5)
        self.assertEqual([placeholder_len(p.decide(r), 5) for r in ("a", "b", "c")], [5, 5, 5])

    def test_drafted_ring_covers_the_async_lookahead(self):
        # core.py: schedule(N) -> appendleft -> pop -> update_from_output(N-1); batch queue of 2.
        # The step reported at the iteration of N got its placeholders at the iteration of N-2,
        # so the ring must keep three hand-outs: N-2, N-1 and N.
        ring = DraftedRing(3)
        ring.push({"old"})          # iteration N-2: placeholders for the step reported now
        ring.push({"mid"})          # iteration N-1
        ring.push({"new"})          # iteration N (just handed out)
        self.assertTrue(should_observe("old", ring.union(), 5, True, 1))
        self.assertTrue(should_observe("mid", ring.union(), 5, True, 1))
        self.assertTrue(should_observe("new", ring.union(), 5, True, 1))
        ring.push({"newer"})        # iteration N+1 evicts N-2
        self.assertFalse(should_observe("old", ring.union(), 5, True, 1))
        self.assertEqual(len(ring), 3)
        self.assertIn("mid", ring)
        with self.assertRaises(ValueError):
            DraftedRing(0)


class TestPlaceholderSizingGlue(unittest.TestCase):
    """The pure parts of _ak_size_placeholders."""

    def test_filter_candidates(self):
        items = [
            ("running", False, False, True),
            ("prefill-chunk", False, True, True),   # base skipped it: keep whatever it has
            ("no-placeholders", False, False, False),
            ("finished", True, False, True),
        ]
        self.assertEqual(filter_candidates(items), ["running"])
        self.assertEqual(filter_candidates([]), [])

    def test_assign_lengths_per_request(self):
        p = policy(alpha=1.0)
        p.observe("code", 3, 5)
        p.observe("prose", 0, 5)
        self.assertEqual(assign_lengths(p, ["code", "prose", "new"], "per-request", 5),
                         {"code": 5, "prose": 3, "new": 5})
        self.assertEqual(assign_lengths(p, [], "per-request", 5), {})

    def test_assign_lengths_batch_uniform(self):
        p = policy(alpha=1.0)
        p.observe("code", 3, 5)
        p.observe("prose", 0, 5)
        self.assertEqual(assign_lengths(p, ["code", "prose"], "batch-uniform", 5),
                         {"code": 3, "prose": 3})
        self.assertEqual(assign_lengths(p, ["code", "new"], "batch-uniform", 5),
                         {"code": 5, "new": 5})

    def test_assign_lengths_clamps_to_engine(self):
        p = policy()
        self.assertEqual(assign_lengths(p, ["a"], "per-request", 4), {"a": 4})
        self.assertEqual(assign_lengths(p, ["a", "b"], "batch-uniform", 2), {"a": 2, "b": 2})


class TestCalibration(unittest.TestCase):
    """Feed the measured position-3 marginals as Bernoulli streams (deterministic seed)."""

    @staticmethod
    def fraction_low(p_accept: float, requests: int = 50, steps: int = 500, tail: int = 300) -> float:
        import random
        rng = random.Random(20260904)
        pol = policy()
        low = total = 0
        for r in range(requests):
            rid = f"r{r}"
            for t in range(steps):
                pol.observe(rid, 3 if rng.random() < p_accept else 1, 5)
                k = pol.decide(rid)
                if t >= steps - tail:
                    total += 1
                    low += k == 3
        return low / total

    def test_prose_marginal_0_39_mostly_k_lo(self):
        self.assertGreater(self.fraction_low(0.39), 0.75)

    def test_borderline_0_44_mostly_k_lo(self):
        self.assertGreater(self.fraction_low(0.44), 0.6)

    def test_code_marginal_0_60_mostly_k_hi(self):
        self.assertLess(self.fraction_low(0.60), 0.4)

    def test_structured_marginal_0_89_always_k_hi(self):
        self.assertLess(self.fraction_low(0.89), 0.02)

    def test_time_to_switch_about_one_over_alpha(self):
        p = policy()
        r = "x"
        n = 0
        while p.decide(r) == 5:      # optimistic start, then all-rejected steps
            p.observe(r, 0, 5)
            n += 1
            self.assertLess(n, 50)
        self.assertLessEqual(n, 7)   # 0.85**6 = 0.377 <= 0.42 -> 6 steps


if __name__ == "__main__":
    unittest.main(verbosity=1)
