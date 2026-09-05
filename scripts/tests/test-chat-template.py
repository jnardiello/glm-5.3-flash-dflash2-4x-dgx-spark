#!/usr/bin/env python3
"""Offline render fixtures for the pinned GLM-5.3 chat template (requires Jinja 3)."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import importlib.util

from jinja2.sandbox import ImmutableSandboxedEnvironment


TEMPLATE = Path(__file__).parent / "fixtures/chat_template-690b705.jinja"
spec = importlib.util.spec_from_file_location("render_chat_template", Path(__file__).parents[1] / "render_chat_template.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def render(messages: list[dict], tools: list[dict] | None = None, **kwargs: object) -> str:
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    environment.filters["tojson"] = lambda value, **options: json.dumps(
        value, ensure_ascii=options.get("ensure_ascii", True), separators=(",", ":")
    )
    template = environment.from_string(adapter.adapt(TEMPLATE.read_text()))
    return template.render(
        messages=messages,
        tools=tools or [],
        add_generation_prompt=kwargs.pop("add_generation_prompt", False),
        **kwargs,
    )


TOOLS = [
    {"type": "function", "function": {"name": "first", "description": "first", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "second", "description": "second", "parameters": {"type": "object"}}},
]


class ChatTemplateRenderTests(unittest.TestCase):
    def test_thinking_off_closes_reasoning_before_generation(self) -> None:
        output = render([{"role": "user", "content": "2+2?"}], add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(output.endswith("<|assistant|><think></think>"))

    def test_thinking_on_preserves_upstream_prefix(self) -> None:
        output = render([{"role": "user", "content": "2+2?"}], add_generation_prompt=True, enable_thinking=True)
        self.assertTrue(output.endswith("<|assistant|><think>"))

    def test_unexpected_upstream_prefix_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            adapter.adapt("unknown template")

    def test_01_basic_generation_prompt(self) -> None:
        output = render([{"role": "user", "content": "ping"}], add_generation_prompt=True)
        self.assertTrue(output.startswith("[gMASK]<sop><|system|>Reasoning Effort: Max"))
        self.assertIn("<|user|>ping<|assistant|><think>", output)

    def test_02_deferred_tool_is_not_eagerly_emitted(self) -> None:
        tools = TOOLS + [{"type": "function", "function": {"name": "later", "defer_loading": True}}]
        output = render([{"role": "user", "content": "use tools"}], tools)
        self.assertIn('"name": "first"', output)
        self.assertNotIn('"name": "later"', output)

    def test_03_tool_results_are_sorted_to_call_order(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "function": {"name": "first", "arguments": {}}},
                {"id": "b", "function": {"name": "second", "arguments": {}}},
            ]},
            {"role": "tool", "tool_call_id": "b", "content": "result-B"},
            {"role": "tool", "tool_call_id": "a", "content": "result-A"},
        ]
        output = render(messages, TOOLS)
        self.assertLess(output.index("result-A"), output.index("result-B"))

    def test_04_duplicate_result_ids_preserve_input_order(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "function": {"name": "first", "arguments": {}}}
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "first-seen"},
            {"role": "tool", "tool_call_id": "a", "content": "second-seen"},
        ]
        output = render(messages, TOOLS)
        self.assertLess(output.index("first-seen"), output.index("second-seen"))

    def test_05_unknown_result_id_preserves_input_order(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "function": {"name": "first", "arguments": {}}}
            ]},
            {"role": "tool", "tool_call_id": "unknown", "content": "unknown-result"},
            {"role": "tool", "tool_call_id": "a", "content": "known-result"},
        ]
        output = render(messages, TOOLS)
        self.assertLess(output.index("unknown-result"), output.index("known-result"))

    def test_06_multimodal_and_null_content(self) -> None:
        output = render([{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": "ignored"}
        ]}, {"role": "assistant", "content": None}])
        self.assertIn("look<|begin_of_image|><|image|><|end_of_image|>", output)

    def test_07_batched_tool_outputs_are_sorted(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "function": {"name": "first", "arguments": {}}},
                {"id": "b", "function": {"name": "second", "arguments": {}}},
            ]},
            {"role": "tool", "content": [
                {"tool_call_id": "b", "output": "batch-B"},
                {"tool_call_id": "a", "output": "batch-A"},
            ]},
        ]
        output = render(messages, TOOLS)
        self.assertLess(output.index("batch-A"), output.index("batch-B"))

    def test_08_clear_thinking_keeps_only_latest_assistant_reasoning(self) -> None:
        messages = [
            {"role": "assistant", "content": "old", "reasoning_content": "old-thought"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "answer", "reasoning_content": "new-thought"},
        ]
        output = render(messages, clear_thinking=True)
        self.assertNotIn("old-thought", output)
        self.assertIn("<think>new-thought</think>", output)


if __name__ == "__main__":
    unittest.main()
