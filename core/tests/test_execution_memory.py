"""Tests for ADAPT.md execution memory.

Validates:
  - write_adapt / read_single_adapt file I/O
  - read_adapt_narrative ordering and deduplication
  - _build_deterministic_adapt output format
  - generate_node_adapt with mocked LLM and fallback
  - Integration with prompt_composer (build_narrative, build_transition_marker)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from framework.graph.execution_memory import (
    _build_deterministic_adapt,
    _format_outputs_summary,
    generate_node_adapt,
    read_adapt_narrative,
    read_single_adapt,
    write_adapt,
)
from framework.graph.goal import Goal
from framework.graph.node import NodeResult, NodeSpec, SharedMemory

# ---------------------------------------------------------------------------
# File I/O: write_adapt / read_single_adapt
# ---------------------------------------------------------------------------


class TestAdaptFileIO:
    def test_write_and_read(self, tmp_path: Path):
        write_adapt(tmp_path, "research", "## Accomplished\nGathered data.")
        content = read_single_adapt(tmp_path, "research")
        assert content == "## Accomplished\nGathered data."

    def test_read_missing_returns_none(self, tmp_path: Path):
        assert read_single_adapt(tmp_path, "nonexistent") is None

    def test_write_overwrites_on_revisit(self, tmp_path: Path):
        write_adapt(tmp_path, "research", "First visit")
        write_adapt(tmp_path, "research", "Second visit")
        assert read_single_adapt(tmp_path, "research") == "Second visit"

    def test_creates_parent_dirs(self, tmp_path: Path):
        write_adapt(tmp_path, "deep-node", "content")
        assert (tmp_path / "conversations" / "deep-node" / "ADAPT.md").exists()


# ---------------------------------------------------------------------------
# read_adapt_narrative
# ---------------------------------------------------------------------------


class TestReadAdaptNarrative:
    def test_empty_path(self, tmp_path: Path):
        assert read_adapt_narrative(tmp_path, []) == ""

    def test_basic_order(self, tmp_path: Path):
        write_adapt(tmp_path, "a", "Phase A done")
        write_adapt(tmp_path, "b", "Phase B done")
        write_adapt(tmp_path, "c", "Phase C done")
        narrative = read_adapt_narrative(tmp_path, ["a", "b", "c"])
        # All three present, in order
        assert "Phase A done" in narrative
        assert "Phase B done" in narrative
        assert "Phase C done" in narrative
        assert narrative.index("Phase A") < narrative.index("Phase B") < narrative.index("Phase C")

    def test_deduplication_keeps_last_occurrence(self, tmp_path: Path):
        write_adapt(tmp_path, "a", "Latest A")
        write_adapt(tmp_path, "b", "Phase B")
        # Path visits a twice (feedback loop): [a, b, a]
        narrative = read_adapt_narrative(tmp_path, ["a", "b", "a"])
        # "a" should appear once, after "b" (at its last position)
        assert narrative.count("Latest A") == 1
        assert narrative.index("Phase B") < narrative.index("Latest A")

    def test_max_entries(self, tmp_path: Path):
        for i in range(15):
            write_adapt(tmp_path, f"node_{i}", f"Phase {i} done")
        path = [f"node_{i}" for i in range(15)]
        narrative = read_adapt_narrative(tmp_path, path, max_entries=10)
        # Should include node_5..node_14 (last 10), not node_0..node_4
        assert "Phase 5 done" in narrative
        assert "Phase 14 done" in narrative
        assert "Phase 4 done" not in narrative

    def test_missing_files_skipped(self, tmp_path: Path):
        write_adapt(tmp_path, "a", "Phase A")
        # "b" has no ADAPT.md
        write_adapt(tmp_path, "c", "Phase C")
        narrative = read_adapt_narrative(tmp_path, ["a", "b", "c"])
        assert "Phase A" in narrative
        assert "Phase C" in narrative

    def test_joined_by_separator(self, tmp_path: Path):
        write_adapt(tmp_path, "a", "Phase A")
        write_adapt(tmp_path, "b", "Phase B")
        narrative = read_adapt_narrative(tmp_path, ["a", "b"])
        assert "\n\n---\n\n" in narrative


# ---------------------------------------------------------------------------
# _format_outputs_summary
# ---------------------------------------------------------------------------


class TestFormatOutputsSummary:
    def test_empty(self):
        assert _format_outputs_summary({}) == "(no outputs)"

    def test_simple(self):
        result = _format_outputs_summary({"key1": "value1", "key2": 42})
        assert "key1: value1" in result
        assert "key2: 42" in result

    def test_none_skipped(self):
        result = _format_outputs_summary({"key1": "value1", "key2": None})
        assert "key1" in result
        assert "key2" not in result

    def test_long_values_truncated(self):
        result = _format_outputs_summary({"key1": "x" * 500})
        assert "..." in result
        assert len(result) < 500


# ---------------------------------------------------------------------------
# _build_deterministic_adapt
# ---------------------------------------------------------------------------


def _make_node_spec(
    node_id: str = "test",
    name: str = "Test Node",
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        name=name,
        description="A test node",
        node_type="event_loop",
        input_keys=[],
        output_keys=["result"],
    )


def _make_goal() -> Goal:
    return Goal(
        id="test-goal",
        name="Test Goal",
        description="Complete the test task",
        success_criteria=[],
    )


class TestDeterministicAdapt:
    def test_success(self):
        node_spec = _make_node_spec()
        result = NodeResult(success=True, output={"result": "done"})
        md = _build_deterministic_adapt(node_spec, result, visit_number=1)
        assert "## Accomplished" in md
        assert "## Key Decisions" in md
        assert "## Handoff" in md
        assert "succeeded" in md
        assert "result" in md

    def test_failure(self):
        node_spec = _make_node_spec()
        result = NodeResult(success=False, output={}, error="Something broke")
        md = _build_deterministic_adapt(node_spec, result, visit_number=2)
        assert "failed" in md
        assert "visit 2" in md


# ---------------------------------------------------------------------------
# generate_node_adapt (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_node_adapt_deterministic_fallback():
    """When LLM is unavailable, should fall back to deterministic markdown."""
    node_spec = _make_node_spec()
    result = NodeResult(success=True, output={"result": "done"})
    memory = SharedMemory()
    goal = _make_goal()

    # Pass llm=None to force fallback
    md = await generate_node_adapt(
        llm=None,
        node_spec=node_spec,
        result=result,
        memory=memory,
        execution_path=["test"],
        goal=goal,
        visit_number=1,
        previous_adapts=[],
    )

    assert "## Accomplished" in md
    assert "## Key Decisions" in md
    assert "## Handoff" in md


@pytest.mark.asyncio
async def test_generate_node_adapt_with_mocked_llm():
    """When LLM returns markdown, should return it as-is."""
    node_spec = _make_node_spec()
    result = NodeResult(success=True, output={"result": "findings"})
    memory = SharedMemory()
    goal = _make_goal()

    llm_markdown = (
        "## Accomplished\nFound 10 relevant papers.\n\n"
        "## Key Decisions\nFocused on 2024 publications.\n\n"
        "## Handoff\nPapers 3 and 7 conflict on methodology."
    )

    mock_response = MagicMock()
    mock_response.content = llm_markdown

    mock_llm = AsyncMock()
    mock_llm.acomplete = AsyncMock(return_value=mock_response)

    with patch("framework.llm.litellm.LiteLLMProvider", return_value=mock_llm):
        md = await generate_node_adapt(
            llm=None,
            node_spec=node_spec,
            result=result,
            memory=memory,
            execution_path=["test"],
            goal=goal,
            visit_number=1,
            previous_adapts=[],
        )

    assert md == llm_markdown


@pytest.mark.asyncio
async def test_generate_node_adapt_llm_failure_fallback():
    """When LLM raises, should fall back to deterministic markdown."""
    node_spec = _make_node_spec()
    result = NodeResult(success=True, output={"result": "done"})
    memory = SharedMemory()
    goal = _make_goal()

    mock_llm = AsyncMock()
    mock_llm.acomplete = AsyncMock(side_effect=RuntimeError("API error"))

    with patch("framework.llm.litellm.LiteLLMProvider", return_value=mock_llm):
        md = await generate_node_adapt(
            llm=None,
            node_spec=node_spec,
            result=result,
            memory=memory,
            execution_path=["test"],
            goal=goal,
            visit_number=1,
            previous_adapts=[],
        )

    # Should get deterministic fallback, not raise
    assert "## Accomplished" in md
    assert "No LLM reflection available" in md


# ---------------------------------------------------------------------------
# Integration with prompt_composer
# ---------------------------------------------------------------------------


class TestPromptComposerIntegration:
    def test_build_narrative_with_adapt_string(self):
        from framework.graph.prompt_composer import build_narrative

        mock_memory = SharedMemory()
        mock_graph = MagicMock()

        adapt_text = "## Accomplished\nDid research.\n\n## Key Decisions\nUsed method A."
        narrative = build_narrative(
            mock_memory,
            ["research"],
            mock_graph,
            execution_narrative=adapt_text,
        )
        assert narrative == adapt_text

    def test_build_narrative_without_adapt_falls_back(self):
        """Backward compat: no execution_narrative falls back to deterministic."""
        from framework.graph.prompt_composer import build_narrative

        mock_memory = SharedMemory()
        mock_memory.write("key1", "value1")
        mock_graph = MagicMock()
        mock_node = MagicMock()
        mock_node.name = "TestNode"
        mock_node.description = "A test"
        mock_graph.get_node.return_value = mock_node

        narrative = build_narrative(mock_memory, ["node1"], mock_graph)
        assert "Phases completed:" in narrative
        assert "key1: value1" in narrative

    def test_build_transition_marker_with_adapt(self):
        from framework.graph.prompt_composer import build_transition_marker

        prev_node = MagicMock()
        prev_node.name = "Research"
        prev_node.description = "Gather data"
        next_node = MagicMock()
        next_node.name = "Analysis"
        next_node.description = "Analyze data"
        mock_memory = SharedMemory()

        adapt_text = (
            "## Accomplished\nFound 10 papers.\n\n## Key Decisions\nFocused on recent work."
        )

        marker = build_transition_marker(
            prev_node,
            next_node,
            mock_memory,
            ["web_search"],
            latest_adapt=adapt_text,
        )
        assert "Phase reflection:" in marker
        assert "Found 10 papers." in marker
        assert "Focused on recent work." in marker
        # Should NOT have the generic prompt
        assert "briefly reflect" not in marker

    def test_build_transition_marker_without_adapt(self):
        """Backward compat: no adapt falls back to generic prompt."""
        from framework.graph.prompt_composer import build_transition_marker

        prev_node = MagicMock()
        prev_node.name = "Research"
        prev_node.description = "Gather data"
        next_node = MagicMock()
        next_node.name = "Analysis"
        next_node.description = "Analyze data"
        mock_memory = SharedMemory()

        marker = build_transition_marker(
            prev_node,
            next_node,
            mock_memory,
            ["web_search"],
        )
        assert "briefly reflect" in marker
        assert "Phase reflection:" not in marker
