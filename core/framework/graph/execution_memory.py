"""ADAPT.md — execution memory as plain markdown files.

After each node completes, an LLM-generated (or deterministic) reflection
is written to {storage_path}/conversations/{node_id}/ADAPT.md. These files
are read in execution-path order to build a running narrative for transition
markers, judges, and system prompts.

No dataclasses. No JSON serialization. No session_state persistence.
The files are on disk — they survive crashes natively.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from framework.graph.goal import Goal
    from framework.graph.node import NodeResult, NodeSpec, SharedMemory
    from framework.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


def write_adapt(storage_path: Path, node_id: str, content: str) -> None:
    """Write ADAPT.md for a node. Overwrites on revisit (latest wins)."""
    adapt_path = storage_path / "conversations" / node_id / "ADAPT.md"
    adapt_path.parent.mkdir(parents=True, exist_ok=True)
    adapt_path.write_text(content, encoding="utf-8")


def read_single_adapt(storage_path: Path, node_id: str) -> str | None:
    """Read one node's ADAPT.md, or None if it doesn't exist."""
    adapt_path = storage_path / "conversations" / node_id / "ADAPT.md"
    if adapt_path.exists():
        return adapt_path.read_text(encoding="utf-8")
    return None


def read_adapt_narrative(
    storage_path: Path,
    execution_path: list[str],
    max_entries: int = 10,
) -> str:
    """Read ADAPT.md files in execution order and concatenate into a narrative.

    Each node appears at most once, at its last position in the path.
    Returns the most recent ``max_entries`` nodes' reflections.
    """
    if not execution_path:
        return ""

    # Deduplicate: keep last occurrence of each node_id
    seen: set[str] = set()
    unique_ordered: list[str] = []
    for node_id in reversed(execution_path):
        if node_id not in seen:
            seen.add(node_id)
            unique_ordered.append(node_id)
    unique_ordered.reverse()

    # Take the most recent max_entries
    recent = unique_ordered[-max_entries:]

    parts: list[str] = []
    for node_id in recent:
        content = read_single_adapt(storage_path, node_id)
        if content:
            parts.append(content.strip())

    return "\n\n---\n\n".join(parts)


def _format_outputs_summary(output: dict[str, Any], max_chars: int = 400) -> str:
    """Format node outputs for the reflection prompt, truncating long values."""
    if not output:
        return "(no outputs)"
    parts: list[str] = []
    total = 0
    for key, value in output.items():
        if value is None:
            continue
        val_str = str(value)
        if len(val_str) > 150:
            val_str = val_str[:150] + "..."
        line = f"- {key}: {val_str}"
        if total + len(line) > max_chars:
            parts.append(f"- ... and {len(output) - len(parts)} more keys")
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts) if parts else "(no outputs)"


def _build_deterministic_adapt(
    node_spec: NodeSpec,
    result: NodeResult,
    visit_number: int,
) -> str:
    """Build a deterministic ADAPT.md when LLM is unavailable."""
    output_keys = list(result.output.keys()) if result.output else []
    quality = "succeeded" if result.success else "failed"

    return f"""## Accomplished
Completed {node_spec.name} (visit {visit_number}): {node_spec.description}. Phase {quality}.

## Key Decisions
(No LLM reflection available.) Outputs: {", ".join(output_keys) or "none"}.

## Handoff
Outputs available for downstream processing."""


async def generate_node_adapt(
    llm: LLMProvider | None,
    node_spec: NodeSpec,
    result: NodeResult,
    memory: SharedMemory,
    execution_path: list[str],
    goal: Goal,
    visit_number: int,
    previous_adapts: list[tuple[str, str]],
) -> str:
    """Generate an ADAPT.md reflection after node completion.

    Returns plain markdown. Falls back to deterministic summary on failure.

    Args:
        llm: LLM provider (unused; function creates its own Haiku instance).
        node_spec: Spec of the completed node.
        result: The node's execution result.
        memory: Current shared memory state.
        execution_path: Node IDs visited so far.
        goal: The execution goal.
        visit_number: How many times this node has been visited.
        previous_adapts: List of (node_name, adapt_content) from prior phases.
    """
    outputs_formatted = _format_outputs_summary(result.output)
    path_str = " -> ".join(execution_path) if execution_path else "(first node)"

    prev_context = ""
    if previous_adapts:
        recent = previous_adapts[-2:]
        prev_lines = [f"- {name}: (see ADAPT.md)" for name, _ in recent]
        prev_context = "\nPrevious phases:\n" + "\n".join(prev_lines)

    success_criteria = ""
    if node_spec.success_criteria:
        success_criteria = f"\nSuccess criteria: {node_spec.success_criteria}"

    prompt = f"""Summarize this completed phase of an agent workflow as markdown.

Goal: {goal.name} -- {goal.description}
Phase: {node_spec.name} -- {node_spec.description}{success_criteria}
Visit: {visit_number}
Path: {path_str}

Outputs:
{outputs_formatted}
{prev_context}

Write EXACTLY three markdown sections. Be concise (1-2 sentences each):

## Accomplished
(what this phase achieved)

## Key Decisions
(notable decisions, findings, or trade-offs)

## Handoff
(what the next phase should know or watch for)"""

    try:
        from framework.config import get_light_llm_config
        from framework.llm.litellm import LiteLLMProvider

        light_cfg = get_light_llm_config()
        if not light_cfg["model"]:
            raise RuntimeError("No light_llm or llm configured in ~/.hive/configuration.json")

        reflection_llm = LiteLLMProvider(
            model=light_cfg["model"],
            api_key=light_cfg["api_key"],
        )
        response = await reflection_llm.acomplete(
            messages=[{"role": "user", "content": prompt}],
            system="You are a concise workflow analyst. Respond with only markdown.",
            max_tokens=light_cfg["max_tokens"],
            max_retries=1,
        )
        if response.content and response.content.strip():
            return response.content.strip()
    except Exception as e:
        logger.debug("LLM reflection failed, using deterministic fallback: %s", e)

    return _build_deterministic_adapt(node_spec, result, visit_number)
