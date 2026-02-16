"""Cost reporter: generates cost.json from L2 runtime logs.

Reads details.jsonl (per-node L2 logs) and computes cost per-node
and total session cost using litellm's pricing database.

The report is written to ``{session_dir}/cost.json`` alongside state.json.
Generation is always best-effort and non-fatal — missing pricing data
or unrecognised models produce ``cost_usd: 0``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost for a single node using litellm pricing."""
    if not model or (input_tokens == 0 and output_tokens == 0):
        return 0.0
    try:
        import litellm

        return litellm.completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    except Exception:
        logger.debug("Could not compute cost for model=%s", model, exc_info=True)
        return 0.0


def generate_cost_report(session_dir: Path, run_id: str = "") -> dict[str, Any]:
    """Generate cost report from L2 logs on disk.

    Args:
        session_dir: Path to session directory
                     (e.g. ``~/.hive/agents/{name}/sessions/{session_id}/``)
        run_id: The run/session ID.  If empty, uses the session dir name.

    Returns:
        Cost report dict.  Also written to ``session_dir/cost.json``.
    """
    from framework.runtime.runtime_log_schemas import NodeDetail
    from framework.runtime.runtime_log_store import RuntimeLogStore

    if not run_id:
        run_id = session_dir.name

    logs_dir = session_dir / "logs"
    details_path = logs_dir / "details.jsonl"

    if not details_path.exists():
        logger.debug("No details.jsonl at %s — skipping cost report", details_path)
        return {}

    # RuntimeLogStore expects a base_path two levels above logs/
    # For new-format sessions: base_path/sessions/{session_id}/logs/details.jsonl
    # _get_run_dir resolves session_ prefixed IDs to base_path/sessions/{id}/logs/
    base_path = session_dir.parent.parent
    store = RuntimeLogStore(base_path)
    node_details: list[NodeDetail] = store.read_node_details_sync(run_id)

    nodes: list[dict[str, Any]] = []
    total_cost = 0.0
    total_input = 0
    total_output = 0

    for nd in node_details:
        cost = _compute_cost(nd.model, nd.input_tokens, nd.output_tokens)
        total_cost += cost
        total_input += nd.input_tokens
        total_output += nd.output_tokens
        nodes.append(
            {
                "node_id": nd.node_id,
                "node_name": nd.node_name,
                "node_type": nd.node_type,
                "model": nd.model,
                "input_tokens": nd.input_tokens,
                "output_tokens": nd.output_tokens,
                "total_tokens": nd.tokens_used or (nd.input_tokens + nd.output_tokens),
                "cost_usd": round(cost, 6),
            }
        )

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "session_id": run_id,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_cost_usd": round(total_cost, 6),
        "currency": "USD",
        "nodes": nodes,
    }

    # Write to disk alongside state.json
    cost_path = session_dir / "cost.json"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    cost_path.write_text(json.dumps(report, indent=2) + "\n")

    logger.info("Cost report written: %s (total=$%.4f)", cost_path, total_cost)
    return report
