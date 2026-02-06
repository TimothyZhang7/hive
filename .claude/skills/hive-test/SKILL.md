---
name: hive-test
description: Run goal-based evaluation tests for agents. Use when you need to verify an agent meets its goals, debug failing tests, or iterate on agent improvements based on test results.
---

# Testing Goal-Driven Agents

This skill helps you test agents built with hive-create. Tests are Python files that run with pytest.

## When to Use This Skill

- **After building an agent** - Verify it meets success criteria
- **Debugging failures** - Understand why tests fail and fix them
- **Iterating on improvements** - Make changes and re-test quickly

## Core Workflow

```
1. list_tests           → Check what tests exist
2. generate_*_tests     → Get test guidelines from MCP tools
3. Write tests          → Use Write tool with provided guidelines
4. run_tests            → Execute tests with pytest
5. debug_test           → Debug failures with verbose output
6. Iterate              → Fix issues and re-run
```

## Key Principle

**MCP tools provide guidelines, you write tests directly:**

- ✅ `generate_constraint_tests`, `generate_success_tests` → Returns templates and guidelines
- ✅ Write tests using the Write tool with provided `file_header` and `test_template`
- ✅ `run_tests` runs pytest and returns structured results
- ✅ `debug_test` re-runs single test with verbose output

**DO NOT** try to generate tests via LLM - the MCP tools give you everything you need.

## Agent Testing Architecture

```
exports/my_agent/
├── __init__.py
├── agent.py              ← Agent to test
├── nodes/__init__.py
├── config.py
├── __main__.py
└── tests/                ← Test files you write
    ├── conftest.py       # Shared fixtures (auto-generated)
    ├── test_constraints.py
    └── test_success_criteria.py
```

## Step-by-Step: Testing an Agent

### Step 1: Check Existing Tests

Always check first before generating new tests:

```python
mcp__agent-builder__list_tests(
    goal_id="your-goal-id",
    agent_path="exports/your_agent"
)
```

### Step 2: Get Test Guidelines

**For constraint tests** (during goal definition):

```python
result = mcp__agent-builder__generate_constraint_tests(
    goal_id="your-goal-id",
    goal_json='{"id": "...", "name": "...", "constraints": [...]}',
    agent_path="exports/your_agent"
)
```

**For success criteria tests** (after agent is built):

```python
result = mcp__agent-builder__generate_success_tests(
    goal_id="your-goal-id",
    goal_json='{"id": "...", "name": "...", "success_criteria": [...]}',
    node_names="node1,node2,node3",
    tool_names="tool1,tool2",
    agent_path="exports/your_agent"
)
```

**Response includes:**
- `output_file`: Where to write tests (e.g., `exports/your_agent/tests/test_constraints.py`)
- `file_header`: Imports, fixtures, and pytest setup
- `test_template`: Format for test functions
- `constraints_formatted` or `success_criteria_formatted`: What to test
- `test_guidelines`: Rules and best practices

### Step 3: Write Tests

Use the Write tool with the provided guidelines:

```python
Write(
    file_path=result["output_file"],
    content=result["file_header"] + "\n\n" + your_test_code
)
```

**Test structure:**
```python
@pytest.mark.asyncio
async def test_constraint_something(mock_mode):
    """Test: Description of what this tests"""
    result = await default_agent.run({"key": "value"}, mock_mode=mock_mode)

    assert result.success, f"Agent failed: {result.error}"

    # Safe output access
    output = result.output or {}
    value = output.get("key", "default")
    assert value == "expected"
```

### Step 4: Create conftest.py

Create shared fixtures using the conftest template:

```python
Write(
    file_path="exports/your_agent/tests/conftest.py",
    content=conftest_template  # From PYTEST_CONFTEST_TEMPLATE
)
```

### Step 5: Run Tests

Use the MCP tool (not pytest directly):

```python
mcp__agent-builder__run_tests(
    goal_id="your-goal-id",
    agent_path="exports/your_agent"
)
```

**Response includes:**
```json
{
  "overall_passed": false,
  "summary": {
    "total": 12,
    "passed": 10,
    "failed": 2,
    "pass_rate": "83.3%"
  },
  "test_results": [...],
  "failures": [...]
}
```

**Options:**
```python
# Run only constraint tests
run_tests(..., test_types='["constraint"]')

# Run with parallel workers
run_tests(..., parallel=4)

# Stop on first failure
run_tests(..., fail_fast=True)
```

### Step 6: Debug Failures

Use the MCP tool for detailed output:

```python
mcp__agent-builder__debug_test(
    goal_id="your-goal-id",
    test_name="test_constraint_something",
    agent_path="exports/your_agent"
)
```

**Response includes:**
- Full verbose output
- Stack trace with line numbers
- Captured logs
- Suggestions for fixing

### Step 7: Categorize and Fix

**IMPLEMENTATION_ERROR** (TypeError, AttributeError, KeyError):
- **Fix:** Edit agent code directly, re-run tests
- **Files:** `agent.py`, `nodes/__init__.py`
- **Fast iteration:** No rebuild needed

**LOGIC_ERROR** (Assertion failures, wrong behavior):
- **Fix:** Update goal definition or agent logic
- **May require:** Going back to hive-create if goal changed significantly

**EDGE_CASE** (Timeout, empty results, boundary conditions):
- **Fix:** Add edge case test and handle it in agent
- **Files:** `tests/test_edge_cases.py`, `agent.py`

## Critical: Safe Output Access

**The framework may return output as strings or dicts. Always use safe access patterns:**

❌ **UNSAFE** (will crash):
```python
approval = result.output["approval_decision"]  # KeyError if missing
category = result.output["analysis"]["category"]  # Fails if nested missing
```

✅ **SAFE** (correct):
```python
output = result.output or {}
approval = output.get("approval_decision", "UNKNOWN")

# Safe nested access
analysis = output.get("analysis", {})
if isinstance(analysis, dict):
    category = analysis.get("category", "unknown")

# Parse JSON if stored as string
import json
recommendation = output.get("recommendation", "{}")
if isinstance(recommendation, str):
    try:
        parsed = json.loads(recommendation)
        if isinstance(parsed, dict):
            approval = parsed.get("approval_decision", "UNKNOWN")
    except json.JSONDecodeError:
        approval = "UNKNOWN"
```

## ExecutionResult Fields

**IMPORTANT:** `result.success=True` means **NO exception occurred**, NOT that the goal was achieved!

```python
# ❌ WRONG - only checks execution didn't crash
assert result.success

# ✅ RIGHT - checks execution AND goal achievement
assert result.success, f"Agent failed: {result.error}"
output = result.output or {}
approval = output.get("approval_decision")
assert approval == "APPROVED", f"Expected APPROVED, got {approval}"
```

**All ExecutionResult fields:**
- `success: bool` - Execution completed without exception
- `output: dict` - Complete memory snapshot
- `error: str | None` - Error message if failed
- `steps_executed: int` - Number of nodes executed
- `total_tokens: int` - Cumulative token usage
- `total_latency_ms: int` - Total execution time
- `path: list[str]` - Node IDs traversed
- `paused_at: str | None` - Node ID if HITL pause occurred
- `node_visit_counts: dict[str, int]` - Times each node executed (for feedback loops)

## Test Guidelines

**Keep tests focused:**
- Generate **8-15 tests total**, not 30+
- 2-3 tests per success criterion
- 1 happy path, 1 boundary case, 1 error handling
- Each test requires real LLM call (~3 seconds, costs money)

**Test structure:**
```python
@pytest.mark.asyncio                    # Always required
async def test_name(mock_mode):         # Always accept mock_mode fixture
    """Test: Clear description"""
    result = await default_agent.run({...}, mock_mode=mock_mode)

    assert result.success, f"Agent failed: {result.error}"

    # Safe access to output
    output = result.output or {}
    # Add specific assertions
```

## Common Test Patterns

**Happy path:**
```python
@pytest.mark.asyncio
async def test_happy_path(mock_mode):
    result = await default_agent.run({"query": "test"}, mock_mode=mock_mode)
    assert result.success
    assert len(result.output) > 0
```

**Boundary condition:**
```python
@pytest.mark.asyncio
async def test_boundary_minimum(mock_mode):
    result = await default_agent.run({"query": "edge case"}, mock_mode=mock_mode)
    assert result.success
    output = result.output or {}
    assert len(output.get("results", [])) >= 1
```

**Error handling:**
```python
@pytest.mark.asyncio
async def test_error_handling(mock_mode):
    result = await default_agent.run({"query": ""}, mock_mode=mock_mode)
    # Should fail gracefully
    assert not result.success or result.output.get("error") is not None
```

**Feedback loops:**
```python
@pytest.mark.asyncio
async def test_feedback_loop_terminates(mock_mode):
    result = await default_agent.run({"input": "test"}, mock_mode=mock_mode)
    visits = getattr(result, "node_visit_counts", {}) or {}
    for node_id, count in visits.items():
        assert count <= 5, f"Node {node_id} visited {count} times (max 5)"
```

## Credential Requirements

**Tests require ALL credentials the agent depends on:**

1. **LLM API key** (always required):
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   ```

2. **Tool-specific credentials** (depends on agent):
   - Check `mcp_servers.json` for what tools the agent uses
   - Use `mcp__agent-builder__check_missing_credentials` to detect missing creds
   - Use hive-credentials skill to collect and store them

**Mock mode** (`MOCK_MODE=1 pytest ...`) only validates structure, NOT behavior:
- ✓ Tests code doesn't crash
- ✗ Does NOT test LLM reasoning
- ✗ Does NOT test constraint validation
- ✗ Does NOT test real API integrations

## Integration with Other Skills

| Scenario | Action |
|----------|--------|
| Agent built, ready to test | Use hive-test to generate success tests |
| LOGIC_ERROR found | Switch to hive-create to update goal |
| IMPLEMENTATION_ERROR | Fix directly, re-run tests |
| EDGE_CASE found | Add test, fix handling |
| All tests pass | Agent validated ✅ |

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| ❌ Write tests without guidelines | ✅ Use `generate_*_tests` first |
| ❌ Run pytest via Bash | ✅ Use `run_tests` MCP tool |
| ❌ Debug with `pytest -vvs` | ✅ Use `debug_test` MCP tool |
| ❌ Use `result.output["key"]` | ✅ Use `result.output.get("key")` |
| ❌ Assume `result.success` means goal achieved | ✅ Check success AND output values |
| ❌ Generate 30+ tests | ✅ Keep it focused: 8-15 tests |

## MCP Tools Reference

```python
# List existing tests
mcp__agent-builder__list_tests(goal_id, agent_path)

# Get constraint test guidelines
mcp__agent-builder__generate_constraint_tests(goal_id, goal_json, agent_path)

# Get success criteria test guidelines
mcp__agent-builder__generate_success_tests(goal_id, goal_json, node_names, tool_names, agent_path)

# Run tests
mcp__agent-builder__run_tests(goal_id, agent_path, test_types='["all"]', parallel=-1, fail_fast=False)

# Debug specific test
mcp__agent-builder__debug_test(goal_id, test_name, agent_path)

# Check missing credentials
mcp__agent-builder__check_missing_credentials(agent_path)
```

## Direct pytest Commands

You can also run tests directly:

```bash
# Run all tests
pytest exports/your_agent/tests/ -v

# Run specific test file
pytest exports/your_agent/tests/test_constraints.py -v

# Run specific test
pytest exports/your_agent/tests/test_constraints.py::test_something -vvs

# Run in mock mode (structure validation only)
MOCK_MODE=1 pytest exports/your_agent/tests/ -v
```

---

**Remember:** The goal is fast iteration. Edit agent code directly for bugs, re-run tests immediately. Only go back to hive-create if the goal definition needs to change.
