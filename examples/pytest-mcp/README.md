# pytest-as-MCP: expose an existing pytest suite as a validation oracle

`pytest_mcp.py` wraps an existing **pytest** suite (e.g. `media_test.py`,
`climate_test.py` — signal-injection + snapshot-vs-reference HIL tests) as an MCP
server, with **no changes to the tests**. It's the deterministic "oracle" half of
the integration described in [docs/windows-backend.md](../../docs/windows-backend.md)
and the README: the IVI agent (or Claude Code) navigates the HMI to a precondition
state and then calls these tools to inject a vehicle signal and assert the result.

Copy `pytest_mcp.py` into the root of your pytest repo (it discovers tests
relative to its own directory, or `PYTEST_MCP_ROOT`).

## Install & run

```bash
pip install "mcp<2" pytest pytest-json-report
python pytest_mcp.py                 # stdio
# or, if launched from elsewhere:
PYTEST_MCP_ROOT=/path/to/suite python pytest_mcp.py
```

Register in an MCP client (`.mcp.json`):

```json
{ "mcpServers": { "pytest": { "command": "python", "args": ["pytest_mcp.py"] } } }
```

## Tools

| Tool | Purpose |
|------|---------|
| `pytest_list_tests` | Collect node IDs (`media_test.py::test_play`), optional `-k` filter. Read-only, runs nothing. |
| `pytest_run_test` | Run one node ID → `{outcome, duration, detail}` (failure traceback truncated). |
| `pytest_run_suite` | Run a file / dir / `-k` / `-m` selection → summary + per-test outcomes. |

Results are structured via `pytest-json-report` (not scraped from stdout). Adding
a new `*_test.py` needs no change here — pytest discovers it automatically.

## Why the pins / style (baked in)

- Pinned to the **1.x MCP SDK** — `mcp` 2.x renamed `FastMCP` → `MCPServer`.
- Pydantic input models are **module-level** and the file avoids
  `from __future__ import annotations`, because FastMCP builds each tool's schema
  by evaluating its annotations against module globals.
- pytest exit code **5** ("no tests collected") is treated as empty, not an error.

## The division of labor

```
IVI agent (robust navigation + defect discovery)
   └─ reaches the precondition state
        └─ pytest_run_test → your suite injects the signal + snapshot-compares → pass/fail
```

The agent handles getting there and finding undocumented screens; your pytest
suite remains the pixel-exact, signal-vs-HMI ground truth.
