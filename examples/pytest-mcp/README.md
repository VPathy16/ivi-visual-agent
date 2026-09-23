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

## Register in a client

MCP is an open protocol, so **any** MCP-capable client can run this server and
hand its tools to whatever model that client is driving (Anthropic, OpenAI,
OpenRouter, a local Ollama model, ...). Set `PYTEST_MCP_ROOT` if the client
launches the server from a different directory than your suite.

**Claude Code / Cursor / Cline** (`.mcp.json`, or `.cursor/mcp.json`) — `command` + `args`:

```json
{ "mcpServers": { "pytest": { "command": "python", "args": ["pytest_mcp.py"],
  "env": { "PYTEST_MCP_ROOT": "/absolute/path/to/your/pytest/suite" } } } }
```

**opencode** (`opencode.json`, or `~/.config/opencode/opencode.json`) — note the
single `command` **array** and `type: "local"`; see [`opencode.json`](./opencode.json):

```json
{ "$schema": "https://opencode.ai/config.json",
  "mcp": { "pytest": { "type": "local", "command": ["python", "pytest_mcp.py"],
    "enabled": true,
    "environment": { "PYTEST_MCP_ROOT": "/absolute/path/to/your/pytest/suite" } } } }
```

Confirm the three tools registered with `/mcp` in opencode's TUI. If you use a
venv, point `command` at that venv's python (e.g.
`["/path/.venv/bin/python", "pytest_mcp.py"]`) — clients inherit the shell env
but don't auto-activate a venv.

**Not an MCP client?** `pytest_mcp.py` is plain Python: `import` it and call
`_run_with_report(...)` directly, or wrap the same three functions in a small
FastAPI/OpenAPI shim for any function-calling model.

> Tool-calling reliability is the model's job, not MCP's. Strong hosted models
> drive these three tools cleanly; a small local model may need a tool-tuned
> variant (`qwen2.5-coder`, `llama3.1`) to call them consistently. The server is
> identical either way.

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
