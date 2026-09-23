#!/usr/bin/env python3
"""MCP server exposing a pytest suite (media_test.py, climate_test.py, ...).

Drop this file into your pytest framework's repo and run it. It exposes three
tools so an MCP client (Claude Code, or the IVI agent) can discover and run your
existing tests without changing any of them:

  * pytest_list_tests  -- collect test node IDs (optionally filtered), no run
  * pytest_run_test    -- run one test by node ID  -> pass/fail + failure detail
  * pytest_run_suite   -- run a file / dir / -k / -m selection -> summary

Adding a new *_test.py needs NO change here -- pytest discovers it automatically.

Requirements:
    pip install "mcp<2" pytest pytest-json-report

Run:
    python pytest_mcp.py                 # stdio transport (local)
    PYTEST_MCP_ROOT=/path/to/suite python pytest_mcp.py   # if run from elsewhere

Register in an MCP client (e.g. .mcp.json):
    {"mcpServers": {"pytest": {"command": "python", "args": ["pytest_mcp.py"]}}}

Gotchas (baked in on purpose):
  * Pinned to the 1.x MCP SDK -- mcp 2.x renamed FastMCP -> MCPServer.
  * Pydantic input models are module-level and this file does NOT use
    `from __future__ import annotations`: FastMCP evaluates each tool's
    annotations against module globals to build the schema, so local/stringized
    annotations break it.
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pytest_mcp")

# The suite root. Defaults to this file's directory; override with PYTEST_MCP_ROOT.
ROOT = os.environ.get("PYTEST_MCP_ROOT", os.path.dirname(os.path.abspath(__file__)))


def _pytest(args, timeout=None):
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout,
    )


def _run_with_report(select_args, timeout):
    """Run pytest for ``select_args`` and return a structured result dict."""
    fd, report = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        proc = _pytest(
            [*select_args, "-p", "no:cacheprovider",
             "--json-report", f"--json-report-file={report}"],
            timeout=timeout,
        )
        try:
            with open(report) as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {
                "error": "pytest produced no JSON report (collection error?)",
                "returncode": proc.returncode,
                "stderr": (proc.stderr or proc.stdout)[-2000:],
            }
    finally:
        try:
            os.remove(report)
        except OSError:
            pass

    tests = []
    for item in data.get("tests", []):
        call = item.get("call") or {}
        detail = ""
        if item.get("outcome") != "passed":
            detail = str(call.get("longrepr") or item.get("longrepr") or "")[:1500]
        tests.append({
            "id": item.get("nodeid"),
            "outcome": item.get("outcome"),
            "duration": round(call.get("duration", 0.0), 3),
            "detail": detail,
        })
    return {"summary": data.get("summary", {}), "tests": tests}


class ListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(default=".", description="File or dir to collect from, e.g. 'media_test.py'")
    keyword: Optional[str] = Field(default=None, description="pytest -k expression, e.g. 'media and not slow'")


@mcp.tool(
    name="pytest_list_tests",
    annotations={"title": "List pytest tests", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def pytest_list_tests(params: ListInput) -> str:
    """Collect test node IDs from the suite. No tests are executed.

    Returns JSON: {"tests": ["media_test.py::test_play", ...], "count": N}.
    """
    args = ["--collect-only", "-q", "-p", "no:cacheprovider", params.path]
    if params.keyword:
        args += ["-k", params.keyword]
    proc = _pytest(args)
    ids = [line.strip() for line in proc.stdout.splitlines() if "::" in line]
    if not ids and proc.returncode not in (0, 5):  # exit 5 = "no tests collected"
        return json.dumps(
            {"tests": [], "count": 0, "error": (proc.stderr or proc.stdout)[-2000:]},
            indent=2,
        )
    return json.dumps({"tests": ids, "count": len(ids)}, indent=2)


class RunTestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str = Field(..., description="Test node id, e.g. 'media_test.py::test_play'", min_length=1)
    timeout_seconds: Optional[float] = Field(default=None, description="Optional hard timeout", ge=1)


@mcp.tool(
    name="pytest_run_test",
    annotations={"title": "Run one pytest test", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def pytest_run_test(params: RunTestInput) -> str:
    """Run a single test case and return its result + failure detail.

    Returns JSON: {"summary": {...}, "tests": [{"id","outcome","duration","detail"}]}.
    Use pytest_list_tests first to get valid node IDs.
    """
    try:
        return json.dumps(
            _run_with_report([params.node_id], params.timeout_seconds), indent=2, default=str
        )
    except subprocess.TimeoutExpired:
        return f"Error: test timed out after {params.timeout_seconds}s"


class RunSuiteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(default=".", description="File or dir to run, e.g. 'media_test.py' or '.'")
    keyword: Optional[str] = Field(default=None, description="pytest -k expression")
    markers: Optional[str] = Field(default=None, description="pytest -m marker expression, e.g. 'smoke'")
    timeout_seconds: Optional[float] = Field(default=None, ge=1)


@mcp.tool(
    name="pytest_run_suite",
    annotations={"title": "Run a pytest suite", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def pytest_run_suite(params: RunSuiteInput) -> str:
    """Run a file / directory / -k / -m selection and return per-test outcomes.

    Returns JSON: {"summary": {passed, failed, total, ...}, "tests": [...]}.
    """
    select = [params.path]
    if params.keyword:
        select += ["-k", params.keyword]
    if params.markers:
        select += ["-m", params.markers]
    try:
        return json.dumps(
            _run_with_report(select, params.timeout_seconds), indent=2, default=str
        )
    except subprocess.TimeoutExpired:
        return f"Error: suite timed out after {params.timeout_seconds}s"


if __name__ == "__main__":
    mcp.run()
