# MCP server

Expose the IVI validation agent to MCP clients (Claude Code, Antigravity,
Cursor, …) so you can run and inspect Android-HMI validations from inside your
assistant. It is a **local, stdio** server: it reuses the agent in-process and
talks to the device over ADB, so it must run on the same machine as the emulator
or head unit.

## Install

```bash
pip install -e '.[mcp]'          # adds the pinned 1.x MCP SDK
```

This installs the `ivi-agent-mcp` command (equivalently `python -m
ivi_agent.mcp_server`).

## Register it with a client

Claude Code — add to `.mcp.json` (or `claude mcp add`):

```json
{
  "mcpServers": {
    "ivi-agent": {
      "command": "ivi-agent-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

The same `command`/`args` shape works for Antigravity, Cursor and Windsurf. Run
`ivi-agent doctor` first to confirm `adb`, the device, and the model are ready.

## Tools

| Tool | Read-only | What it does |
|------|-----------|--------------|
| `ivi_run_validation` | no | Run a plain-language goal against the connected IVI. Writes full evidence (screenshots, `result.json`, `report.html`, `replay.html`, `events.jsonl`) and returns a compact summary: outcome, subgoals, grounding, scene-graph coverage, crashes. |
| `ivi_device_state` | yes | Snapshot the live screen without touching it: title(s), visible text, clickable elements + centers, screenshot path. |
| `ivi_inspect_trace` | yes | Summarize a finished run directory: outcome, per-step actions + grounding, event-kind counts, crashes, replay path. |
| `ivi_scene_graph` | yes | Manual-vs-HMI scene-graph coverage and pending divergences (candidate defects) for a knowledge profile. |
| `ivi_list_runs` | yes | List recent runs (newest first, paginated) with outcomes. |
| `ivi_doctor` | yes | Preflight local dependencies (adb, scrcpy, ollama, tesseract) and model availability. |

All inputs mirror the CLI (`config_path`, `knowledge_profile`, `serial`,
`display_id`, the `exec_profile` = `fast|balanced|strict`, and
`verification_level` = `off|final|checkpoints|strict`). Tools return JSON
strings.

## Notes

- Pinned to the 1.x MCP SDK: version 2.x renamed `FastMCP` to `MCPServer` and
  changed the API. Migrating to 2.x is tracked separately.
- Only `ivi_run_validation` drives the device; every other tool is read-only.
