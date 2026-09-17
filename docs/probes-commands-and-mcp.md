# Design: state probes, recovery commands, and MCP

**Status: design (not yet implemented).** This documents how the agent will run
operator-declared commands / Python functions to (a) read ground truth such as
**CCF (Car Configuration File)** parameters and VHAL signals, (b) recover when
stuck, and (c) be driven from an IDE over MCP. Implementation follows the phased
plan in §7.

## Why

Today the agent verifies against **what the screen shows**. That proves the HMI,
not the vehicle. The most valuable automotive check is a **divergence**:

```
CCF says seat-massage = ENABLED
        │
        ├── UI exposes it AND VHAL SEAT_MASSAGE_STATE flips ON   → PASS
        ├── UI never exposes it                                  → coverage DEFECT
        └── UI says "running" but VHAL stays OFF                 → actuator DEFECT
```

To do that the agent needs to read CCF/VHAL/logs — i.e. run commands. Those
commands must be **operator-controlled and allowlisted**, never invented by the
model or carried in a shareable manual.

## Security model (read this first)

1. **Commands live in `config.json`, never in `manual.json`.** The manual is
   knowledge, is embedded in a shareable PDF, and may come from an OEM. It must
   never carry anything executable.
2. **Allowlist only.** The agent/MCP invokes a probe **by name** from the set the
   operator declared. The model may *request* `ccf_seat_massage`; it can never
   compose a shell string. No command text ever comes from model output or the
   manual.
3. **Read-only by default.** `probes` are reads. Anything that mutates
   (`recovery`, `execute`) is off unless the operator sets an explicit enable
   flag (`allow_recovery`, `allow_execute`).
4. **Bounded + audited.** Every probe/command has a timeout; every invocation is
   a trace event (`probe`, `recovery`, kind + name + duration + exit) in
   `events.jsonl`, so a run is fully auditable.
5. **Python hooks are dotted paths the operator installed** (e.g.
   `myteam.ivi_probes:read_ccf`) — imported from the operator's environment, not
   downloaded or eval'd.

## 1. Config schema

```jsonc
{
  // Ground-truth reads. Named, allowlisted, read-only.
  "probes": {
    "ccf_seat_massage": {
      "type": "adb_shell",
      "command": "getprop persist.vendor.ccf.seat_massage",
      "parse": "regex:(\\S+)",          // optional: capture group 1 is the value
      "timeout": 5,
      "description": "CCF flag: seat-massage feature coded present"
    },
    "vhal_seat_massage_state": {
      "type": "adb_shell",
      "command": "dumpsys car_service --property SEAT_MASSAGE_STATE",
      "parse": "regex:value:\\s*(\\d+)"
    },
    "ccf_dump": {
      "type": "python",
      "callable": "myteam.ivi_probes:read_ccf",  // fn(device, args) -> str | dict
      "description": "Full CCF parameter dump as JSON"
    }
  },

  // Assertions tie a probe to an expected value for a goal -> UI-vs-state check.
  "assertions": [
    {
      "when_goal_contains": "seat massage",
      "probe": "vhal_seat_massage_state",
      "expect": "1",
      "means": "seat massage actuator running"
    }
  ],

  // CCF feature coverage: each coded-present feature should be reachable in the UI.
  "ccf_coverage": {
    "probe": "ccf_dump",                 // returns a map of feature -> "1"/"0"
    "feature_screens": {                 // feature key -> scene-graph screen id
      "seat_massage": "screen.seat_massage",
      "ambient_light": "screen.ambient"
    }
  },

  // Recovery hooks the agent may run when stuck. OFF unless allow_recovery=true.
  "allow_recovery": false,
  "recovery": {
    "go_home":      {"type": "adb_shell", "command": "input keyevent KEYCODE_HOME"},
    "relaunch_app": {"type": "adb_shell", "command": "am start -n com.example.iviwv/.MainActivity"},
    "dump_logcat":  {"type": "adb_shell", "command": "logcat -d -t 200"}
  },

  // Direct action execution over MCP. OFF unless allow_execute=true.
  "allow_execute": false
}
```

`type` is `adb_shell` (run via the session's `adb -s <serial> shell …`) or
`python` (import `module:function`, call with the `AdbDevice` + parsed args).
`parse` is optional (`regex:…` capture, or `json:<path>`); without it the raw
stdout is returned.

## 2. State probes + divergence check (the moat)

At verification time, for the active goal:

1. Run any `assertions` whose `when_goal_contains` matches the goal.
2. Compare the probe value to `expect`.
3. Combine with the existing **visual** verdict:

| Visual verdict | Probe/assertion | Result |
| --- | --- | --- |
| pass | matches | **PASS** (UI + state agree) |
| pass | mismatch | **DEFECT** — display/actuator divergence |
| fail | matches | investigate (UI missed a real success) |
| fail | mismatch | FAIL |

A divergence becomes a scene-graph finding (`state_divergence`) alongside the
existing `undocumented_screen` findings, and lands in `result.json` +
`report.html`. Probe values are emitted as `vehicle_state` trace events (the
event schema was left open for exactly this).

## 3. CCF feature-coverage check

Independently of any single task, `ccf_coverage` answers *"is every coded feature
actually in the HMI?"*:

- Read the CCF map via the named probe.
- For each feature coded **present**, check the scene graph for its
  `feature_screens[...]` node with status `confirmed` (reached in some run).
- **Coded present but screen never reachable → coverage DEFECT.**
- **Coded absent but the screen exists / is reachable → surfaced for review.**

This turns "does the build match its variant coding?" into an automatic report —
a very automotive, very defensible check.

## 4. Recovery when stuck

When an action makes no progress or a subgoal stalls (the loop-detection path),
and `allow_recovery` is true, the agent may run a `recovery` hook before
re-planning — e.g. `go_home` then re-observe, or `relaunch_app`, and always
`dump_logcat` into the run directory for the trace. Each recovery is a trace
event; recovery never runs a command outside the declared set.

## 5. MCP server

Expose the agent to Claude Code / Windsurf / an IDE. Read-only tools are always
available; mutating tools require the enable flags above.

| Tool | Mutates? | Purpose |
| --- | --- | --- |
| `ivi_list_probes` | no | list declared probes/assertions |
| `ivi_run_probe(name)` | no | run one probe, return parsed value |
| `ivi_get_vehicle_state` | no | run all assertion probes, return values |
| `ivi_capture_screen` / `ivi_get_display` | no | screenshot / display info |
| `ivi_get_trace(run_id)` / `ivi_inspect_logs` | no | events.jsonl / logcat dump |
| `ivi_graph_show` / `ivi_graph_review` | graph only | coverage + triage findings |
| `ivi_run_test(goal, serial, display_id)` | device | run a goal end-to-end |
| `ivi_execute_action(action)` | device | one action (needs `allow_execute`) |
| `ivi_run_recovery(name)` | device | one recovery hook (needs `allow_recovery`) |

The model calling these still only names allowlisted probes/recoveries; the MCP
layer enforces the same allowlist and enable flags as the agent.

## 6. What the model is told

The planner/verifier prompt gains a short, read-only list: *"Ground-truth probes
available: ccf_seat_massage, vhal_seat_massage_state. You may request a probe by
name to confirm state; you cannot run arbitrary commands."* So the model can lean
on state when the pixels are ambiguous (exactly the seat-selection case), but has
no path to arbitrary execution.

## 7. Phased implementation plan

1. **Probe runner + schema + trace.** `probes.py`: load config, run `adb_shell` /
   `python` probes with timeout + `parse`, emit `probe` / `vehicle_state` events.
   Config keys `probes`; `ivi-agent probe run <name>` CLI. (Read-only, safe.)
2. **Assertions + divergence.** Wire assertions into verification; add
   `state_divergence` findings to the scene graph, `result.json`, report.
3. **CCF coverage.** `ccf_coverage` report + `ivi-agent ccf-coverage --profile`.
4. **Recovery hooks.** `recovery` + `allow_recovery` in the stuck path.
5. **MCP server.** `mcp_server/` exposing the table in §5, honoring the allowlist
   and enable flags.

Each phase is independently useful and testable with fakes (no real vehicle
needed for unit tests).

## Non-goals / guardrails

- No arbitrary command execution — ever — from the model or the manual.
- No storing secrets in the manual; probe commands that need credentials read
  them from the operator's environment.
- Probes are advisory to *validation*; they never change what the agent taps
  except through declared `recovery` hooks under `allow_recovery`.

See also: [creating-a-manual.md](creating-a-manual.md) (knowledge authoring) and
the **Living scene graph** section of the [README](../README.md).
