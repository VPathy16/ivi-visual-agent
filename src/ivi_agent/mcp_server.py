#!/usr/bin/env python3
"""MCP server exposing the IVI validation agent to MCP clients.

Lets an MCP client (Claude Code, Antigravity, Cursor, ...) drive Android-HMI /
IVI validations against a locally connected device or emulator: run a plain
-language goal, inspect the live screen, read a past run's evidence, check
manual-vs-HMI scene-graph coverage, list runs, and preflight the local setup.

This is a *local* server: it reuses the agent in-process and talks to the device
over ADB, so it runs on stdio. The agent logic lives in plain helper functions
(``run_validation``, ``device_state``, ...) that return dicts and need no MCP
dependency — they are unit-testable and reused by the thin FastMCP tool wrappers
in ``build_server()``. Install the server with the ``[mcp]`` extra.

NOTE: this module intentionally does NOT use ``from __future__ import
annotations``. FastMCP resolves each tool's parameter annotation by evaluating
it against module globals; the Pydantic input models are defined locally in
``build_server()``, so they must remain real class objects (not stringized
annotations) for FastMCP to find them.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from .adb import AdbDevice
from .config import Config, PROFILES, VERIFICATION_LEVELS
from .knowledge import KnowledgeBase
from .embeddings import resolve_text_embedder
from .model import OllamaVisionModel
from .agent import GoalAgent
from .graph import SceneGraph
from .perception import (
    extract_screen_titles,
    extract_tree_headings,
    extract_ui_elements,
    extract_visible_text,
)
from .replay import _load_events, _load_json, build_replay

# ---------------------------------------------------------------------------
# Builders (shared construction, so no tool duplicates wiring)
# ---------------------------------------------------------------------------


def _build_config(config_path: Optional[str], exec_profile: Optional[str]) -> Config:
    return Config.load(config_path).apply_profile(exec_profile)


def _build_model(config: Config) -> OllamaVisionModel:
    return OllamaVisionModel(
        config.ollama_url,
        config.model,
        timeout=config.model_timeout_seconds,
        prefer_ui_tree=config.prefer_ui_tree,
        enable_ocr=config.enable_ocr,
        max_image_dimension=config.max_image_dimension,
        grounding_mode=config.grounding_mode,
        lenient=config.lenient_planning,
        num_ctx=config.model_context_tokens,
    )


def _build_knowledge(
    config: Config, knowledge_profile: Optional[str], knowledge_root: Optional[str]
) -> Optional[KnowledgeBase]:
    profile = knowledge_profile or config.knowledge_profile
    if not profile:
        return None
    root = Path(knowledge_root or config.knowledge_root)
    embedder = resolve_text_embedder(
        config.ollama_url,
        config.embedding_model,
        config.use_embeddings,
        config.embedding_backend,
    )
    return KnowledgeBase.open(root, profile, embedder=embedder)


# ---------------------------------------------------------------------------
# Helpers — pure, dict-returning, MCP-agnostic, unit-testable
# ---------------------------------------------------------------------------


def _summarize_result(result_dict: dict[str, Any]) -> dict[str, Any]:
    """Compact, context-friendly view of a run result (full detail on disk)."""
    subgoals = result_dict.get("subgoals", [])
    run_dir = result_dict.get("run_directory", "")
    replay = Path(run_dir) / "replay.html" if run_dir else None
    return {
        "goal": result_dict.get("goal"),
        "outcome": result_dict.get("outcome"),
        "reason": result_dict.get("reason"),
        "run_directory": run_dir,
        "replay_html": str(replay) if replay and replay.is_file() else None,
        "report_html": str(Path(run_dir) / "report.html") if run_dir else None,
        "subgoals": {
            "total": len(subgoals),
            "passed": sum(1 for s in subgoals if s.get("status") == "passed"),
            "items": [
                {"number": s.get("number"), "description": s.get("description"),
                 "status": s.get("status"), "evidence": s.get("evidence")}
                for s in subgoals
            ],
        },
        "grounding": result_dict.get("grounding", {}),
        "scene_graph": (result_dict.get("scene_graph", {}) or {}).get("coverage", {}),
        "pending_findings": (result_dict.get("scene_graph", {}) or {}).get(
            "pending_findings", []
        ),
        "crashes": result_dict.get("crashes", []),
    }


def run_validation(
    goal: str,
    *,
    config_path: Optional[str] = None,
    knowledge_profile: Optional[str] = None,
    knowledge_root: Optional[str] = None,
    exec_profile: Optional[str] = None,
    verification_level: Optional[str] = None,
    serial: Optional[str] = None,
    display_id: Optional[int] = None,
    output: str = "runs",
    dry_run: bool = False,
    device: Any = None,
    model: Any = None,
    knowledge: Any = None,
) -> dict[str, Any]:
    """Run one goal against the connected IVI and return a result summary."""
    config = _build_config(config_path, exec_profile)
    if verification_level:
        config.verification_level = verification_level
    device = device or AdbDevice(serial, display_id)
    model = model or _build_model(config)
    if knowledge is None:
        knowledge = _build_knowledge(config, knowledge_profile, knowledge_root)
    agent = GoalAgent(device, model, config, knowledge=knowledge)
    result = agent.run(goal, Path(output), dry_run=dry_run)
    return _summarize_result(result.to_dict())


def device_state(
    *,
    serial: Optional[str] = None,
    display_id: Optional[int] = None,
    output: Optional[str] = None,
    device: Any = None,
    max_elements: int = 30,
) -> dict[str, Any]:
    """Snapshot the live screen: titles, visible text, clickable candidates."""
    device = device or AdbDevice(serial, display_id)
    out_dir = Path(output) if output else Path(tempfile.gettempdir())
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / "device_state.png"
    device.capture(shot)
    try:
        ui_dump = device.ui_dump()
    except Exception:  # noqa: BLE001 - a tree-less screen is still reportable
        ui_dump = ""
    titles = extract_screen_titles(ui_dump) or extract_tree_headings(ui_dump)
    # extract_ui_elements returns only clickable candidates already.
    elements = extract_ui_elements(ui_dump, limit=max_elements)
    return {
        "screen_size": list(device.screen_size()),
        "titles": titles,
        "visible_text": extract_visible_text(ui_dump)[:max_elements],
        "clickable_elements": [
            {"label": e.label, "center": [round(e.center[0], 4), round(e.center[1], 4)],
             "role": e.role}
            for e in elements
        ],
        "ui_tree_available": bool(ui_dump),
        "screenshot": str(shot),
    }


def inspect_trace(run_directory: str) -> dict[str, Any]:
    """Summarize a past run directory: result, event counts, per-step brief."""
    directory = Path(run_directory)
    if not directory.is_dir():
        return {"error": f"no such run directory: {run_directory}"}
    result = _load_json(directory / "result.json")
    events = _load_events(directory / "events.jsonl")
    kinds: dict[str, int] = {}
    for event in events:
        kind = str(event.get("kind", "?"))
        kinds[kind] = kinds.get(kind, 0) + 1
    steps = [
        {
            "number": s.get("number"),
            "action": (s.get("action") or {}).get("type"),
            "target": (s.get("action") or {}).get("target"),
            "grounded_by": s.get("grounded_by"),
            "screen_changed": s.get("screen_changed"),
            "decision_seconds": s.get("decision_seconds"),
        }
        for s in result.get("steps", [])
    ]
    return {
        "run_directory": str(directory),
        "goal": result.get("goal"),
        "outcome": result.get("outcome"),
        "reason": result.get("reason"),
        "grounding": result.get("grounding", {}),
        "event_counts": kinds,
        "total_events": len(events),
        "steps": steps,
        "crashes": result.get("crashes", []),
        "replay_html": str(directory / "replay.html")
        if (directory / "replay.html").is_file()
        else None,
    }


def scene_graph_state(
    knowledge_profile: str, knowledge_root: str = "knowledge"
) -> dict[str, Any]:
    """Report manual-vs-HMI scene-graph coverage and pending divergences."""
    graph_path = Path(knowledge_root).resolve() / knowledge_profile / "scene_graph.json"
    if not graph_path.is_file():
        return {
            "error": f"no scene graph for profile {knowledge_profile!r} at {graph_path}. "
            "Run `ivi-agent graph build` or a validation first."
        }
    graph = SceneGraph.load(graph_path)
    return {
        "profile": knowledge_profile,
        "path": str(graph_path),
        "coverage": graph.coverage(),
        "pending_findings": [
            {"id": f.id, "kind": f.kind, "detail": f.detail, "status": f.status}
            for f in graph.pending_findings()
        ],
    }


def list_runs(output: str = "runs", limit: int = 20, offset: int = 0) -> dict[str, Any]:
    """List recent run directories (newest first) with their outcomes."""
    root = Path(output)
    if not root.is_dir():
        return {"total": 0, "count": 0, "offset": offset, "runs": [], "has_more": False}
    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and (d / "result.json").is_file()),
        key=lambda d: d.name,
        reverse=True,
    )
    total = len(dirs)
    page = dirs[offset : offset + limit]
    runs = []
    for directory in page:
        result = _load_json(directory / "result.json")
        runs.append(
            {
                "run_directory": str(directory),
                "goal": result.get("goal"),
                "outcome": result.get("outcome"),
                "started_at": result.get("started_at"),
                "crashes": len(result.get("crashes", []) or []),
            }
        )
    return {
        "total": total,
        "count": len(runs),
        "offset": offset,
        "runs": runs,
        "has_more": total > offset + len(runs),
        "next_offset": offset + len(runs) if total > offset + len(runs) else None,
    }


def preflight(config_path: Optional[str] = None) -> dict[str, Any]:
    """Check local dependencies (adb, scrcpy, ollama, tesseract) and the model."""
    from .cli import run_doctor_checks

    config = Config.load(config_path)
    checks = run_doctor_checks(config)
    return {
        "ok": all(passed for _, passed, _ in checks),
        "checks": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in checks
        ],
    }


# ---------------------------------------------------------------------------
# FastMCP server (imports mcp lazily so helpers work without the extra)
# ---------------------------------------------------------------------------


def build_server():  # type: ignore[no-untyped-def]
    """Construct and return the FastMCP server. Requires the ``[mcp]`` extra."""
    try:
        from mcp.server.fastmcp import FastMCP
        from pydantic import BaseModel, Field, ConfigDict
    except ImportError as exc:  # pragma: no cover - import-guard message
        raise SystemExit(
            "The MCP server needs the 'mcp' package. Install it with:\n"
            "  pip install 'ivi-visual-agent[mcp]'"
        ) from exc

    mcp = FastMCP("ivi_agent_mcp")
    _profiles = ", ".join(sorted(PROFILES))
    _levels = ", ".join(VERIFICATION_LEVELS)

    class _Base(BaseModel):
        model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    class RunValidationInput(_Base):
        goal: str = Field(..., description="Goal in plain language, e.g. 'Start the seat massage'", min_length=1)
        config_path: Optional[str] = Field(default=None, description="Path to a JSON config file")
        knowledge_profile: Optional[str] = Field(default=None, description="Local manual knowledge profile (e.g. 'benz')")
        knowledge_root: Optional[str] = Field(default=None, description="Directory holding knowledge profiles")
        exec_profile: Optional[str] = Field(default=None, description=f"Execution profile: {_profiles}")
        verification_level: Optional[str] = Field(default=None, description=f"How hard to prove success: {_levels}")
        serial: Optional[str] = Field(default=None, description="ADB device serial")
        display_id: Optional[int] = Field(default=None, description="Android display id", ge=0)
        output: str = Field(default="runs", description="Evidence output directory")
        dry_run: bool = Field(default=False, description="Plan one action without executing it")

    class DeviceStateInput(_Base):
        serial: Optional[str] = Field(default=None, description="ADB device serial")
        display_id: Optional[int] = Field(default=None, description="Android display id", ge=0)
        output: Optional[str] = Field(default=None, description="Where to save the screenshot (temp dir if unset)")
        max_elements: int = Field(default=30, description="Cap on listed elements/text lines", ge=1, le=200)

    class InspectTraceInput(_Base):
        run_directory: str = Field(..., description="Path to a run directory (e.g. runs/20260917T151917Z)", min_length=1)

    class SceneGraphInput(_Base):
        knowledge_profile: str = Field(..., description="Knowledge profile name (e.g. 'benz')", min_length=1)
        knowledge_root: str = Field(default="knowledge", description="Directory holding knowledge profiles")

    class ListRunsInput(_Base):
        output: str = Field(default="runs", description="Run directory root")
        limit: int = Field(default=20, description="Max runs to return", ge=1, le=100)
        offset: int = Field(default=0, description="Runs to skip for pagination", ge=0)

    class PreflightInput(_Base):
        config_path: Optional[str] = Field(default=None, description="Path to a JSON config file")

    @mcp.tool(
        name="ivi_run_validation",
        annotations={"title": "Run IVI Validation", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def ivi_run_validation(params: RunValidationInput) -> str:
        """Run a plain-language validation goal against the connected IVI/HMI.

        Drives the device (taps/swipes) to reach the goal, grounding each step by
        accessibility tree -> OpenCV -> vision model, and writes full evidence to
        a run directory (screenshots, result.json, report.html, replay.html,
        events.jsonl). Use this to execute a test; use ivi_device_state to only
        look, ivi_inspect_trace to read a finished run.

        Returns JSON: {goal, outcome (pass|fail|inconclusive), reason,
        run_directory, replay_html, subgoals{total,passed,items}, grounding,
        scene_graph coverage, pending_findings, crashes}.
        """
        try:
            return json.dumps(run_validation(
                params.goal, config_path=params.config_path,
                knowledge_profile=params.knowledge_profile, knowledge_root=params.knowledge_root,
                exec_profile=params.exec_profile, verification_level=params.verification_level,
                serial=params.serial,
                display_id=params.display_id, output=params.output, dry_run=params.dry_run,
            ), indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {type(exc).__name__}: {exc}"

    @mcp.tool(
        name="ivi_device_state",
        annotations={"title": "Inspect Live IVI Screen", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def ivi_device_state(params: DeviceStateInput) -> str:
        """Snapshot the current IVI screen without changing anything.

        Captures a screenshot and reads the accessibility tree, returning the
        screen title(s), visible text, and clickable elements with their centers.
        Use this to see what is on screen before deciding what to validate.

        Returns JSON: {screen_size:[w,h], titles[], visible_text[],
        clickable_elements[{label,center,clickable}], ui_tree_available, screenshot}.
        """
        try:
            return json.dumps(device_state(
                serial=params.serial, display_id=params.display_id,
                output=params.output, max_elements=params.max_elements,
            ), indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {type(exc).__name__}: {exc}"

    @mcp.tool(
        name="ivi_inspect_trace",
        annotations={"title": "Inspect Run Evidence", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def ivi_inspect_trace(params: InspectTraceInput) -> str:
        """Summarize a finished run directory's evidence.

        Reads result.json and events.jsonl and returns the outcome, per-step
        actions and grounding, event-kind counts, crashes, and the replay path.

        Returns JSON: {run_directory, goal, outcome, reason, grounding,
        event_counts, total_events, steps[], crashes[], replay_html}.
        """
        return json.dumps(inspect_trace(params.run_directory), indent=2, default=str)

    @mcp.tool(
        name="ivi_scene_graph",
        annotations={"title": "Scene Graph Coverage", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def ivi_scene_graph(params: SceneGraphInput) -> str:
        """Report manual-vs-HMI scene-graph coverage and pending divergences.

        Returns confirmed/undocumented screen counts, defects, and any
        divergences awaiting review (candidate HMI defects) for a profile.

        Returns JSON: {profile, path, coverage{...}, pending_findings[{id,kind,detail,status}]}.
        """
        return json.dumps(
            scene_graph_state(params.knowledge_profile, params.knowledge_root),
            indent=2, default=str,
        )

    @mcp.tool(
        name="ivi_list_runs",
        annotations={"title": "List Runs", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def ivi_list_runs(params: ListRunsInput) -> str:
        """List recent validation runs (newest first) with their outcomes.

        Returns JSON: {total, count, offset, runs[{run_directory,goal,outcome,
        started_at,crashes}], has_more, next_offset}.
        """
        return json.dumps(
            list_runs(params.output, params.limit, params.offset), indent=2, default=str
        )

    @mcp.tool(
        name="ivi_doctor",
        annotations={"title": "Preflight Check", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def ivi_doctor(params: PreflightInput) -> str:
        """Check local dependencies (adb, scrcpy, ollama, tesseract) and the model.

        Returns JSON: {ok, checks[{name,passed,detail}]}.
        """
        return json.dumps(preflight(params.config_path), indent=2, default=str)

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
