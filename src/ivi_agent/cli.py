from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .adb import AdbDevice
from .agent import GoalAgent
from .config import Config, PROFILES, VERIFICATION_LEVELS
from .model import OllamaVisionModel
from .suite import load_suite_cases, run_suite


def run_doctor_checks(config: Config) -> list[tuple[str, bool, str]]:
    """Return preflight checks as (name, passed, detail) — no printing.

    Shared by the `doctor` CLI command and the MCP server so both report the
    same local-dependency and model-availability state.
    """
    checks: list[tuple[str, bool, str]] = []
    for executable in ("adb", "scrcpy", "ollama", "tesseract"):
        location = shutil.which(executable)
        checks.append((executable, bool(location), location or "not found in PATH"))
    try:
        with urllib.request.urlopen(f"{config.ollama_url.rstrip('/')}/api/tags", timeout=3) as response:
            payload = json.loads(response.read())
        names = [item.get("name", "") for item in payload.get("models", [])]
        available = config.model in names
        checks.append((f"model {config.model}", available, "available" if available else "not pulled"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        checks.append(("Ollama API", False, str(exc)))
    return checks


def doctor(config: Config) -> int:
    checks = run_doctor_checks(config)
    width = max(len(name) for name, _, _ in checks)
    for name, passed, detail in checks:
        print(f"{'OK' if passed else 'MISSING':7} {name:<{width}}  {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def launch_scrcpy(serial: str | None, record: str | None) -> int:
    command = ["scrcpy"]
    if serial:
        command += ["--serial", serial]
    if record:
        command += ["--record", record]
    try:
        return subprocess.call(command)
    except FileNotFoundError:
        print("scrcpy was not found. Install it and add it to PATH.", file=sys.stderr)
        return 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local goal-driven visual agent for Android IVI")
    root.add_argument("--config", help="Path to JSON configuration")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check local dependencies")

    run = commands.add_parser("run", help="Run a goal against a connected IVI")
    run.add_argument("--goal", required=True, help="Goal stated in plain language")
    run.add_argument("--serial", help="ADB device serial")
    run.add_argument("--display-id", type=int, help="Android display ID")
    run.add_argument("--output", default="runs", help="Evidence output directory")
    run.add_argument("--dry-run", action="store_true", help="Plan one action without executing it")
    run.add_argument("--knowledge-profile", help="Local manual knowledge profile")
    run.add_argument("--knowledge-root", help="Directory containing knowledge profiles")
    run.add_argument(
        "--profile",
        dest="exec_profile",
        choices=sorted(PROFILES),
        help="Execution profile: fast (minimum overhead) | balanced (default) | "
        "strict (verify every step). config.json / other flags override it.",
    )
    run.add_argument(
        "--verification-level",
        choices=list(VERIFICATION_LEVELS),
        help="How hard to prove success: off | final | checkpoints | strict. "
        "Overrides config and profile.",
    )

    suite = commands.add_parser("suite", help="Run independent goals from Home")
    suite.add_argument("--cases", required=True, help="Path to suite case JSON")
    suite.add_argument("--serial", help="ADB device serial")
    suite.add_argument("--display-id", type=int, help="Android display ID")
    suite.add_argument(
        "--output", default="runs/suites", help="Suite evidence output directory"
    )
    suite.add_argument("--knowledge-profile", help="Local manual knowledge profile")
    suite.add_argument("--knowledge-root", help="Directory containing knowledge profiles")
    suite.add_argument(
        "--profile",
        dest="exec_profile",
        choices=sorted(PROFILES),
        help="Execution profile: fast | balanced | strict (see `run --help`).",
    )
    suite.add_argument(
        "--verification-level",
        choices=list(VERIFICATION_LEVELS),
        help="How hard to prove success: off | final | checkpoints | strict.",
    )

    mirror = commands.add_parser("scrcpy", help="Open a live scrcpy view")
    mirror.add_argument("--serial", help="ADB device serial")
    mirror.add_argument("--record", help="Optional MP4 recording path")

    replay = commands.add_parser(
        "replay", help="Build an interactive replay.html for a finished run"
    )
    replay.add_argument("--run", required=True, help="Path to a run directory")

    manual = commands.add_parser("manual", help="Build a RAG-friendly PDF manual")
    manual_commands = manual.add_subparsers(dest="manual_command", required=True)
    manual_build = manual_commands.add_parser(
        "build", help="Validate a manual source folder and generate its PDF"
    )
    manual_build.add_argument(
        "--source", required=True, help="Folder containing manual.json and images/"
    )
    manual_build.add_argument("--output", required=True, help="Generated PDF path")

    knowledge = commands.add_parser("knowledge", help="Manage local PDF knowledge")
    knowledge_commands = knowledge.add_subparsers(
        dest="knowledge_command", required=True
    )
    knowledge_index = knowledge_commands.add_parser(
        "index", help="Create a local searchable profile from a PDF"
    )
    knowledge_index.add_argument("pdf", help="RAG-friendly PDF manual")
    knowledge_index.add_argument("--profile", required=True, help="Profile name")
    knowledge_index.add_argument(
        "--output", default="knowledge", help="Knowledge profile root"
    )
    knowledge_query = knowledge_commands.add_parser(
        "query", help="Retrieve relevant manual knowledge for a goal"
    )
    knowledge_query.add_argument("--profile", required=True, help="Profile name")
    knowledge_query.add_argument("--goal", required=True, help="Goal or current state")
    knowledge_query.add_argument(
        "--root", default="knowledge", help="Knowledge profile root"
    )
    knowledge_query.add_argument("--limit", type=int, default=4)

    graph = commands.add_parser(
        "graph", help="Inspect and review the living scene graph (HMI defects)"
    )
    graph_commands = graph.add_subparsers(dest="graph_command", required=True)
    graph_build = graph_commands.add_parser(
        "build", help="Seed the expected scene graph from an indexed profile's manual"
    )
    graph_build.add_argument("--profile", required=True, help="Profile name")
    graph_build.add_argument("--root", default="knowledge", help="Knowledge profile root")
    graph_build.add_argument(
        "--force", action="store_true", help="Overwrite an existing scene graph"
    )
    graph_show = graph_commands.add_parser(
        "show", help="Show coverage, nodes, and pending divergence findings"
    )
    graph_show.add_argument("--profile", required=True, help="Profile name")
    graph_show.add_argument("--root", default="knowledge", help="Knowledge profile root")
    graph_review = graph_commands.add_parser(
        "review", help="Approve a finding (legitimate) or mark it a defect"
    )
    graph_review.add_argument("--profile", required=True, help="Profile name")
    graph_review.add_argument("--root", default="knowledge", help="Knowledge profile root")
    graph_review.add_argument("--finding", required=True, help="Finding id, e.g. F0001")
    graph_review.add_argument(
        "--decision", required=True, choices=["approve", "defect"]
    )
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "manual":
            from .manual_pdf import build_manual_pdf

            summary = build_manual_pdf(Path(args.source), Path(args.output))
            print(json.dumps(summary, indent=2))
            raise SystemExit(0)
        if args.command == "knowledge":
            from .knowledge import KnowledgeBase, index_pdf
            from .embeddings import resolve_image_embedder, resolve_text_embedder

            knowledge_config = Config.load(args.config)
            text_embedder = resolve_text_embedder(
                knowledge_config.ollama_url,
                knowledge_config.embedding_model,
                knowledge_config.use_embeddings,
                knowledge_config.embedding_backend,
            )
            if args.knowledge_command == "index":
                image_embedder = resolve_image_embedder(
                    knowledge_config.clip_model, knowledge_config.icon_matching
                )
                summary = index_pdf(
                    Path(args.pdf),
                    Path(args.output),
                    args.profile,
                    text_embedder=text_embedder,
                    image_embedder=image_embedder,
                )
            else:
                summary = KnowledgeBase.open(
                    Path(args.root), args.profile, embedder=text_embedder
                ).query(args.goal, args.limit)
            print(json.dumps(summary, indent=2))
            raise SystemExit(0)
        if args.command == "graph":
            from .graph import SceneGraph
            from .knowledge import KnowledgeBase

            profile_dir = Path(args.root).resolve() / args.profile
            graph_path = profile_dir / "scene_graph.json"
            if args.graph_command == "build":
                if graph_path.is_file() and not args.force:
                    print(
                        f"scene graph already exists: {graph_path} (use --force to overwrite)",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                kb = KnowledgeBase.open(Path(args.root), args.profile)
                manual = {
                    "screens": [
                        chunk["data"]
                        for chunk in kb.chunks
                        if chunk.get("kind") == "screen" and isinstance(chunk.get("data"), dict)
                    ]
                }
                built = SceneGraph.from_manual(manual)
                built.save(graph_path)
                print(json.dumps({"screens": len(built.nodes), "edges": len(built.edges), "path": str(graph_path)}, indent=2))
                raise SystemExit(0)
            if not graph_path.is_file():
                print(f"no scene graph yet: {graph_path} (run `graph build` or a run first)", file=sys.stderr)
                raise SystemExit(1)
            graph_obj = SceneGraph.load(graph_path)
            if args.graph_command == "show":
                print(json.dumps({
                    "coverage": graph_obj.coverage(),
                    "nodes": [
                        {"id": n.id, "name": n.name, "source": n.source, "status": n.status, "times_seen": n.times_seen}
                        for n in graph_obj.nodes.values()
                    ],
                    "pending_findings": [
                        {"id": f.id, "kind": f.kind, "detail": f.detail} for f in graph_obj.pending_findings()
                    ],
                }, indent=2))
                raise SystemExit(0)
            if args.graph_command == "review":
                finding = graph_obj.review(args.finding, args.decision)
                graph_obj.save(graph_path)
                print(json.dumps({"finding": finding.id, "status": finding.status, "node": finding.node_id}, indent=2))
                raise SystemExit(0)
        if args.command == "replay":
            from .replay import build_replay

            run_dir = Path(args.run)
            if not run_dir.is_dir():
                print(f"no such run directory: {run_dir}", file=sys.stderr)
                raise SystemExit(1)
            out = build_replay(run_dir)
            print(json.dumps({"replay": str(out)}, indent=2))
            raise SystemExit(0)
        config = Config.load(args.config)
        config.apply_profile(getattr(args, "exec_profile", None))
        # An explicit --verification-level wins over config and profile.
        cli_level = getattr(args, "verification_level", None)
        if cli_level:
            config.verification_level = cli_level
        if args.command == "doctor":
            raise SystemExit(doctor(config))
        if args.command == "scrcpy":
            raise SystemExit(launch_scrcpy(args.serial, args.record))
        device = AdbDevice(args.serial, args.display_id)
        model = OllamaVisionModel(
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
        from .knowledge import KnowledgeBase
        from .embeddings import resolve_text_embedder

        profile = args.knowledge_profile or config.knowledge_profile
        knowledge_root = Path(args.knowledge_root or config.knowledge_root)
        text_embedder = resolve_text_embedder(
            config.ollama_url,
            config.embedding_model,
            config.use_embeddings,
            config.embedding_backend,
        )
        knowledge_base = (
            KnowledgeBase.open(knowledge_root, profile, embedder=text_embedder)
            if profile
            else None
        )
        if args.command == "suite":
            summary = run_suite(
                device,
                model,
                config,
                load_suite_cases(Path(args.cases)),
                Path(args.output),
                knowledge=knowledge_base,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            print(json.dumps(summary, indent=2))
            raise SystemExit(0 if summary["outcome"] == "pass" else 2)
        agent = GoalAgent(
            device,
            model,
            config,
            knowledge=knowledge_base,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        result = agent.run(args.goal, Path(args.output), dry_run=args.dry_run)
        print(json.dumps(result.to_dict(), indent=2))
        raise SystemExit(0 if result.outcome == "pass" else 2)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
