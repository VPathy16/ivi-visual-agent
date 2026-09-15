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
from .config import Config
from .model import OllamaVisionModel
from .suite import load_suite_cases, run_suite


def doctor(config: Config) -> int:
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

    suite = commands.add_parser("suite", help="Run independent goals from Home")
    suite.add_argument("--cases", required=True, help="Path to suite case JSON")
    suite.add_argument("--serial", help="ADB device serial")
    suite.add_argument("--display-id", type=int, help="Android display ID")
    suite.add_argument(
        "--output", default="runs/suites", help="Suite evidence output directory"
    )
    suite.add_argument("--knowledge-profile", help="Local manual knowledge profile")
    suite.add_argument("--knowledge-root", help="Directory containing knowledge profiles")

    mirror = commands.add_parser("scrcpy", help="Open a live scrcpy view")
    mirror.add_argument("--serial", help="ADB device serial")
    mirror.add_argument("--record", help="Optional MP4 recording path")

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

            if args.knowledge_command == "index":
                summary = index_pdf(Path(args.pdf), Path(args.output), args.profile)
            else:
                summary = KnowledgeBase.open(Path(args.root), args.profile).query(
                    args.goal, args.limit
                )
            print(json.dumps(summary, indent=2))
            raise SystemExit(0)
        config = Config.load(args.config)
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
        )
        from .knowledge import KnowledgeBase

        profile = args.knowledge_profile or config.knowledge_profile
        knowledge_root = Path(args.knowledge_root or config.knowledge_root)
        knowledge_base = (
            KnowledgeBase.open(knowledge_root, profile) if profile else None
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
