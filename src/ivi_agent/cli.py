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

    mirror = commands.add_parser("scrcpy", help="Open a live scrcpy view")
    mirror.add_argument("--serial", help="ADB device serial")
    mirror.add_argument("--record", help="Optional MP4 recording path")
    return root


def main() -> None:
    args = parser().parse_args()
    try:
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
        agent = GoalAgent(
            device,
            model,
            config,
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
