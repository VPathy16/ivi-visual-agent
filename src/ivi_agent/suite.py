from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .adb import AdbDevice
from .agent import GoalAgent
from .config import Config
from .knowledge import KnowledgeBase
from .model import OllamaVisionModel


def load_suite_cases(path: Path) -> list[dict[str, str]]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Suite file must contain a non-empty cases list")
    cases: list[dict[str, str]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Suite case {index} must be an object")
        name = raw.get("name")
        goal = raw.get("goal")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Suite case {index} needs a name")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"Suite case {index} needs a goal")
        name = name.strip()
        if name in names:
            raise ValueError(f"Duplicate suite case name: {name}")
        names.add(name)
        cases.append({"name": name, "goal": goal.strip()})
    return cases


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "case"


def _write_suite_report(summary: dict[str, Any], directory: Path) -> None:
    (directory / "suite-result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows: list[str] = []
    for case in summary["cases"]:
        report = Path(case["run_directory"]) / "report.html"
        relative_report = report.relative_to(directory)
        rows.append(
            "<tr>"
            f"<td>{html.escape(case['name'])}</td>"
            f"<td>{html.escape(case['goal'])}</td>"
            f"<td class=\"{html.escape(case['outcome'])}\">"
            f"{html.escape(case['outcome'].upper())}</td>"
            f"<td>{case['actions']}</td>"
            f"<td>{case['duration_seconds']:.1f}s</td>"
            f"<td>{html.escape(case['reason'])}</td>"
            f"<td><a href=\"{html.escape(str(relative_report))}\">report</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>IVI smoke suite</title>
<style>
body {{ font: 15px system-ui; margin: 32px; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dadce0; padding: 8px; text-align: left; }}
th {{ background: #f1f3f4; }}
.pass {{ color: #137333; font-weight: 700; }}
.fail {{ color: #b3261e; font-weight: 700; }}
.inconclusive {{ color: #b06000; font-weight: 700; }}
</style></head><body>
<h1>IVI smoke suite: {html.escape(summary['outcome'].upper())}</h1>
<p>{summary['passed']}/{summary['total']} passed · {summary['duration_seconds']:.1f}s</p>
<table><thead><tr><th>Case</th><th>Goal</th><th>Outcome</th><th>Actions</th>
<th>Duration</th><th>Evidence / reason</th><th>Details</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    (directory / "suite-report.html").write_text(document, encoding="utf-8")


def run_suite(
    device: AdbDevice,
    model: OllamaVisionModel,
    config: Config,
    cases: list[dict[str, str]],
    output_root: Path,
    knowledge: KnowledgeBase | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = progress or (lambda _message: None)
    started = datetime.now(timezone.utc)
    directory = (output_root / started.strftime("%Y%m%dT%H%M%SZ")).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    device.ensure_ready()
    device.wake_if_needed()
    used_slugs: set[str] = set()
    for index, case in enumerate(cases, start=1):
        slug = _slug(case["name"])
        if slug in used_slugs:
            slug = f"{slug}-{index}"
        used_slugs.add(slug)
        emit(f"Case {index}/{len(cases)}: {case['name']} — returning Home")
        device.go_home()
        case_started = datetime.now(timezone.utc)
        agent = GoalAgent(
            device,
            model,
            config,
            knowledge=knowledge,
            progress=lambda message, prefix=case["name"]: emit(f"[{prefix}] {message}"),
        )
        result = agent.run(case["goal"], directory / slug)
        case_finished = datetime.now(timezone.utc)
        results.append(
            {
                "name": case["name"],
                "goal": case["goal"],
                "outcome": result.outcome,
                "reason": result.reason,
                "actions": len(result.steps),
                "duration_seconds": (case_finished - case_started).total_seconds(),
                "run_directory": result.run_directory,
            }
        )
        emit(f"Case {index}/{len(cases)}: {case['name']} — {result.outcome.upper()}")
    finished = datetime.now(timezone.utc)
    passed = sum(case["outcome"] == "pass" for case in results)
    summary: dict[str, Any] = {
        "outcome": "pass" if passed == len(results) else "fail",
        "passed": passed,
        "total": len(results),
        "duration_seconds": (finished - started).total_seconds(),
        "run_directory": str(directory.resolve()),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "cases": results,
    }
    _write_suite_report(summary, directory)
    return summary
