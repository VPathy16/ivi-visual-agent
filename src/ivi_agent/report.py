from __future__ import annotations

import html
import json
from pathlib import Path

from .types import RunResult


def write_report(result: RunResult) -> None:
    directory = Path(result.run_directory)
    (directory / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    step_rows = []
    for step in result.steps:
        action = step.action
        step_rows.append(
            "<tr>"
            f"<td>{step.number}</td>"
            f"<td><a href='{html.escape(step.screenshot)}'><img src='{html.escape(step.screenshot)}'></a></td>"
            f"<td>{html.escape(action.type)}</td>"
            f"<td>{html.escape(action.target)}</td>"
            f"<td>{action.confidence:.2f}</td>"
            f"<td>{html.escape(action.reason)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>IVI Agent Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}
.outcome{{font-size:1.4rem;font-weight:700}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:.5rem;text-align:left;vertical-align:top}}
img{{width:280px;height:auto}} th{{background:#f4f4f4}}
</style></head><body>
<h1>IVI Visual Agent Report</h1>
<p><strong>Goal:</strong> {html.escape(result.goal)}</p>
<p class="outcome">Outcome: {html.escape(result.outcome.upper())}</p>
<p>{html.escape(result.reason)}</p>
<p>{html.escape(result.started_at)} — {html.escape(result.finished_at)}</p>
<table><thead><tr><th>#</th><th>Screen</th><th>Action</th><th>Target</th><th>Confidence</th><th>Reason</th></tr></thead>
<tbody>{''.join(step_rows)}</tbody></table></body></html>"""
    (directory / "report.html").write_text(document, encoding="utf-8")

