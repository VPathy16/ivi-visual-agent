"""Interactive replay console for a finished run.

Turns a run directory (``result.json`` + ``events.jsonl`` + the step screenshots)
into a single self-contained ``replay.html`` you can open or send to someone:
a step scrubber that shows, for each step, the screen, the action taken, how it
was grounded (a11y / CV / model), what the reasoning event stream said, and any
scene-graph divergence or crash — plus a run summary with the per-phase wall
timing. This is the shareable analog of a live test console; everything is
inlined (screenshots as data URIs, JS and CSS in the file) so it works offline
from ``file://`` with no server and no network.

Design notes:
* Pure standard library. Screenshots are base64-embedded so the file is
  portable; a run with a handful of steps stays small.
* Best-effort: a missing ``events.jsonl`` or screenshot degrades gracefully
  rather than failing, so replay works on old runs too.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def _data_uri(path: Path) -> str:
    """Return a base64 PNG data URI for a screenshot, or '' if unreadable."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if not raw:
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def build_replay_data(directory: Path) -> dict[str, Any]:
    """Assemble the replay payload from a run directory (no HTML)."""
    result = _load_json(directory / "result.json")
    events = _load_events(directory / "events.jsonl")
    # Group events by step so each step carries its own reasoning stream; keep
    # the step-less ones (plan, run_start, done, crash) as the run timeline.
    per_step: dict[int, list[dict[str, Any]]] = {}
    run_events: list[dict[str, Any]] = []
    for event in events:
        step = event.get("step")
        if isinstance(step, int):
            per_step.setdefault(step, []).append(event)
        else:
            run_events.append(event)

    steps = []
    for step in result.get("steps", []):
        number = step.get("number")
        shot = step.get("screenshot", "")
        after = f"step-{number:02d}-after.png" if isinstance(number, int) else ""
        steps.append(
            {
                **step,
                "image": _data_uri(directory / shot) if shot else "",
                "after_image": _data_uri(directory / after) if after else "",
                "events": per_step.get(number, []) if isinstance(number, int) else [],
            }
        )
    return {"result": result, "steps": steps, "run_events": run_events}


def build_replay(directory: Path) -> Path:
    """Write ``replay.html`` into ``directory`` and return its path."""
    directory = Path(directory)
    data = build_replay_data(directory)
    # Embed the payload safely inside a <script> tag: only "</" needs escaping so
    # the parser can't see a premature </script>.
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    document = _TEMPLATE.replace("__DATA__", payload)
    out = directory / "replay.html"
    out.write_text(document, encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IVI Replay</title>
<style>
:root{
  --bg:#0f1216; --panel:#171b21; --line:#262c35; --text:#e6e9ee; --muted:#9aa4b2;
  --a11y:#3fb950; --cv:#d29922; --model:#a371f7; --pass:#3fb950; --fail:#f85149;
  --accent:#58a6ff;
}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.45}
header{padding:16px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:1.05rem;margin:0 0 4px} .goal{color:var(--muted);font-size:.9rem}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.72rem;
  font-weight:700;letter-spacing:.02em}
.b-pass{background:rgba(63,185,80,.15);color:var(--pass)}
.b-fail{background:rgba(248,81,73,.15);color:var(--fail)}
.b-other{background:rgba(154,164,178,.15);color:var(--muted)}
.b-a11y{background:rgba(63,185,80,.15);color:var(--a11y)}
.b-cv{background:rgba(210,153,34,.15);color:var(--cv)}
.b-model{background:rgba(163,113,247,.15);color:var(--model)}
.summary{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:10px;font-size:.82rem;color:var(--muted)}
.summary b{color:var(--text)}
.phases{display:flex;gap:2px;margin-top:8px;height:10px;border-radius:4px;overflow:hidden;max-width:560px}
.phases div{height:100%} .phases-legend{font-size:.72rem;color:var(--muted);margin-top:4px}
main{display:grid;grid-template-columns:230px 1fr;gap:0;min-height:calc(100vh - 130px)}
nav{border-right:1px solid var(--line);background:var(--panel);overflow-y:auto;max-height:calc(100vh - 130px)}
.step-item{padding:10px 14px;border-bottom:1px solid var(--line);cursor:pointer;font-size:.85rem}
.step-item:hover{background:#1d222a}
.step-item.active{background:#1f2a3a;border-left:3px solid var(--accent);padding-left:11px}
.step-item .row{display:flex;justify-content:space-between;align-items:center;gap:6px}
.step-item .sub{color:var(--muted);font-size:.75rem;margin-top:3px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
section{padding:18px 22px;overflow-y:auto;max-height:calc(100vh - 130px)}
.stage{display:grid;grid-template-columns:minmax(240px,380px) 1fr;gap:22px;align-items:start}
.shot{width:100%;border:1px solid var(--line);border-radius:8px;background:#000}
.shot-tabs{display:flex;gap:6px;margin-bottom:8px}
.shot-tabs button{background:var(--panel);color:var(--muted);border:1px solid var(--line);
  border-radius:6px;padding:3px 10px;font-size:.75rem;cursor:pointer}
.shot-tabs button.on{color:var(--text);border-color:var(--accent)}
.kv{margin:0 0 14px} .kv div{display:flex;gap:10px;padding:4px 0;border-bottom:1px solid var(--line);font-size:.86rem}
.kv .k{color:var(--muted);min-width:110px} .kv .v{color:var(--text);word-break:break-word}
h2{font-size:.9rem;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.events{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem}
.ev{padding:6px 8px;border-left:2px solid var(--line);margin-bottom:4px;background:#12161c;border-radius:0 4px 4px 0}
.ev .t{color:var(--accent);font-weight:700}
.ev .f{color:var(--muted);display:block;margin-top:2px;white-space:pre-wrap;word-break:break-word}
.controls{display:flex;gap:8px;align-items:center;margin-bottom:14px}
.controls button{background:var(--accent);color:#04121f;border:0;border-radius:6px;
  padding:6px 14px;font-weight:700;cursor:pointer}
.controls button.ghost{background:var(--panel);color:var(--text);border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:.82rem;margin-bottom:8px}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#12161c;color:var(--muted);font-weight:600}
.warn{color:var(--fail)} .empty{color:var(--muted);font-size:.85rem;padding:8px 0}
@media(max-width:760px){main{grid-template-columns:1fr}nav{max-height:none}
  .stage{grid-template-columns:1fr}section{max-height:none}}
</style></head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="goal" id="goal"></div>
  <div class="summary" id="summary"></div>
  <div class="phases" id="phases"></div>
  <div class="phases-legend" id="phases-legend"></div>
</header>
<main>
  <nav id="steplist"></nav>
  <section id="detail"></section>
</main>
<script>
const DATA = __DATA__;
const R = DATA.result || {};
const STEPS = DATA.steps || [];
const PHASE_COLORS = {startup:"#8b949e",retrieval:"#58a6ff",planning:"#a371f7",
  capture:"#d29922",ui_dump:"#f85149",ocr:"#db61a2",settle:"#3fb950",
  step_retrieval:"#1f6feb",graph_observe:"#56d4dd",logcat_dump:"#6e7681"};
const esc = s => String(s==null?"":s).replace(/[&<>"]/g,c=>(
  {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const groundClass = g => g==="a11y"?"b-a11y":g==="cv"?"b-cv":"b-model";
const groundLabel = g => g==="a11y"?"A11Y":g==="cv"?"CV":"MODEL";

function renderHeader(){
  document.getElementById("title").textContent = "IVI Visual Agent — Replay";
  const oc = (R.outcome||"").toUpperCase();
  const cls = oc==="PASS"?"b-pass":oc==="FAIL"?"b-fail":"b-other";
  document.getElementById("goal").innerHTML =
    `<span class="badge ${cls}">${esc(oc||"—")}</span> &nbsp;`+
    `${esc(R.goal||"")} <br><span style="font-size:.8rem">${esc(R.reason||"")}</span>`;
  const g = R.grounding||{}, cov=(R.scene_graph||{}).coverage||{};
  const parts=[];
  if(g.total_steps!=null) parts.push(`<b>${g.total_steps}</b> steps`);
  if(g.accessibility_fast_path_steps!=null)
    parts.push(`${g.accessibility_fast_path_steps} a11y / ${g.cv_fast_path_steps||0} cv / ${g.model_steps||0} model`);
  if(g.total_decision_seconds!=null) parts.push(`decision <b>${g.total_decision_seconds}s</b>`);
  if(g.total_wall_seconds!=null) parts.push(`wall <b>${g.total_wall_seconds}s</b>`);
  if(cov.manual_screens!=null)
    parts.push(`scene graph <b>${cov.confirmed_screens||0}/${cov.manual_screens}</b> · defects ${cov.defects||0}`);
  if((R.crashes||[]).length) parts.push(`<span class="warn">crashes ${R.crashes.length}</span>`);
  document.getElementById("summary").innerHTML = parts.join(" · ");
  renderPhases(g.phase_seconds||{});
}

function renderPhases(ph){
  const el=document.getElementById("phases"), leg=document.getElementById("phases-legend");
  const keys=Object.keys(ph); if(!keys.length){el.style.display="none";return;}
  const total=keys.reduce((a,k)=>a+(ph[k]||0),0)||1;
  el.innerHTML=keys.map(k=>{
    const c=PHASE_COLORS[k]||"#6e7681", w=((ph[k]||0)/total*100).toFixed(1);
    return `<div title="${esc(k)}: ${ph[k]}s" style="width:${w}%;background:${c}"></div>`;
  }).join("");
  leg.innerHTML="phase wall: "+keys.map(k=>
    `<span style="color:${PHASE_COLORS[k]||'#6e7681'}">■</span>${esc(k)} ${ph[k]}s`).join("  ");
}

function renderList(){
  const nav=document.getElementById("steplist");
  if(!STEPS.length){nav.innerHTML='<div class="empty" style="padding:14px">No steps recorded.</div>';return;}
  nav.innerHTML=STEPS.map((s,i)=>{
    const a=s.action||{}, g=s.grounded_by||"model";
    const changed = s.screen_changed===false?'· no change':s.screen_changed?'· changed':'';
    return `<div class="step-item" data-i="${i}">
      <div class="row"><span>Step ${esc(s.number)}</span>
        <span class="badge ${groundClass(g)}">${groundLabel(g)}</span></div>
      <div class="sub">${esc(a.type||"")} ${esc(a.target||a.direction||"")} ${changed}</div>
    </div>`;
  }).join("");
  nav.querySelectorAll(".step-item").forEach(el=>
    el.onclick=()=>select(+el.dataset.i));
}

let current=0, shotMode="before";
function select(i){
  current=Math.max(0,Math.min(STEPS.length-1,i));
  document.querySelectorAll(".step-item").forEach((el,idx)=>
    el.classList.toggle("active",idx===current));
  renderDetail();
}

function evField(ev){
  return Object.keys(ev).filter(k=>!["seq","ts","kind","step"].includes(k))
    .map(k=>`${k}=${typeof ev[k]==="object"?JSON.stringify(ev[k]):ev[k]}`).join("  ");
}

function renderDetail(){
  const s=STEPS[current]; if(!s){document.getElementById("detail").innerHTML=
    '<div class="empty">Nothing to show.</div>';return;}
  const a=s.action||{}, g=s.grounded_by||"model";
  const img = shotMode==="after" && s.after_image ? s.after_image : s.image;
  const hasAfter = !!s.after_image;
  const evHtml = (s.events||[]).map(ev=>
    `<div class="ev"><span class="t">${esc(ev.kind)}</span>
      <span class="f">${esc(evField(ev))}</span></div>`).join("")
    || '<div class="empty">No event stream for this step.</div>';
  document.getElementById("detail").innerHTML=`
    <div class="controls">
      <button class="ghost" onclick="select(current-1)">← Prev</button>
      <button class="ghost" onclick="select(current+1)">Next →</button>
      <span style="color:var(--muted);font-size:.8rem">Step ${esc(s.number)} of ${STEPS.length} — use ← → keys</span>
    </div>
    <div class="stage">
      <div>
        <div class="shot-tabs">
          <button class="${shotMode==='before'?'on':''}" onclick="shotMode='before';renderDetail()">Before action</button>
          ${hasAfter?`<button class="${shotMode==='after'?'on':''}" onclick="shotMode='after';renderDetail()">After action</button>`:''}
        </div>
        ${img?`<img class="shot" src="${img}" alt="screen">`:'<div class="empty">No screenshot embedded.</div>'}
      </div>
      <div>
        <div class="kv">
          <div><span class="k">Action</span><span class="v">${esc(a.type||"")} ${esc(a.target||a.direction||"")}</span></div>
          <div><span class="k">Grounded by</span><span class="v"><span class="badge ${groundClass(g)}">${groundLabel(g)}</span></span></div>
          <div><span class="k">Confidence</span><span class="v">${a.confidence!=null?a.confidence:"—"}</span></div>
          <div><span class="k">Screen changed</span><span class="v">${s.screen_changed==null?"—":s.screen_changed}</span></div>
          <div><span class="k">Decision</span><span class="v">${s.decision_seconds!=null?s.decision_seconds+"s":"—"}</span></div>
          ${a.reason?`<div><span class="k">Reason</span><span class="v">${esc(a.reason)}</span></div>`:''}
          ${s.error?`<div><span class="k warn">Error</span><span class="v warn">${esc(s.error)}</span></div>`:''}
        </div>
        <h2>Reasoning / event stream</h2>
        <div class="events">${evHtml}</div>
      </div>
    </div>
    ${renderFooter()}`;
}

function renderFooter(){
  let html="";
  const subs=R.subgoals||[];
  if(subs.length){
    html+='<h2>Subgoals</h2><table><thead><tr><th>#</th><th>Milestone</th><th>Status</th><th>Evidence</th></tr></thead><tbody>'
      +subs.map(s=>{const c=s.status==="passed"?"b-pass":s.status==="failed"?"b-fail":"b-other";
        return `<tr><td>${esc(s.number)}</td><td>${esc(s.description)}</td>
          <td><span class="badge ${c}">${esc((s.status||"").toUpperCase())}</span></td>
          <td>${esc(s.evidence)}</td></tr>`;}).join("")+'</tbody></table>';
  }
  const findings=((R.scene_graph||{}).pending_findings)||[];
  if(findings.length){
    html+='<h2 class="warn">HMI divergences pending review</h2><table><thead><tr><th>ID</th><th>Kind</th><th>Detail</th></tr></thead><tbody>'
      +findings.map(f=>`<tr><td>${esc(f.id)}</td><td>${esc(f.kind)}</td><td>${esc(f.detail)}</td></tr>`).join("")
      +'</tbody></table>';
  }
  const crashes=R.crashes||[];
  if(crashes.length){
    html+='<h2 class="warn">Crashes / ANRs</h2><table><thead><tr><th>Kind</th><th>Package</th><th>Summary</th><th>Line</th></tr></thead><tbody>'
      +crashes.map(c=>`<tr><td>${esc(c.kind)}</td><td>${esc(c.package)}</td><td>${esc(c.summary)}</td><td>logcat:${esc(c.line)}</td></tr>`).join("")
      +'</tbody></table>';
  }
  return html;
}

document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")select(current-1);
  if(e.key==="ArrowRight")select(current+1);
});
renderHeader();renderList();select(0);
</script>
</body></html>
"""
