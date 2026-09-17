# Future upgrades — roadmap & ideas

A running backlog of ideas for the IVI Visual Agent, captured from design
discussions. **Reference only — not yet implemented** unless a row says otherwise.
Detailed designs link out to their own file in this folder.

North star: *don't build a better phone agent — build the AI validation engineer
for the whole cockpit.* Keep it **local/offline** by default; that's the moat a
general phone-automation tool can't follow.

## Priority backlog

| # | Idea | Why | Effort | Detail |
| --- | --- | --- | --- | --- |
| 1 | **Vehicle-state probes, CCF divergence, recovery, MCP** | The moat: "UI says running but VHAL says off → defect"; CCF feature-coverage; recover when stuck; drive from an IDE | L (phased) | [probes-commands-and-mcp.md](probes-commands-and-mcp.md) |
| 2 | **On-device AccessibilityService driver (hybrid)** | Kills adb overhead (the current wall-clock bottleneck); native taps + instant a11y | M–L | [on-device-driver.md](on-device-driver.md) |
| 3 | **Graph-path replay (learned routes)** | Once the scene graph confirms a route, replay it deterministically — faster than a human, no model | M | §Graph-path replay below |
| 4 | **Fast / Engineer execution profiles** | `--mode fast` (reactive) vs `--mode engineer` (plan + verify + diagnostics + report) | S–M | §Execution profiles below |
| 5 | **Carry-forward observation** | Reuse the after-state as the next step's observation → removes the duplicate `uiautomator dump` per step | S | §Speed quick wins below |
| 6 | **History compression** | Fold old screenshots into text summaries → smaller context, lower latency, longer horizons | M | §Latency & context below |
| 7 | **IVIWorld benchmark** | Our own benchmark: 100 automotive tasks × {stock AAOS, Canvas, Unity, WebView, OEM} — Artemis becomes the baseline | L | §IVIWorld below |
| 8 | **Cross-platform executor (AAOS + QNX)** | The name is "IVI Visual Agent", not "Android agent" — abstract the executor beyond Android | L | §Cross-platform below |
| 9 | **Model-backend abstraction (OpenRouter / OpenAI-compatible)** | A/B a cloud VLM vs local qwen3-vl on the same runs; local stays default | M | §Model backends below |
| 10 | **Device laboratory / test farm** | One agent server driving rows of benches/emulators through the same regression suite | L | §Device lab below |
| 11 | **Trace replay viewer** | Turn `events.jsonl` + screenshots into a clickable step-by-step failure viewer | M | §Trace replay below |
| 12 | **CV robustness (color match, ORB screen match)** | Better icon matching on low-contrast/scaled icons; "am I on this screen" corroboration | S | §CV robustness below |

Rough sequencing: **2 (on-device driver) or 1 (probes) next** — 2 attacks the
speed bottleneck, 1 opens the validation moat. 3–6 are cheap speed/quality wins.
7–8 are the long-horizon strategic bets.

---

## Details for ideas without their own file

### Graph-path replay (learned routes)
Once the living scene graph has **confirmed** a route (e.g. Home→Seat→program→Start),
replay it as a deterministic a11y/CV tap sequence with **no planning and no
verify**, falling back to the agent only when the live screen diverges from the
graph. First run self-learns the path; later runs execute at pure tap speed —
below a human, because there is no thinking. Pairs with the scene graph already
built (`graph.py`).

### Execution profiles (fast / engineer)
- `--mode fast`: the reactive loop (a11y → cv → model), minimal verify — for
  routine regression.
- `--mode engineer`: requirement decomposition → persistent plan → manual
  retrieval → state verification (probes) → diagnostics → written report — for
  deep validation. Most pieces exist; this formalizes two presets over them.

### Speed quick wins
- **Carry-forward observation**: today each step captures a screenshot + `uiautomator
  dump` at start *and* after the action (same screen). Reuse the after-state as the
  next step's observation to remove one full dump (~1–2s) and one screenshot per
  step. Loop refactor in `agent.run()`; needs on-device testing.
- Lazy `ui_dump` when neither a11y nor progress/cue detection needs it.
- Smaller model (`qwen3-vl:2b`) for residual model steps.

### Latency & context — history compression
Replace older screenshots in history with short text summaries and chunk
completed steps into recallable "eras" (Artemis-style). Smaller KV cache → faster
model steps and support for 100+ step long-horizon runs.

### IVIWorld benchmark
A first-class automotive benchmark: ~100 tasks across Climate, Seat, Media,
Bluetooth, Navigation, Profiles, ADAS settings, Ambient lighting, Vehicle
settings, Camera, Charging, Energy, Notifications, Voice, Multi-display — run
against stock AAOS, custom Canvas UI, Unity IVI, WebView IVI, OEM-like
proprietary UI, and eventually real benches. Report per-surface pass rates;
publish local-vs-cloud. Makes AndroidWorld a sanity check and Artemis a baseline.

### Cross-platform executor (AAOS + QNX)
Abstract the `Device`/executor so the same brain drives non-Android cockpits.
Android/AAOS via adb or the on-device driver; QNX via its own channel (screen
capture + injected input) where permitted. Keeps perception/grounding/graph/trace
unchanged behind the executor interface.

### Model backends
Add an `OpenAICompatibleVisionModel` (OpenRouter, OpenAI, vLLM, LM Studio,
llama.cpp) behind the existing model interface, selected by config
(`model_backend`, `api_base`, key from env). Local Ollama stays the default;
cloud is opt-in for benchmarking / when privacy allows. `GoalAgent` already
depends on the model interface, not Ollama, so this is additive.

### Device laboratory / test farm
An agent server orchestrating many targets (emulators + physical benches of
different OEMs/OS versions) running one shared regression suite in parallel — an
automotive AI test farm. Builds on the suite runner + per-run trace.

### Trace replay viewer
A small viewer that reads a run's `events.jsonl` + screenshots (+ future
`vehicle_signals.jsonl`) and renders a clickable timeline: click "Step 17 failed"
→ see the before/after screenshots, the action, the verify verdict, and (future)
the VHAL divergence. Turns the trace into an engineer-facing artifact.

### CV robustness
- Match icon templates in **color** as well as grayscale (helps colored UI icons).
- Use `screen_similarity` (ORB) as a corroborating "am I on the expected screen"
  signal for the scene graph.
- Auto-tune `cv_match_threshold` per profile from observed match scores.

---

## Done (shipped, for context)
These started as ideas and are now implemented on the main branch:
- Local semantic RAG (fastembed, no Ollama model needed)
- OpenCV icon fast-path + grounding telemetry
- Accessibility fast-path (a11y → cv → model)
- Living scene graph (seed from manual, grow, flag HMI defects)
- Subgoal success-cue advancement + completion
- Structured run trace (events.jsonl / agent.log / task.json / plan.json)
- Multi-display capture + display-id mapping
- Speed: settle tuning, optional per-step verify, wall-clock timing
