# Deep-dive: making this a *true* Android IVI validation agent

**Status: research / design reference.** A technical study of what separates a UI
automation demo from a production **Android IVI validation agent**, an honest gap
analysis against this repo, and a pillar-by-pillar plan to close the gap. Grounded
in Android/AAOS engineering and automotive QA practice; sources at the end.

---

## 1. What "true validation agent" means

A UI-automation agent answers *"can I reach and tap this control?"* A **validation
agent** answers *"does the HMI behave per specification, and can I prove it?"* —
which requires four things a tapping agent doesn't have:

1. **Oracles beyond pixels.** The truth is the vehicle state (VHAL/CAN), the
   system state (logs, services), and the spec (requirements/CCF) — not just what
   the screen draws. "UI says massage ON, VHAL says OFF" is the defect a pure UI
   agent can never see.
2. **Requirements traceability.** Every check maps to a requirement / test case,
   so results become **coverage** an engineering org can act on (ASPICE/ISO 26262
   process context).
3. **Determinism & evidence.** Reproducible runs, flake control, and an audit
   trail (screens + signals + logs + reasoning) an engineer can open at "step 17
   failed."
4. **Automotive-specific correctness.** Driver-distraction rules, multi-display /
   cluster, rotary/voice input, projection, boot KPIs, persistence across ignition
   cycles — dimensions that don't exist on a phone.

This repo today is a strong **agent core** (local VLM planner/grounder/verifier,
a11y→CV→model grounding, manual RAG, a living scene graph, run trace). To become
a *validation* agent it needs the oracle, traceability, determinism, scale, and
automotive-domain layers below.

---

## 2. Why IVI validation ≠ mobile UI automation (the hard problems)

- **No single "screen."** AAOS is **multi-display**: center stack, instrument
  **cluster**, passenger display, HUD. Tests must target and correlate several
  displays at once. (`screencap -d`, `SurfaceFlinger --display-id`, per-display
  input.)
- **Non-standard surfaces.** OEM HMIs are often **Unity/Qt/Flutter/Canvas** with
  **no accessibility tree** — pixel-only. Some cockpits run **QNX** for the cluster
  and Android for IVI on the same head unit.
- **Non-touch input.** **Rotary controllers, steering-wheel buttons, physical
  knobs, voice (VPA)** are primary inputs in many vehicles — not just touch.
- **Driver-distraction lockouts.** AAOS enforces **CarUxRestrictions**: activities
  must be "distraction optimized" or they're **blocked while moving**; the driving
  state (Parked / Idling / Moving) changes what the HMI is *allowed* to show. A
  validation agent must **simulate driving state** and verify the restriction
  behavior — and the rules vary by OEM and region. [AOSP driver distraction]
- **State that isn't on screen.** Climate/seat/drive-mode changes drive real
  **actuators**; correctness lives in **VHAL/CAN**, not the toast.
- **Time & lifecycle dimensions.** Boot/KPI timing, thermal throttling, **ignition
  cycles / persistence**, OTA, connectivity (BT, **Android Auto / CarPlay**
  projection). None are typical phone tests.
- **Variant explosion.** **CCF (Car Configuration File)** / variant coding means
  the *same build* exposes different features per vehicle — the HMI must match its
  coding, and that's a per-variant validation matrix.

---

## 3. Honest capability inventory (this repo, today)

| Layer | Have | Gap toward validation |
| --- | --- | --- |
| Perception | screenshot, uiautomator XML, OCR, perceptual hash, numbered grid | multi-display correlation; cluster; QNX; robust custom-surface parsing |
| Grounding | a11y → CV(template) → VLM point/grid | rotary/voice/hard-key input; scale/rotation-robust CV; on-device speed |
| Decision | local VLM plan/verify, subgoals, cue advancement, loop/dead-action memory | requirement-driven test plans; recovery hooks |
| Knowledge | manual RAG (semantic), **living scene graph** (defect findings) | requirements traceability; CCF coverage; per-variant models |
| **Oracles** | **visual only** | **VHAL/CAN/CCF/logs/UXR** — the biggest gap |
| Evidence | run trace (events.jsonl/agent.log), HTML report, grounding+wall metrics | signals timeline, replay viewer, requirement coverage report |
| Scale | single-device suite runner, AndroidWorld bridge | device farm, parallelism, CI, HIL/bench |
| Safety | protected regions, text-input guard, bench-only warnings | driving-state simulation, actuator interlocks, allowlisted commands |

The single highest-value gap is **oracles**: without VHAL/CAN/CCF/log truth, it
validates *appearance*, not *behavior*.

---

## 4. The eight pillars

### Pillar A — Perception across every surface
- **Structured first:** AAOS/Android **accessibility tree** (fast, exact) and
  `uiautomator`. Prefer these when present.
- **Pixel fallback:** OCR + **CV** (template + feature) + the **VLM** for
  Unity/Qt/Flutter/Canvas surfaces with no tree.
- **Multi-display:** capture and reason over center stack **and cluster** (and
  passenger/HUD), keyed by physical display id; correlate state across them (e.g.
  a telltale on the cluster after a center-stack action).
- **QNX cluster:** where the cluster is QNX, ingest its framebuffer/screenshot via
  the platform's channel; treat as a pixel-only display in the same pipeline.
- Robustness: day/night themes, localization (RTL, CJK), DPI/scale, animations.

### Pillar B — Action / execution across every input
- Touch: adb `input` today; **on-device AccessibilityService `dispatchGesture`**
  (API 24+) for low latency and native gestures. (See the on-device-driver design.)
- **Rotary controller & steering-wheel buttons:** inject the corresponding
  keyevents/rotary events (AAOS rotary framework); validate focus traversal.
- **Hard keys / knobs:** keyevent injection or HIL signal.
- **Voice (VPA):** drive the assistant (intent/utterance injection or audio) and
  validate the HMI + action result.
- A **`Device` driver abstraction** so adb, on-device service, and HIL/bench
  channels are interchangeable behind one interface.

### Pillar C — Ground-truth oracles (the moat)
This is what makes it *validation*.
- **VHAL via CarPropertyManager / `dumpsys car_service`:** read/inject vehicle
  properties (e.g. `HVAC_*`, seat, drive mode, `PERF_VEHICLE_SPEED`,
  gear/ignition). Assert **UI ⇔ signal** agreement; flag divergence as a defect.
  [AOSP CarPropertyManager / VHAL]
- **CAN / CAN-FD:** where available (HIL/rest-bus simulation), the deeper truth
  behind VHAL.
- **CCF / variant coding:** read the coding, then check **feature coverage** —
  every coded-present feature must be reachable in the HMI; coded-absent features
  must not appear. (Design already sketched in probes-commands-and-mcp.)
- **System logs:** `logcat` (crashes/ANRs/exceptions during a step), `dumpsys`
  (activity, window, car watchdog), **Car Watchdog** for health/kills.
- **Driving state / UXR:** inject driving state and assert **CarUxRestrictions**
  behavior (blocked-while-moving, no-video, no-config-screen). [AOSP UXR]
- **Timing/KPI:** boot-to-HMI, screen-transition latency, first-frame — measured,
  not eyeballed.

All oracles must be **operator-declared and allowlisted** (commands in config,
never in a shareable manual), read-only by default, bounded, and audited.

### Pillar D — World model, knowledge & traceability
- **Manual RAG** (have) → keep as the recognition/knowledge source.
- **Living scene graph** (have) → the observed vs expected HMI model; extend nodes
  with **linked VHAL signals** and CCF features so a screen "knows" its ground
  truth.
- **Requirements traceability:** a requirement/test-case id on each check, so the
  report is **coverage** (which reqs verified/failed/untested) — the artifact an
  ASPICE/ISO 26262 process consumes. Import from ALM (Polarion/DOORS/Jira) later.

### Pillar E — Orchestration, reliability & determinism
- **Two profiles:** `fast` (reactive a11y→CV→model) and `engineer` (requirement
  decomposition → plan → state verification → diagnostics → report).
- **Flake control:** explicit settle/stability, idempotent steps, bounded retries,
  and a rule that **"flake" is never a root cause** — a failing assertion is a
  finding until proven environmental.
- **Determinism:** pinned local model, fixed seeds/temperature 0, recorded inputs;
  **graph-path replay** turns a confirmed route into a deterministic script (fast,
  repeatable) with the agent as fallback on divergence.
- **Recovery:** declared recovery hooks (home, relaunch, dump logs) on a stall.

### Pillar F — Models & latency budget
- **Local-first** (privacy/IP): the moat. Grounding-capable VLM (qwen3-vl) for
  point grounding; grid fallback for others.
- **Latency:** the a11y→CV→model ladder already drops most taps to sub-second; the
  **on-device driver** removes the adb I/O that now dominates wall-clock.
- **On-device small models:** Gemma/LiteRT (MediaPipe LLM Inference) for residual
  decisions where a tiny model suffices; quantized (int4), mmapped weights.
- **Backend abstraction:** optional OpenAI-compatible/OpenRouter backend for
  benchmarking vs cloud SOTA (local stays default).

### Pillar G — Scale, CI & bench integration
- **Device farm:** one orchestrator driving many emulators + physical benches
  (OEM/OS matrix) running one regression suite in parallel.
- **CI hooks:** run on build, gate merges, publish coverage + defect diffs.
- **HIL / bench:** integrate rest-bus simulation and real ECUs; drive driving-state
  and CAN alongside the HMI.
- **Existing AOSP tooling to interop with, not reinvent:** **Spectatio** (AOSP
  automotive test framework) and the Automotive Snippet Library for
  service/app-level assertions — the agent complements these with autonomous,
  manual-driven exploration. [AOSP Spectatio]

### Pillar H — Safety & security
- **Bench-only** by default; explicit gating before anything on a moving/production
  vehicle; actuator interlocks; protected regions (have).
- **Driving-state awareness:** never issue distraction-unsafe actions in a Moving
  state unless the test *is* the restriction check.
- **Security:** allowlisted commands only; no model- or manual-supplied shell;
  least privilege; the tester (brain) stays **outside** the system under test.

### Pillar I — Evidence, metrics & benchmarking
- **Per-run artifact:** screens + `events.jsonl` + **`vehicle_signals.jsonl`** +
  logs + report; a **replay viewer** ("step 17 failed → see UI + VHAL divergence").
- **Metrics that matter:** task success rate, **defect precision/recall** (are the
  flagged divergences real?), requirement coverage, wall-clock, grounding mix
  (a11y/CV/model), flake rate.
- **IVIWorld benchmark:** ~100 automotive tasks × {stock AAOS, Canvas, Unity,
  WebView, OEM} × driving states — *our* benchmark; AndroidWorld becomes a sanity
  check and general agents (e.g. ARTEMIS) become baselines.

---

## 5. IVI-specific validation checklist (dimensions to cover)

- Driver distraction / **CarUxRestrictions** across Parked/Idling/Moving, per
  region/OEM.
- **Multi-display / cluster** correlation; telltales; HUD.
- **Rotary / steering-wheel / hard-key** navigation and focus.
- **Voice / VPA** flows and their HMI + action results.
- **Projection**: Android Auto / Apple CarPlay handoff and HMI.
- **Boot / KPI** timing; cold vs warm; ignition-cycle **persistence** of settings.
- **Thermal** throttling behavior; **connectivity** (BT pairing, Wi-Fi, cellular).
- **OTA / variant (CCF)** matrix; **localization** (languages, RTL, units);
  **day/night** and theme; accessibility settings.
- Actuator domains: **HVAC, seats, ambient light, drive modes, charging/energy,
  camera/surround view**.

---

## 6. Standards & process context (why traceability matters)

The agent is a **test tool** (not itself a safety item), but it must produce
evidence the automotive process consumes:
- **Automotive SPICE (ASPICE):** test cases ↔ requirements ↔ results traceability.
- **ISO 26262:** functional-safety process; HMI for safety-relevant functions
  (telltales, warnings) needs rigorous, traceable verification.
- **HMI ergonomics / distraction:** ISO 15005 / 17287, **NHTSA driver-distraction
  guidelines**, UNECE (e.g. R79/R10 context), regional variants (e.g. GB/T in
  China) — the *rules* the UXR checks encode.

Designing outputs as **traceable coverage** (not just pass/fail logs) is what
makes this land with an OEM/Tier-1 QA org.

---

## 7. Target architecture (blueprint)

```
                        REQUIREMENTS / TEST SPEC / CCF
                                    │
                              TEST ORCHESTRATOR
                       fast profile   |   engineer profile
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                            │
   PERCEPTION                  DECISION/GRODING                 ORACLES
  a11y · uiauto · OCR      a11y → CV → VLM (+ graph        VHAL/CarProperty ·
  CV · VLM · multi-        replay) · plan/verify           CAN · CCF · logcat ·
  display · cluster              │                          CarWatchdog · UXR/
        │                        │                          driving-state · KPI
        └───────────┬────────────┴───────────┬──────────────────┘
                    ▼                         ▼
             DEVICE EXECUTOR            WORLD MODEL
       adb · on-device a11y svc ·   scene graph (+VHAL/CCF links) ·
       rotary/keys · voice · HIL    manual RAG · requirement trace
                    │
                    ▼
        EVIDENCE: screens · events.jsonl · vehicle_signals.jsonl ·
        logs · report · replay viewer · requirement COVERAGE
```
The brain stays **off the system under test**; only a thin driver may live
on-device.

---

## 8. Maturity roadmap (L0 → L4)

- **L0 — UI automator (done):** reach & tap from a goal; a11y→CV→model; scene
  graph; trace. *This repo.*
- **L1 — State-aware validator:** VHAL/CarProperty oracle + UI⇔signal divergence;
  logcat/ANR capture; `vehicle_signals.jsonl` + report. *(probes design, Pillar C.)*
- **L2 — Spec-driven & deterministic:** requirement traceability + coverage;
  engineer profile; graph-path replay; recovery hooks; CCF feature coverage.
- **L3 — Full-cockpit & inputs:** multi-display/cluster correlation; rotary/voice/
  hard-key; driving-state/UXR checks; projection; boot/KPI; on-device driver for
  speed.
- **L4 — Scale & standards:** device farm + CI + HIL/bench; ALM integration;
  ASPICE/26262-shaped coverage reports; IVIWorld benchmark published.

Recommended near-term order: **L1 oracle (probes/VHAL)** → **on-device driver
(speed)** → **L2 traceability + engineer mode** → **cluster/inputs**.

---

## 9. Risks & open questions

- **Access on real head units:** OEM units are often ADB-locked / no sideloading →
  the on-device a11y app may be impossible; adb or a vendor test channel is the
  fallback. Confirm per-target.
- **CAN/HIL availability:** deepest oracle needs bench hardware; VHAL is the
  Android-side proxy where CAN isn't reachable.
- **CCF/variant source of truth:** where does coding come from per unit, and is it
  readable non-intrusively?
- **Custom-surface perception limits:** Unity/QNX with no tree lean entirely on
  CV/VLM — accuracy vs speed trade-off; may need per-OEM icon crops / templates.
- **Small-model ceiling:** on-device models handle grounded taps well but weak on
  ambiguous reasoning; hybrid (host brain) mitigates.
- **Determinism vs LLM variance:** temperature 0 + replay + strict oracles reduce
  it; never let "flake" mask a real defect.

## 10. Success criteria (how we'd know it's "true")

1. It flags a **display/actuator divergence** (UI vs VHAL) automatically.
2. It reports **requirement coverage**, not just pass/fail.
3. It validates **driving-state/UXR** behavior across Parked/Idling/Moving.
4. Runs are **reproducible** and produce an engineer-openable **evidence trail**
   (screens + signals + logs).
5. It runs **unattended in CI across a device matrix**, and beats a human on
   wall-clock for rote flows while catching defects a human tester misses.

---

## Sources

- [Create an accessibility service — Android Developers](https://developer.android.com/guide/topics/ui/accessibility/service)
- [AccessibilityService.TakeScreenshotCallback — Android Developers](https://developer.android.com/reference/kotlin/android/accessibilityservice/AccessibilityService.TakeScreenshotCallback)
- [Spectatio: Automotive test framework — AOSP](https://source.android.com/docs/automotive/tools/spectatio)
- [Vehicle HAL / CarPropertyManager — AOSP & guides](https://source.android.com/docs/automotive/vhal)
- [Consume car driving state and UX restrictions — AOSP](https://source.android.com/docs/automotive/driver_distraction/consume)
- [Car User Experience Restrictions rules — AOSP](https://source.android.com/docs/automotive/driver_distraction/car_uxr)
- [Driver Distraction Guidelines — AOSP](https://source.android.com/docs/automotive/driver_distraction/guidelines)
- [CarUxRestrictions — Android Developers](https://developer.android.com/reference/android/car/drivingstate/CarUxRestrictions)

See also, in this repo: [future-upgrades/](../future-upgrades/README.md),
[future-upgrades/probes-commands-and-mcp.md](../future-upgrades/probes-commands-and-mcp.md),
[future-upgrades/on-device-driver.md](../future-upgrades/on-device-driver.md),
[creating-a-manual.md](../creating-a-manual.md).
