# Design: on-device AccessibilityService driver (hybrid)

**Status: design (not implemented).** An on-device Android app that acts as the
agent's I/O layer — reading the UI and performing taps in-process — while the
"brain" (model, RAG, scene graph, trace) stays on the host. Goal: remove the adb
overhead that currently dominates wall-clock.

## Why
In a fully a11y/CV-grounded run, decision time is ~0.1s but wall-clock is ~15s —
almost all of it **adb round-trips**: `uiautomator dump` (~1–2s/step) and
`screencap` over adb, plus `adb input` taps. An in-process **AccessibilityService**
reads the tree and performs gestures with no bridge, so those seconds largely
disappear.

## Architecture: hybrid (brain off-device)
```
        HOST                                   DEVICE / HEAD UNIT
  ┌───────────────┐   adb forward / TCP   ┌──────────────────────────┐
  │  GoalAgent    │◄─────────────────────►│  Agent Driver App        │
  │  model + RAG  │   GET /tree           │  (AccessibilityService)  │
  │  scene graph  │   GET /screenshot     │  - reads a11y tree       │
  │  trace + CV   │   POST /tap|swipe|text│  - takeScreenshot()      │
  └───────────────┘                       │  - dispatchGesture()     │
                                          └──────────────────────────┘
```
The host keeps everything it has; only the **executor** changes from `AdbDevice`
to an `OnDeviceServiceDevice` that calls the app over localhost (via
`adb forward tcp:PORT tcp:PORT`) or the network.

**Why brain-off-device (validation methodology):** the tester shouldn't be part
of the system under test. Keeping the model/graph on the host means the agent
doesn't consume the IVI's CPU/RAM, can still observe boot/crash/ANR states where
an on-device app would be down, and isn't something you also have to validate.
The on-device piece is a thin driver, not the brain.

## Two build pieces

### Piece 1 — Python `Device` driver abstraction (host; testable without a device)
Refactor so `AdbDevice` implements a `Device` interface:
`ensure_ready`, `screen_size`, `capture(path)`, `ui_dump()`, `execute(action, size)`,
`wait_until_stable(...)`. Add `OnDeviceServiceDevice(base_url)` implementing the
same interface over HTTP. `GoalAgent`, `suite`, and the CLI select the driver by
config (`device_driver: "adb" | "onservice"`). Everything else is unchanged.

### Piece 2 — Android AccessibilityService app (device; you test on the emulator)
A small Kotlin/Java app exposing a local HTTP server:
- `GET /tree` → the accessibility node tree as uiautomator-style XML (so the host's
  existing perception parses it unchanged).
- `GET /screenshot` → PNG (`takeScreenshot`, API 30+; MediaProjection fallback).
- `POST /tap {x,y}` / `POST /swipe {...}` / `POST /text {...}` → `dispatchGesture`
  / input.
- Started/enabled once via Settings; host connects with `adb forward`.

## Constraints & risks
- **Can't run/verify in the cloud dev container** (no emulator/KVM). The Android
  app is written on the host and tested on your device — expect an iterate-on-device
  loop like the WebView app.
- **AccessibilityService permission** must be user-enabled; screenshot API has
  version quirks.
- **Real OEM head units may block installing a privileged accessibility app** —
  keep the **adb driver as the fallback** (the driver abstraction makes this free).
- Custom/Unity/Canvas screens with no a11y tree still need the screenshot +
  CV/model path — the app provides the screenshot, the host does CV.

## Suggested order
1. Piece 1 (driver interface + `OnDeviceServiceDevice`) — pure Python, unit-tested
   on the host; adb stays the default.
2. Piece 2 (the app) — build, enable, and validate on the emulator; point the host
   at it via `device_driver: "onservice"` and re-measure wall-clock.

Expected payoff: the ~15s fully-grounded run drops toward a few seconds, since the
per-step adb dumps/taps are replaced by in-process calls.
