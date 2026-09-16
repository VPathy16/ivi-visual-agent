# AndroidWorld integration

Run the local IVI visual agent on the
[AndroidWorld](https://github.com/google-research/android_world) benchmark (116
programmatic tasks across 20 real apps, deterministic reward).

The IVI agent keeps its own planner / grounder / verifier; this integration
only adapts it to AndroidWorld's environment interface and scoring.

## Requirements

AndroidWorld drives a **real Android emulator**, so it cannot run in a container
without nested virtualization. You need a machine with hardware acceleration
(`/dev/kvm` on Linux, or an Apple Silicon Mac):

1. Android SDK + a running emulator on the AndroidWorld-supported system image
   (see AndroidWorld's README; it expects an AVD named `AndroidWorldAvd`).
2. `pip install git+https://github.com/google-research/android_world.git`
   (installs `android_world` and its dependencies, including `numpy`).
3. This package installed: `pip install -e .`
4. Ollama running a grounding-capable model:
   `ollama pull qwen3-vl:8b-instruct` (needs Ollama ≥ 0.12.7).
5. A config with `grounding_mode: "point"` and the Qwen3-VL model — the shipped
   `config.example.json` already does this: `cp config.example.json config.json`.

## Run

```bash
# One specific task
python -m ivi_agent.integrations.androidworld.run_benchmark \
    --config config.json --console-port 5554 \
    --task ContactsAddContact

# A random sample of N tasks (smoke run)
python -m ivi_agent.integrations.androidworld.run_benchmark \
    --config config.json --console-port 5554 --n-tasks 20 --seed 0
```

The runner prints a per-task PASS/FAIL and a JSON summary with the success rate.
Add `--emulator-setup` on the first run to let AndroidWorld install the apps its
tasks need.

## How it works

| Piece | File | Responsibility |
| --- | --- | --- |
| `IviVisualAgent` | `agent.py` | Subclasses `EnvironmentInteractingAgent`; runs the IVI decision loop one `step()` at a time, keeping plan / history / loop-detection state across steps, and signals `done` (AndroidWorld computes the reward). |
| `bridge` | `bridge.py` | Pure translation: AndroidWorld screenshot → PNG, AndroidWorld UI elements → uiautomator-style XML (so the existing perception layer works unchanged), and IVI `Action` → `JSONAction` kwargs. |
| `run_benchmark` | `run_benchmark.py` | Launches the env, iterates the task registry, runs the agent, reports results. |

`bridge` has no `android_world` dependency and is unit-tested in
`tests/test_androidworld_bridge.py`; the `IviVisualAgent` adapter is imported
lazily so those tests run without the benchmark installed.

## Notes and caveats

- The agent's own visual verifier is used only to decide when to stop; scoring is
  always AndroidWorld's `task.is_successful`.
- A small local model will score low. Use `qwen3-vl:8b-instruct` or larger and
  `grounding_mode: "point"` for a meaningful number.
- This adapter is verified for import, construction, and translation logic, and
  has been run end-to-end on an Android 13 (arm64) emulator with
  `qwen3-vl:8b-instruct`, passing built-in-app tasks (Wi-Fi on/off, contacts,
  clock). Expect to tune `transition_pause`, `max_actions`, and confidence
  thresholds for your device/model.
- **Sliders/brightness are a known limitation.** AndroidWorld's `JSONAction`
  exposes no precise coordinate drag (its `swipe`/`scroll` are coarse and
  directional), so exactly setting a seek bar to max/min is unreliable through
  the bridge. The planner can emit `swipe`, but it maps to a whole-screen
  directional swipe, not a thumb drag.
- **Third-party task apps do not install/run on arm64.** For the full 116-task
  suite use an x86_64 emulator on a Linux/KVM host; the built-in-app tasks
  (Settings/Contacts/Clock) run fine on Apple Silicon arm64.
