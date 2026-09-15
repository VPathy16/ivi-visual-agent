# IVI Visual Agent

A local, goal-driven visual agent that tries Android IVI tasks like a careful first-time
user. It observes the current screen, asks a locally hosted vision model for one safe
action, executes that action through ADB, and records evidence after every step.

This is an early bench-testing MVP. Do not operate it in a moving vehicle or connect it
to a production vehicle without an appropriate safety review.

## What it does

- Accepts a plain-language goal such as `Open Bluetooth settings`.
- Captures the IVI display through ADB.
- Includes the Android UI hierarchy when it is available.
- Uses a local Ollama vision model to select one action at a time.
- Selects numbered accessibility/OCR candidates instead of allowing arbitrary taps.
- Uses a fast text-only plan when Android exposes usable UI controls.
- Resizes screenshots before the slower visual fallback.
- Converts swipes into bounded direction/region gestures.
- Restricts the model to grounded taps, gestures, Back, Home, wait, optional text, or finish.
- Requires an independent visual verification before reporting success.
- Writes screenshots, JSON results, and an HTML report under `runs/`.
- Launches `scrcpy` for live observation or session recording.

## Prerequisites

- Python 3.11 or newer
- Android platform-tools (`adb`)
- `scrcpy`
- Ollama
- Tesseract OCR
- An Ollama vision model, initially configured as `qwen3.5:4b`
- An Android IVI or emulator with ADB enabled and authorized

On macOS with Homebrew, install the missing device tools:

```bash
brew install android-platform-tools scrcpy tesseract
```

Pull and start the local model:

```bash
ollama pull qwen3.5:4b
ollama serve
```

The planner runs with hidden model reasoning disabled for lower latency. You can change
the model in `config.json` after benchmarking another vision-capable Ollama model.

## Set up the project

```bash
cd ivi-visual-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp config.example.json config.json
ivi-agent --config config.json doctor
```

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

Authorize and check the device:

```bash
adb devices
adb shell wm size
```

## First safe run

Start the live display in one terminal:

```bash
ivi-agent scrcpy --record session.mp4
```

Ask the model to propose one action without executing it:

```bash
ivi-agent --config config.json run \
  --goal "Open the Bluetooth settings screen" \
  --dry-run
```

Inspect the generated report. When the proposal and safety configuration look right,
run the closed loop:

```bash
ivi-agent --config config.json run \
  --goal "Open the Bluetooth settings screen"
```

If several devices are attached, add `--serial DEVICE_SERIAL`. For an IVI with several
Android displays, add `--display-id DISPLAY_ID` after verifying the correct ID for the
platform.

## Safety configuration

Coordinates in `protected_regions` are normalized rectangles in the form
`[left, top, right, bottom]`. This example prevents taps in the upper-right corner:

```json
{
  "protected_regions": [[0.8, 0.0, 1.0, 0.2]]
}
```

Text input is disabled by default. The model prompt also forbids data deletion, calls,
purchases, factory reset, software updates, and surprising permission grants. These
prompt restrictions supplement the controller policy; safety-critical restrictions
should also be encoded in controller-side policy before broader use.

## Current limitations

- Icon-only controls on custom-rendered surfaces still depend on the visual model.
- Small local models can make poor navigation decisions even when output is grounded;
  confidence thresholds and loop detection stop rather than guess.
- ADB screenshot support for secondary IVI displays varies by Android build.
- Bluetooth pairing needs a discoverable test phone and may require a human or a second
  automation channel to confirm the PIN on that phone.
- Visual success is evidence, not proof of underlying service state. Later versions
  should cross-check system signals such as Bluetooth connection state.
