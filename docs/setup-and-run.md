# Setup & run — macOS, Linux, Windows

Step-by-step for each OS: **start** (install), **run the manual** (build → index →
seed graph), and **run** the agent against an **emulator** and a **direct
(physical) device / head unit**.

Common to all platforms you need: **Python 3.12+**, **Ollama** with a vision
model pulled, the **Android platform-tools** (`adb`) and **emulator** (only for
the emulator path), and **Poppler** (`pdftoppm`, only to index a manual PDF).

Pull the vision model once (any OS, after Ollama is installed):

```bash
ollama pull qwen3-vl:8b-instruct
```

A minimal `config.json` (same on every OS — see
[creating-a-manual.md](creating-a-manual.md) and the README for all keys):

```json
{
  "model": "qwen3-vl:8b-instruct",
  "grounding_mode": "point",
  "knowledge_profile": "benz",
  "use_embeddings": true,
  "embedding_backend": "fastembed",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "cv_fast_path": true,
  "scene_graph": true,
  "trace": true
}
```

> **numpy note (any OS):** if `android_world` is installed in the same
> environment, keep `numpy<2` and `opencv-python-headless<5` (the `[cv]` extra
> already pins this). Use a dedicated venv for IVI work to avoid the conflict.

---

## macOS

### Start
```bash
brew install ollama poppler
brew install --cask android-platform-tools        # adb
# Android emulator + SDK: install via Android Studio, or `brew install --cask android-commandlinetools`
ollama serve &                                     # if not already running
ollama pull qwen3-vl:8b-instruct

cd ~/ivi-visual-agent
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[docs,embeddings,cv]'
```
Put the SDK tools on PATH (adjust root to your install):
```bash
export ANDROID_SDK_ROOT="$HOME/Library/Android/sdk"   # or /opt/homebrew/share/android-commandlinetools
export PATH="$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"
```

### Run the manual
```bash
ivi-agent manual build --source examples/benz-mbux-manual --output output/pdf/benz.pdf
ivi-agent --config config.json knowledge index output/pdf/benz.pdf --profile benz
ivi-agent graph build --profile benz
```

### Run on an emulator
```bash
emulator -list-avds
emulator -avd <AVD_NAME> -no-audio &
adb wait-for-device && adb devices
adb install -r ~/Downloads/ivi-sample.apk
adb shell am start -n com.example.iviwv/.MainActivity
ivi-agent --config config.json run --serial emulator-5554 --knowledge-profile benz --goal "Open the climate screen"
```

### Run on a direct (physical) device / head unit
1. On the device: enable **Developer options → USB debugging**, connect USB, accept the RSA prompt.
2. Verify and run:
```bash
adb devices                                        # note the serial
adb -s <SERIAL> shell dumpsys SurfaceFlinger --display-id   # find DISPLAY_ID if multi-display
ivi-agent --config config.json run --serial <SERIAL> --knowledge-profile benz --goal "Open the climate screen"
```
Wireless (Android 11+): `adb pair <IP>:<PAIR_PORT>` then `adb connect <IP>:5555`.

---

## Linux

### Start
```bash
# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:8b-instruct
# Tools
sudo apt-get update && sudo apt-get install -y poppler-utils android-sdk-platform-tools
# Emulator + SDK: install Android cmdline-tools/Studio; set ANDROID_SDK_ROOT below.

cd ~/ivi-visual-agent
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[docs,embeddings,cv]'

export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export PATH="$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"
```
> The emulator needs KVM: `sudo apt-get install -y qemu-kvm` and add yourself to
> the `kvm` group. Headless CI can use `emulator -no-window`.

### Run the manual
```bash
ivi-agent manual build --source examples/benz-mbux-manual --output output/pdf/benz.pdf
ivi-agent --config config.json knowledge index output/pdf/benz.pdf --profile benz
ivi-agent graph build --profile benz
```

### Run on an emulator
```bash
emulator -list-avds
emulator -avd <AVD_NAME> -no-audio -no-snapshot &
adb wait-for-device && adb devices
adb install -r ./ivi-sample.apk
adb shell am start -n com.example.iviwv/.MainActivity
ivi-agent --config config.json run --serial emulator-5554 --knowledge-profile benz --goal "Open the climate screen"
```

### Run on a direct (physical) device / head unit
1. Device: enable **USB debugging**; connect USB; accept the RSA prompt.
2. If the device isn't detected, add a udev rule for the vendor id and
   `sudo udevadm control --reload && sudo udevadm trigger`, then `adb kill-server && adb start-server`.
```bash
adb devices
adb -s <SERIAL> shell dumpsys SurfaceFlinger --display-id
ivi-agent --config config.json run --serial <SERIAL> --knowledge-profile benz --goal "Open the climate screen"
```
Network head unit: `adb connect <IP>:5555`.

---

## Windows (PowerShell)

### Start
```powershell
winget install Ollama.Ollama
winget install Google.PlatformTools          # adb
# Poppler (for pdftoppm): scoop install poppler   — or conda install -c conda-forge poppler
# Emulator + SDK: install Android Studio, or the command-line tools.
ollama pull qwen3-vl:8b-instruct

cd $HOME\ivi-visual-agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[docs,embeddings,cv]"
```
Put the SDK tools on PATH for this session:
```powershell
$env:ANDROID_SDK_ROOT = "$env:LOCALAPPDATA\Android\Sdk"
$env:Path = "$env:ANDROID_SDK_ROOT\platform-tools;$env:ANDROID_SDK_ROOT\emulator;$env:Path"
```

### Run the manual
```powershell
ivi-agent manual build --source examples\benz-mbux-manual --output output\pdf\benz.pdf
ivi-agent --config config.json knowledge index output\pdf\benz.pdf --profile benz
ivi-agent graph build --profile benz
```
> If `pdftoppm` isn't found, `knowledge index` can't render the PDF — install
> Poppler and reopen the shell so PATH updates.

### Run on an emulator
```powershell
emulator -list-avds
Start-Process emulator -ArgumentList '-avd','<AVD_NAME>','-no-audio'
adb wait-for-device; adb devices
adb install -r $HOME\Downloads\ivi-sample.apk
adb shell am start -n com.example.iviwv/.MainActivity
ivi-agent --config config.json run --serial emulator-5554 --knowledge-profile benz --goal "Open the climate screen"
```

### Run on a direct (physical) device / head unit
1. Install the OEM/Google **USB driver**; on the device enable **USB debugging**; connect USB; accept the RSA prompt.
2. Verify and run:
```powershell
adb devices
adb -s <SERIAL> shell dumpsys SurfaceFlinger --display-id
ivi-agent --config config.json run --serial <SERIAL> --knowledge-profile benz --goal "Open the climate screen"
```
Network head unit: `adb connect <IP>:5555`.

---

## After a run (any OS)

```bash
ivi-agent graph show --profile benz                # coverage + pending findings
```

When the agent reaches a screen or takes a path the manual doesn't describe, it
finishes the task (using the model that once) and flags the path for review
instead of failing. You decide whether it's a real path or a defect:

```bash
ivi-agent graph review --profile benz --finding F0001 --approve   # legit path
ivi-agent graph review --profile benz --finding F0001 --defect    # a bug
```

**Approving writes the path into the manual** (a new screen page, or the control
that reaches it). The manual becomes the agent's memory: the next run reads that
page, so the a11y fast-path and retrieval handle the hop with **no model call**.
Marking it a defect leaves it flagged and does *not* teach it. So the model is
needed to discover a path once; after you approve it, that path is free forever.

The newest `runs/<timestamp>/` holds `result.json` (outcome, grounding, scene
graph), `report.html`, `replay.html`, `events.jsonl`, `agent.log`, and step
screenshots. macOS `open <file>`, Linux `xdg-open <file>`, Windows `start <file>`.

`replay.html` is an interactive, self-contained step-by-step console (screens,
actions, grounding, reasoning stream, phase timing, scene-graph coverage,
crashes) written automatically at the end of every run — one file you can open
offline or send to someone. Rebuild it for any past run with:

```bash
ivi-agent replay --run runs/<timestamp>
```

Speed vs rigour is one flag: `ivi-agent run --profile fast` (minimum overhead) /
`balanced` (default) / `strict` (verify every step). It overrides `config.json`
for the keys it owns.

How hard the agent *proves* success is a second knob, `--verification-level`
(or `verification_level` in config.json):
`off` (trust cheap title/cue signals) / `final` (model confirms the overall goal
once) / `checkpoints` (default — model verifies each step of the final subgoal) /
`strict` (model verifies every step of every subgoal). `--profile fast` implies
`final`, `--profile strict` implies `strict`; an explicit `--verification-level`
wins over both.

For trustworthy results, set `"target_package"` and `"relaunch_before_run": true`
in `config.json`: the run force-stops and relaunches the app cold before each
validation, so a "pass" reflects the flow actually happening — not state a
previous run left behind. Off by default.

To drive validations from an MCP client (Claude Code, Antigravity, Cursor),
install the server with `pip install -e '.[mcp]'` and run `ivi-agent-mcp`. See
[docs/mcp-server.md](mcp-server.md) for the tools and client config.

## Notes that bite on every OS

- **Re-activate the venv in every new terminal** before `ivi-agent`
  (`source .venv/bin/activate`, or `.\.venv\Scripts\Activate.ps1` on Windows), or
  use `python -m ivi_agent …`.
- **Use the exact serial** from `adb devices` for `--serial`.
- **Multi-display** targets: `screencap -d` needs a physical display id, not `0`;
  the agent resolves this from `dumpsys SurfaceFlinger --display-id`. Prefer
  omitting `--display-id` (auto-selects primary); pass `--display-id 0` (HWC
  index) only if needed.
- **Apple Silicon / arm hosts:** use arm64 system images for the emulator.
- **Real head units:** must be Android/AAOS with ADB enabled; QNX or ADB-locked
  units can't be driven. Bench-test stationary units only.
- The APK (`ivi-sample.apk`) is only for the sample mock; against a real head
  unit you drive the existing HMI directly (no install).

See also: [creating-a-manual.md](creating-a-manual.md) and the main [README](../README.md).
