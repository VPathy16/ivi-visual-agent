# Windows desktop (.exe) backend

The agent is platform-agnostic: everything above the device layer — the grounding
ladder (a11y → CV → vision model), manual RAG, living scene graph + defect
detection, verification levels, replay, and the learn-on-approve loop — is shared.
Automating Windows `.exe` apps is therefore a **port of one layer**, not a new
product: implement `ivi_agent.device.Device` for Windows, and the brain works
unchanged.

## Architecture

```
GoalAgent ── uses ──▶ Device (protocol, src/ivi_agent/device.py)
                        ├── AdbDevice                 (Android, over ADB)
                        └── WindowsDevice             (Windows, over Appium/WinAppDriver)
```

`Device` is a structural `Protocol`; both backends conform without subclassing
(a test asserts this).

## The linchpin: `uia_to_uiautomator`

The shared perception layer parses **uiautomator XML** — `<node>` elements with
`bounds='[x1,y1][x2,y2]'`, `text`, `content-desc`, `resource-id`, `class`,
`clickable`. Appium's Windows driver returns the **UI-Automation** tree via
`driver.page_source`, whose tags are control types with attributes `Name`,
`AutomationId`, `ClassName`, and geometry (`x`/`y`/`width`/`height`).

`ivi_agent.windows.uia.uia_to_uiautomator` maps one to the other:

| UIA | uiautomator |
|-----|-------------|
| `Name` | `text` |
| `AutomationId` | `resource-id` |
| element tag (control type) | `class` (`Edit`→`EditText`) |
| `x,y,width,height` | `bounds='[x,y][x+w,y+h]'` |
| control type ∈ actionable set / `IsKeyboardFocusable` | `clickable` |

Get this mapping right and `extract_ui_elements`, `resolve_target_in_tree`, the
a11y fast-path and the scene graph read Windows screens exactly like Android ones
(covered by `tests/test_windows_backend.py`, no Windows required).

## Why Appium (and the escape hatch)

- **Appium core + `Appium-Python-Client` are Apache-2.0** — clean for a product.
- It gives `page_source` (UIA XML), `get_screenshot_as_png`, and `windows:`
  gesture actions over the standard WebDriver protocol.
- **Caveat:** Appium drives Microsoft's **WinAppDriver**, a *free but
  closed-source, no-longer-actively-maintained* binary (its own EULA; weaker on
  modern WinUI3 / custom-drawn apps). Because the `Device` protocol isolates this,
  a **direct-UIA backend** (`pywinauto`, BSD-3, or `uiautomation`, Apache-2.0 — no
  server) is a drop-in replacement if WinAppDriver's staleness or licensing bites.
  We already own the tree parsing + grounding, so Appium's element-finding isn't
  load-bearing.

## Status (this is Phase 0 — GPU/Windows-free scaffolding)

Done and tested here:
- `Device` protocol; `AdbDevice` and `WindowsDevice` both conform.
- `uia_to_uiautomator` adapter + the perception stack reading its output.
- `WindowsDevice` over an injected driver: `capture`, `ui_dump`, `screen_size`,
  and tap / input_text / enter execution (verified with a fake driver).

Phase 1 (on a Windows box): wire the live Appium session (`WindowsDevice.connect`),
**handle DPI scaling + window-origin offset + multi-monitor** coordinate mapping,
drive a real app (Calculator/Notepad), poll-based `wait_until_stable`.

Phase 2: swipe/gesture parity, clean-start (`relaunch` via launch/taskkill),
crash capture from the **Windows Event Log (Application)** / WER dumps.

## Install & connect

```bash
pip install -e '.[windows]'
# plus a running Appium server with the windows driver + WinAppDriver:
#   appium driver install --source=npm appium-windows-driver
#   appium driver run windows install-wad      # installs WinAppDriver
#   (enable Windows Developer Mode)
```

```python
from ivi_agent.windows import WindowsDevice
device = WindowsDevice.connect(app=r"C:\\Windows\\System32\\calc.exe")
```

Then construct `GoalAgent(device, model, config, knowledge=...)` exactly as with
`AdbDevice` — the rest of the agent is identical.
