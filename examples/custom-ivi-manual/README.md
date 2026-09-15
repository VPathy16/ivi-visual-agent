# Custom IVI manual source folder

This folder is the editable source for a RAG-friendly IVI PDF manual.

```text
custom-ivi-manual/
  manual.json
  images/
    home.png
    audio-hub.png
    sources-open.png
    bt-selected.png
    icon-audio-hub.png
    icon-sources.png
    icon-linkwave.png
```

## How to use it

1. Copy this folder and give it a vehicle or UI-profile name.
2. Replace the files under `images/` with screenshots and icon crops from that UI.
3. Edit `manual.json` so every image has a human-readable name and meaning.
4. Build the PDF:

```bash
ivi-agent manual build \
  --source examples/custom-ivi-manual \
  --output output/pdf/sample-custom-ivi-rag-manual.pdf
```

The generator checks the JSON, duplicate IDs, image paths and image readability before
creating the PDF. Image paths must remain inside this source folder.

## What to describe

- `icons`: proprietary icons and what each one means.
- `screens`: a screenshot, its stable landmarks, and visible controls.
- `tasks`: the semantic steps, expected result after each step, pass evidence and
  forbidden actions.

Give each step a short observable `milestone`, preferably using the exact title visible
after the action, such as `Open the Audio Hub screen`. This lets the agent advance
without another model decision when that title appears.

Do not put pixel coordinates in the JSON. The runtime agent must find each documented
control on the current screenshot.
