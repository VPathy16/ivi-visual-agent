# Custom IVI PDF RAG architecture

## Decision

Do not train the local vision model for the first version. Treat the vehicle manual as
a local multimodal knowledge base. The manual teaches the meaning of proprietary icons,
which screen a control opens, and how success appears. The live device remains the
source of truth for what is visible and where it is located.

This works with a Unity UI delivered through native `.so` libraries because it does not
depend on Android exposing every Unity control in the accessibility tree.

## Simple architecture

```text
                        One-time indexing

  PDF manual ---------------------------------------------+
      |                                                    |
      +--> extracted text --> semantic chunks ------------+
      |                                                    |
      +--> rendered pages --> icon/screen examples --------+--> local index

                         Every agent step

  user goal + current screenshot
      |
      +--> retrieve relevant instructions and icon examples
      |
      +--> planner chooses one visible semantic target
      |
      +--> visual grounder finds that target on the live screen
      |
      +--> ADB taps it --> capture next screen --> repeat
      |
      +--> independent verifier checks the documented success state
```

Only the PDF is indexed. No manual content needs to leave the test machine.

## What the PDF should contain

The PDF must be useful to both a person and the retrieval pipeline:

- A stable manual ID and UI version.
- A glossary for each proprietary icon, including its meaning and common synonyms.
- One section per screen with a screen name and distinctive visual landmarks.
- Relationships such as "Audio tile opens Audio Hub" rather than pixel coordinates.
- Goal procedures written as semantic actions.
- Observable success conditions.
- Safety limits and recovery behavior.

The PDF must not encode screen coordinates. Coordinates come from the current screenshot.

## Index format

The indexer renders every PDF page and extracts its text. It creates small JSON records:

```json
{
  "chunk_id": "task.select_bt_source",
  "manual_id": "sample-custom-ivi-v1",
  "page": 5,
  "kind": "task",
  "screen": "audio_sources",
  "text": "Open Audio, open Sources, then select BT Audio.",
  "icon_ids": ["audio_hub", "sources", "linkwave"],
  "page_image": "pages/page-05.png"
}
```

For a small manual, local keyword search plus synonyms is enough. Embeddings can be
added when there are many manuals, languages, or alternate names. Page images and icon
crops are retrieved together with text so the vision model can compare proprietary
symbols visually.

## Runtime contract

For the goal `Select Bluetooth as the media source`, retrieval should provide:

```json
{
  "goal": "Select Bluetooth as the media source",
  "instructions": [
    "Open the Audio tile from Home",
    "Open Sources in Audio Hub",
    "Select BT Audio"
  ],
  "icon_examples": ["audio_hub", "sources", "linkwave"],
  "success": [
    "Audio Hub displays BT Audio as the selected source",
    "A selected indicator is visible beside BT Audio"
  ],
  "forbidden": [
    "Do not open Bluetooth pairing",
    "Do not change the Bluetooth radio state"
  ]
}
```

The planner treats these instructions as a provisional route. Before every tap it must:

1. Confirm the expected control is visible on the current screenshot.
2. Ground the target to a live accessibility element or visually detected bounds.
3. Refuse low-confidence or unsafe actions.
4. Capture the resulting screen and confirm the expected transition.
5. Re-query the manual or recover with Back when the observed screen differs.

## Components to add to this project

### 1. Manual authoring and PDF generator

The generator is implemented. Start with the editable sample folder, replace its
images, and edit `manual.json`:

```bash
ivi-agent manual build \
  --source examples/custom-ivi-manual \
  --output output/pdf/sample-custom-ivi-rag-manual.pdf
```

The manifest provides stable IDs, human names, synonyms, screen relationships, task
steps, success evidence, and forbidden actions. The generated PDF includes both
extractable text and the associated visual examples.

### 2. Manual indexer

```bash
ivi-agent knowledge index output/pdf/sample-custom-ivi-rag-manual.pdf \
  --profile sample-custom-ivi
```

It should create:

```text
knowledge/sample-custom-ivi/
  manifest.json
  chunks.jsonl
  pages/
  icons/
  index.sqlite
```

### 3. Retriever

```bash
ivi-agent knowledge query --profile sample-custom-ivi \
  --goal "Select Bluetooth as the media source"
```

Return the top task, screen, and icon chunks. Keep retrieval deterministic and small.

### 4. RAG-aware planner

Add the retrieved instructions and page/icon images to the existing planner input. The
planner still receives the current screenshot, UI dump, previous actions, and blocked
actions. Documentation never overrides live evidence.

### 5. Verifier

Use the success criteria retrieved from the manual. A visible `BT Audio` menu item is not
enough; the selected state or the active source label must also be visible.

## Why this is preferable to training

- A new or revised PDF can update behavior without retraining model weights.
- Knowledge can be scoped by vehicle program and software version.
- Every action can cite the manual page that motivated it.
- The same general vision model can operate different IVI skins.
- Failures are easier to diagnose as retrieval, grounding, or verification errors.

Training becomes useful only if proprietary icons vary so much that visual retrieval
cannot match them reliably across themes, resolutions, and animations. Even then, train
a small icon detector before considering full vision-language model fine-tuning.

## MVP acceptance test

The first implementation is complete when it can:

1. Index the included sample PDF entirely offline.
2. Retrieve the correct three semantic actions for the Bluetooth-source goal.
3. Show the planner the relevant icon examples.
4. Locate each control from a live screenshot without stored coordinates.
5. Reject completion while Bluetooth is only an unselected menu option.
6. Pass only after `BT Audio` is visibly selected.
