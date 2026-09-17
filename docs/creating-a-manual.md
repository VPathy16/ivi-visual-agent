# Creating a manual (knowledge profile)

A **manual** is the knowledge the agent uses to drive an IVI it has never seen:
which screens exist, what the controls look like, and the steps for each task. It
is authored as a folder — a `manual.json` plus an `images/` directory — and
compiled into a self-contained PDF that the agent indexes into a **profile**.

One manual feeds three subsystems:

| Subsystem | What it uses | Payoff |
| --- | --- | --- |
| **RAG retrieval** | text of every entry (name, meaning, synonyms, steps) | the planner gets the right instruction/icon for the goal |
| **OpenCV fast-path** | `icon` crops in `images/` | taps grounded in ~0.2s **without a model call** |
| **Scene graph** | `screens` + `controls[].result` | expected model; live divergences flagged as defects |
| **Subgoal cues** | `steps[].success_text` | in-screen steps (e.g. "program selected") advance without re-tapping |

> **The single biggest lever is icon crops.** Every tappable control that has a
> good, real crop can be grounded by OpenCV instead of the vision model — turning
> a ~15s step into ~0.2s. Aim to cover *as many taps as possible* with icons.

---

## 1. Folder layout

Start from the scaffold:

```bash
cp -R examples/benz-mbux-manual examples/my-vehicle-manual
```

```
examples/my-vehicle-manual/
├── manual.json          # the structured knowledge (authored by you)
└── images/
    ├── home.png             # full screen reference (optional)
    ├── climate.png
    ├── icon-climate.png     # ICON CROP — used by the CV fast-path
    ├── icon-seat-massage.png
    └── ...
```

- **Screen images** (`home.png`, …) are references for retrieval/documentation.
- **Icon crops** (`icon-*.png`) are what the CV fast-path template-matches on the
  live screen. These are the important ones — see §4.

Do **not** put device tap coordinates anywhere. The agent locates every control
on the live screen at runtime; hard-coded coordinates would break on any layout
or resolution change.

---

## 2. `manual.json` schema

```jsonc
{
  "schema_version": 1,
  "manual": {
    "id": "my-vehicle-v1",           // stable id, becomes the manual_id
    "title": "My Vehicle IVI",
    "version": "1.0",
    "ui_profile": "custom",
    "description": "One-line description."
  },

  "icons": [
    {
      "id": "climate",                     // referenced by a screen control's icon_id
      "name": "Climate",                   // used for retrieval + graph/label matching
      "image": "images/icon-climate.png",  // ICON CROP -> CV fast-path template
      "meaning": "Opens the climate/HVAC screen.",
      "opens": "screen.climate",           // documentation of what it leads to
      "synonyms": ["A/C", "air conditioning", "temperature", "HVAC"]
    }
  ],

  "screens": [
    {
      "id": "screen.home",
      "name": "Home",                      // MUST match the on-screen title (see §5)
      "image": "images/home.png",
      "description": "Custom launcher with large tiles.",
      "landmarks": ["Status bar", "Climate / Seat / Vehicle tiles"],
      "controls": [
        {
          "id": "home.climate_tile",
          "name": "Climate tile",
          "icon_id": "climate",            // links the control to an icon crop
          "action": "Activate the tile",
          "result": "screen.climate"       // -> a scene-graph transition edge
        }
      ]
    }
  ],

  "tasks": [
    {
      "id": "task.start_seat_massage",
      "name": "Start the seat massage",
      "goal": "Start the seat massage",
      "steps": [
        {
          "screen": "screen.home",
          "target": "home.seat_tile",
          "instruction": "Activate the Seat Comfort tile.",
          "milestone": "Open the seat massage screen", // matches a planned subgoal
          "expected": "The seat massage programs are visible",
          "success_text": "Seat massage"    // substring that, once visible, = done
        },
        {
          "screen": "screen.seat_massage",
          "target": "seat.program_wave",
          "instruction": "Select a massage program such as Wave.",
          "milestone": "Select a massage program",
          "expected": "A program is highlighted as selected",
          "success_text": "Selected"        // in-screen cue -> no re-tap loop
        }
      ],
      "success": ["A massage program is selected", "The massage is running"],
      "forbidden": ["Do not change drive mode", "Do not adjust seat position"]
    }
  ]
}
```

### Field-by-field: how each is consumed

- **`icons[].image`** → the CV fast-path template. *This is what makes a tap fast.*
- **`icons[].name` / `synonyms` / `meaning`** → retrieval text. Rich synonyms mean
  the icon is retrieved for more phrasings of a goal (and only a *retrieved* icon
  is eligible for the CV fast-path).
- **`screens[].name`** → matched against the live screen title by the scene graph.
  Keep it close to the actual on-screen title (§5).
- **`screens[].controls[].result`** → a documented transition edge in the scene
  graph. When the agent moves between those screens, the edge is *confirmed*; an
  undocumented move is flagged for review.
- **`tasks[].steps[].milestone`** → aligned to the planner's subgoals (the planner
  is given the task, so it tends to reuse these milestone strings).
- **`tasks[].steps[].success_text`** → the substring that advances an in-screen
  subgoal. Set it to text the HMI actually shows when the step is done (e.g.
  `"Selected"`, `"Massage running"`), **not** the prose in `expected`.
- **`success` / `forbidden`** → success criteria and safety guardrails surfaced to
  the planner/verifier.

---

## 3. Build → index → seed the graph

```bash
python -m pip install -e '.[docs,embeddings,cv]'
brew install poppler                       # pdftoppm, required to index

ivi-agent manual build   --source examples/my-vehicle-manual --output output/pdf/my-vehicle.pdf
ivi-agent --config config.json knowledge index output/pdf/my-vehicle.pdf --profile my-vehicle
ivi-agent graph build    --profile my-vehicle
```

`manual build` validates that every referenced image exists and embeds the JSON +
images into the PDF, so the profile is fully reproducible from one file. `knowledge
index` renders pages, extracts text, and (with `use_embeddings`) writes semantic
vectors. `graph build` seeds the expected scene graph.

> **Re-indexing:** the profile is created once. To change the manual, rebuild the
> PDF and `rm -rf knowledge/<profile>` before re-indexing. Re-run `graph build`
> afterward (the graph lives inside the profile directory).

---

## 4. Icons — covering as many taps as possible with OpenCV

This is where you make runs fast. The fast-path fires for a step when **all** hold:

1. `cv_fast_path: true` in `config.json`.
2. The retriever surfaces an `icon` chunk for the current subgoal with score
   ≥ `cv_min_retrieval_score` (default 2.0).
3. A multi-scale template match of that icon's crop on the live screen scores
   ≥ `cv_match_threshold` (default 0.75). The match score becomes the tap's
   confidence, so it must also clear `minimum_action_confidence`.

### To cover more taps, add an icon per tappable control

Don't limit `icons` to screen-openers. Add an icon entry (with a real crop) for
**every control you want grounded fast** — program cards, a Start button, toggles —
and give each distinctive `name`/`synonyms` so it's retrieved for the right
subgoal. A control can reference it via `icon_id`.

### Getting good crops

Real crops from a live screenshot are essential — the placeholder images from
`scripts/generate_benz_manual_images.py` will **not** match (they score ~0.35).

```bash
# capture a screen, then crop the icon boxes from it
RUN=$(ls -dt runs/*/ | head -1)
python3 scripts/crop_icons_from_screenshot.py "$RUN/step-01.png"   # edit ICONS/positions inside for your HMI
```

Crop guidelines:
- Crop from a **native-resolution** screenshot (`runs/*/step-NN.png`), not a scaled
  or Retina capture.
- Include **just the distinctive glyph/box**; a little surrounding padding is fine
  because the screen is static, but avoid capturing large shared backgrounds or
  neighbouring controls.
- One crop per control; keep them visually distinct so matches are unambiguous.
- Never re-run the placeholder generator after cropping — it overwrites your crops.

### Verify a crop before trusting it

```bash
python3 - <<'PY'
from pathlib import Path
from ivi_agent.config import Config
from ivi_agent.knowledge import KnowledgeBase
from ivi_agent.embeddings import resolve_text_embedder
from ivi_agent.vision_match import locate_template
cfg = Config.load("config.json")
emb = resolve_text_embedder(cfg.ollama_url, cfg.embedding_model, cfg.use_embeddings, cfg.embedding_backend)
kb = KnowledgeBase.open(Path("knowledge"), "my-vehicle", embedder=emb)
res = kb.query("<a goal that should surface the icon>", cfg.knowledge_top_k)
shot = sorted(Path('runs').glob('*/step-01.png'))[-1].read_bytes()
for c in res["chunks"]:
    img = c.get("image")
    if c["kind"] == "icon" and isinstance(img, str) and Path(img).is_file():
        m = locate_template(shot, Path(img).read_bytes(), 0.0)
        print(c["id"], "score=%.3f" % (m[0] if m else -1), "retrieval=%.2f" % c.get("score", 0))
PY
```

- **score ≥ 0.75 and retrieval ≥ 2.0** → the fast-path will fire.
- **score just under 0.75** → re-crop tighter, or lower `cv_match_threshold` *and*
  `minimum_action_confidence` together (keep them equal).
- **retrieval < 2.0** → add synonyms so the icon ranks higher, or lower
  `cv_min_retrieval_score`.

> Text-label "icons" (a word in a coloured box) match poorly in general; real
> bitmap/logo icons on a production IVI match far more reliably. For a static
> mock, an exact crop still matches ~1.0.

---

## 5. Screen names, the scene graph, and defects

The scene graph matches a reached screen to a manual node by **screen-title
tokens** (single characters and pure numbers are ignored, so status-bar clutter
like `Driver 22.0°C` doesn't interfere). To keep matching reliable:

- Set `screens[].name` close to the **actual on-screen title** ("Climate", not a
  long internal name). A verbose manual name still matches a short title, but
  don't drift far.
- Every `control.result` you declare becomes an **expected transition**. Model the
  real navigation so legitimate moves are confirmed and only genuine surprises are
  flagged.
- After runs, triage what the agent found:
  ```bash
  ivi-agent graph show   --profile my-vehicle          # coverage + pending findings
  ivi-agent graph review --profile my-vehicle --finding F0001 --decision approve  # or --decision defect
  ```
  A screen the live HMI shows but the manual doesn't describe is a
  `pending_review` finding — **approve** it (legitimate, fold into the model) or
  mark it a **defect** (the HMI diverges from its spec).

---

## 6. Task steps and cues (avoiding re-tap loops)

Navigation subgoals advance when a new screen title appears. **In-screen** steps
(select a program, toggle a setting) have no title change, so declare a
`success_text` cue that the HMI shows when the step completes:

- `"success_text": "Selected"` — advances once the program shows as selected.
- `"success_text": "Massage running"` — the final running state.

Match the cue to text the HMI actually renders. Without it, a small model may
re-tap the same control repeatedly until the overall-goal recheck rescues it.

---

## 7. Checklist

- [ ] `manual.json` validates and `manual build` succeeds (all images referenced).
- [ ] `screens[].name` ≈ the real on-screen titles.
- [ ] Every navigable move is a `control.result` edge.
- [ ] An `icon` with a **real crop** exists for every control you want CV-grounded.
- [ ] Each in-screen step has a `success_text` cue matching real UI text.
- [ ] Crops verified at score ≥ 0.75; retrieval ≥ 2.0.
- [ ] Indexed with `use_embeddings` on if you want semantic recall.
- [ ] No tap coordinates anywhere in the manual.

See also: [`examples/benz-mbux-manual/`](../examples/benz-mbux-manual/) (a complete
worked example) and the **Living scene graph** and **OpenCV fast-path** sections of
the main [README](../README.md).
