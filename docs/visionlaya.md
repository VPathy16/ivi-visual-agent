# VisionLaya — a fast image→decision model for the IVI agent

**Goal:** replace the slow autoregressive VLM, for the common learned cases, with
a small **non-autoregressive** image classifier that emits a **structured,
calibrated** decision in tens of milliseconds — and falls back to the VLM only on
novel or low-confidence screens.

Named after [Laya](https://laya.convaiinnovations.com/) (ConvAI's non-autoregressive
decision engine): same idea — typed decisions with calibrated probabilities in
~33ms — but with a **vision** backbone instead of a text one.

## Why it can be faster than the VLM

1. **Non-autoregressive:** one forward pass + read a probability off a head,
   versus the VLM generating a JSON answer token-by-token (each token a full
   pass through ~8B params).
2. **Small:** a 5–400M vision model versus ~8B.

Together that's roughly **50–500×** faster: ~2–10s → ~10–50ms (GPU) / ~100–300ms
(CPU). The speed is essentially free from the architecture; **accuracy is bought
with data**, and only for bounded tasks.

## Two decision heads

| Head | Task | Input → Output | Difficulty |
|------|------|----------------|------------|
| **VERIFY** | `bool` | screenshot (+goal) → goal satisfied? + confidence | tractable — build first |
| **GROUND** | `choice` | screenshot (+goal +N candidate boxes) → which element | the prize — needs more data |

Grounding is framed as **pick 1-of-N** over the candidates we already have (from
the a11y tree / detection), *not* open-ended "where do I tap" — that keeps it a
fast classification and trainable on modest data.

## Where it fits — it complements the VLM, never blindly replaces it

```
a11y tree (~0.001s)  →  VisionLaya (~30ms, learned screens, calibrated)  →  VLM (~seconds, novel/uncertain)
```

VisionLaya's **calibrated confidence** decides when to fall through to the VLM.
Novel HMIs stay correct (VLM); familiar ones get fast. The set of "familiar"
grows as the agent runs — the flywheel below.

## The data flywheel (why this is feasible for *this* project)

The agent generates its own labels:

1. VLM-driven runs write screenshots + outcomes; the **approve/defect** review
   loop ratifies correct vs wrong decisions.
2. `visionlaya export` turns those runs into a labeled dataset.
3. Train VisionLaya on your Mac; it takes over the learned cases.
4. Faster runs → more runs → more labels → better model.

## Using it

Data export is **GPU-free** and works today:

```bash
visionlaya export --runs runs --out data/visionlaya.jsonl
# -> {"exported": N, "by_task": {...}, "verify_labels": {"verify:pass": .., "verify:fail": ..}}
```

### Bootstrapping from Hugging Face (solve the cold-start)

You don't have to collect hundreds of IVI runs first — import public Android
grounding data, then fine-tune on your own runs:

```bash
pip install -e '.[hf]'
visionlaya import-hf --dataset androidcontrol \
  --hf-path UI-MOPD/AndroidControl-Star --split train --limit 5000 \
  --out data/android.jsonl
# ScreenSpot is best kept as an EVAL set (it's a benchmark).
```

Each row becomes a GROUND `Example` (image, instruction, tap point). Then train
on the combined HF + IVI data. **Two honest caveats:**
- **Domain gap:** these are phone/desktop/web UIs, not automotive IVI — use HF to
  learn *general* grounding, then fine-tune on your runs for the IVI last mile.
- **Licenses:** this is a product — many GUI datasets are research-only. Check
  each dataset's license before shipping a model trained on it.

If the head-only baseline grounds poorly, the fix is a **GUI-grounding-pretrained
backbone** (pass `--backbone`), not more head training.

Training runs on a 16GB Apple-Silicon MacBook (frozen backbone + small head; MPS
or CPU). Needs the extra:

```bash
pip install -e '.[visionlaya]'
visionlaya train --data data/visionlaya.jsonl --task verify --out models/verify.pt
```

## Status & roadmap

- [x] Decision schema (`verify` / `ground`) and training `Example`
- [x] Dataset exporter from agent runs (passed-run labels; conservative v0)
- [x] Head-only VERIFY trainer scaffold (frozen backbone, MPS/CPU)
- [ ] Collect enough runs (hundreds+ examples) — the real bottleneck
- [ ] Inference backend + wire into the agent's grounding/verification ladder
- [ ] GROUND head (pick-1-of-N over a11y candidates)
- [ ] Calibration eval (reliability curve) so the confidence threshold is honest

**Honest note:** the model is not trained here — this repo ships the GPU-free
data + training scaffolding. Weights are produced by you (or ConvAI) from the
exported dataset. Labels are v0 heuristics; the review loop and more runs improve
them.
