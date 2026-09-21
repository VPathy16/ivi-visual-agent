"""VisionLaya: a fast image -> structured-decision model for the IVI agent.

The IVI agent's slow step is the autoregressive vision-language model (VLM):
a few seconds per grounding/verification call. VisionLaya is the plan to replace
that, for the common learned cases, with a small *non-autoregressive* image
classifier (Laya-style) that emits a structured, calibrated decision in tens of
milliseconds — and falls back to the VLM only on novel/uncertain screens.

This package is the GPU-free foundation:

* ``schema``  — the decision schema (verify / ground) and the training Example.
* ``dataset`` — turn the agent's own runs (screenshots + outcomes + approved
  paths) into a labeled training set. The agent generates its own data.
* ``train``   — a Mac-friendly, head-only training scaffold (frozen pretrained
  backbone + a small head; runs on Apple-Silicon MPS or CPU). Needs the
  ``[visionlaya]`` extra (torch / torchvision / timm).

See ``docs/visionlaya.md`` for the full design and the a11y -> VisionLaya -> VLM
architecture.
"""

from __future__ import annotations

from .schema import Example, VERIFY, GROUND

__all__ = ["Example", "VERIFY", "GROUND"]
