"""Head-only trainer for the VisionLaya VERIFY head — Mac-friendly.

Strategy that fits a 16GB Apple-Silicon MacBook: freeze a small pretrained
vision backbone, cache each screenshot's embedding once, then train a tiny
classifier head on top. That keeps memory tiny (embeddings, not gradients
through the backbone) and runs on MPS or CPU.

torch / torchvision / timm are imported lazily so importing this package never
requires them; install the ``[visionlaya]`` extra to actually train:

    pip install -e '.[visionlaya]'
    visionlaya export --runs runs --out data/visionlaya.jsonl
    visionlaya train  --data data/visionlaya.jsonl --task verify --out models/verify.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import read_jsonl
from .schema import VERIFY


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch  # noqa: PLC0415
        import timm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "VisionLaya training needs the '[visionlaya]' extra:\n"
            "  pip install -e '.[visionlaya]'"
        ) from exc
    return torch, timm


def pick_device() -> str:
    """Prefer Apple MPS, then CUDA, then CPU."""
    torch, _ = _require_torch()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_verify_head(
    data_path: Path,
    out_path: Path,
    backbone: str = "mobilevit_xs",
    epochs: int = 20,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train the VERIFY head (goal-state classifier) on exported examples.

    Freezes ``backbone``, embeds each screenshot once, and fits a small MLP head
    mapping embedding -> P(goal satisfied). Saves the head (and metadata) to
    ``out_path``. Returns a small training summary.
    """
    torch, timm = _require_torch()
    from PIL import Image  # noqa: PLC0415

    device = pick_device()
    examples = [e for e in read_jsonl(data_path) if e.task == VERIFY and e.satisfied is not None]
    if not examples:
        raise SystemExit(f"No VERIFY examples in {data_path}. Run `visionlaya export` first.")

    encoder = timm.create_model(backbone, pretrained=True, num_classes=0).eval().to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    config = timm.data.resolve_data_config({}, model=encoder)
    transform = timm.data.create_transform(**config)

    # Cache embeddings once (the memory-light part: no backbone gradients).
    feats, labels = [], []
    with torch.no_grad():
        for ex in examples:
            try:
                image = Image.open(ex.image).convert("RGB")
            except OSError:
                continue
            tensor = transform(image).unsqueeze(0).to(device)
            feats.append(encoder(tensor).squeeze(0).cpu())
            labels.append(1.0 if ex.satisfied else 0.0)
    if not feats:
        raise SystemExit("No readable screenshots referenced by the dataset.")
    X = torch.stack(feats).to(device)
    y = torch.tensor(labels, device=device).unsqueeze(1)

    head = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(head(X), y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head": head.state_dict(), "backbone": backbone, "task": VERIFY,
         "embedding_dim": X.shape[1]},
        out_path,
    )
    return {"task": VERIFY, "examples": len(labels), "device": device,
            "final_loss": round(last_loss, 4), "model": str(out_path)}
