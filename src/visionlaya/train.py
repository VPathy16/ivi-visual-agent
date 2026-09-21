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
from .schema import GROUND, VERIFY


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


def train_ground_head(
    data_path: Path,
    out_path: Path,
    backbone: str = "mobilevit_xs",
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 40,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train the GROUND head: (image, instruction) -> normalized tap point.

    Instruction-conditioned (the same screen has different targets per goal), so
    it concatenates a frozen image embedding and a frozen text embedding and
    fits a small MLP -> sigmoid(x, y). Memory-light: both encoders are frozen and
    embeddings are cached. Reports mean point error and accuracy@0.05 (fraction
    of predictions within 0.05 normalized distance of the label).

    Honest note: a head on a frozen ImageNet backbone is a *baseline* grounder;
    if accuracy@0.05 is low, switch ``backbone`` to a GUI-grounding-pretrained
    encoder rather than training the head harder.
    """
    torch, timm = _require_torch()
    from PIL import Image  # noqa: PLC0415
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "GROUND training needs sentence-transformers:\n  pip install -e '.[visionlaya]'"
        ) from exc

    device = pick_device()
    examples = [e for e in read_jsonl(data_path) if e.task == GROUND and e.point is not None]
    if not examples:
        raise SystemExit(f"No GROUND examples in {data_path}. Import/export a dataset first.")

    encoder = timm.create_model(backbone, pretrained=True, num_classes=0).eval().to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    config = timm.data.resolve_data_config({}, model=encoder)
    transform = timm.data.create_transform(**config)
    text_encoder = SentenceTransformer(text_model, device=device)

    feats, targets = [], []
    with torch.no_grad():
        for ex in examples:
            try:
                image = Image.open(ex.image).convert("RGB")
            except OSError:
                continue
            img_vec = encoder(transform(image).unsqueeze(0).to(device)).squeeze(0).cpu()
            txt_vec = torch.tensor(text_encoder.encode(ex.instruction or ex.goal))
            feats.append(torch.cat([img_vec, txt_vec]))
            targets.append(torch.tensor(ex.point, dtype=torch.float32))
    if not feats:
        raise SystemExit("No readable screenshots referenced by the dataset.")
    X = torch.stack(feats).to(device)
    y = torch.stack(targets).to(device)

    head = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 2), torch.nn.Sigmoid(),
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = head(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    with torch.no_grad():
        pred = head(X)
        dist = torch.sqrt(((pred - y) ** 2).sum(dim=1))
        acc05 = float((dist <= 0.05).float().mean().item())
        mean_err = float(dist.mean().item())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head": head.state_dict(), "backbone": backbone, "text_model": text_model,
         "task": GROUND, "input_dim": X.shape[1]},
        out_path,
    )
    return {"task": GROUND, "examples": len(targets), "device": device,
            "final_loss": round(last_loss, 4), "mean_point_error": round(mean_err, 4),
            "accuracy@0.05": round(acc05, 4), "model": str(out_path)}
