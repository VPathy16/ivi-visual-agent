from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class KnowledgeError(ValueError):
    pass


def _tokens(value: str) -> list[str]:
    aliases = {
        "bt": "bluetooth",
        "media": "audio",
        "inputs": "source",
        "sources": "source",
    }
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(value.lower()):
        tokens.append(aliases.get(token, token))
    return tokens


def _safe_profile(profile: str) -> str:
    if not PROFILE_PATTERN.fullmatch(profile):
        raise KnowledgeError(
            "profile must use only letters, numbers, dots, underscores, or hyphens"
        )
    return profile


def _attachment_bytes(reader: Any, name: str) -> bytes | None:
    values = reader.attachments.get(name)
    if not values:
        return None
    value = values[0] if isinstance(values, list) else values
    return bytes(value)


def _attachment_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise KnowledgeError(f"unsafe embedded attachment path: {name}")
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise KnowledgeError(f"unsafe embedded attachment path: {name}")
    return destination


def _item_text(kind: str, item: dict[str, Any]) -> str:
    parts = [kind, str(item.get("id", "")), str(item.get("name", ""))]
    for key in (
        "goal",
        "meaning",
        "description",
        "opens",
        "action",
        "result",
        "screen",
        "target",
        "instruction",
        "milestone",
        "expected",
    ):
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("synonyms", "landmarks", "success", "forbidden"):
        value = item.get(key)
        if isinstance(value, list):
            parts.extend(str(entry) for entry in value)
    controls = item.get("controls", [])
    if isinstance(controls, list):
        for control in controls:
            if isinstance(control, dict):
                parts.extend(
                    str(value)
                    for value in control.values()
                    if isinstance(value, str)
                )
    steps = item.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                parts.extend(str(value) for value in step.values() if isinstance(value, str))
    return " | ".join(part for part in parts if part)


def _structured_chunks(manual: dict[str, Any], assets: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    metadata = manual.get("manual", {})
    chunks.append(
        {
            "id": f"manual.{metadata.get('id', 'overview')}",
            "kind": "manual",
            "name": metadata.get("title", "Manual overview"),
            "text": _item_text("manual", metadata),
            "data": metadata,
        }
    )
    for kind in ("icons", "screens", "tasks"):
        singular = kind[:-1]
        for item in manual.get(kind, []):
            if not isinstance(item, dict):
                continue
            chunk: dict[str, Any] = {
                "id": str(item.get("id", f"{singular}.unknown")),
                "kind": singular,
                "name": str(item.get("name", item.get("goal", ""))),
                "text": _item_text(singular, item),
                "data": item,
            }
            image_name = item.get("image")
            if isinstance(image_name, str):
                image_path = _attachment_path(assets, image_name)
                if image_path.is_file():
                    chunk["image"] = str(image_path.relative_to(assets.parent))
            chunks.append(chunk)
    return chunks


def index_pdf(pdf: Path, knowledge_root: Path, profile: str) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise KnowledgeError(
            "PDF indexing is not installed. Run: python -m pip install -e '.[docs]'"
        ) from exc

    pdf = pdf.resolve()
    if not pdf.is_file():
        raise KnowledgeError(f"PDF was not found: {pdf}")
    profile = _safe_profile(profile)
    knowledge_root = knowledge_root.resolve()
    knowledge_root.mkdir(parents=True, exist_ok=True)
    destination = knowledge_root / profile
    if destination.exists():
        raise KnowledgeError(
            f"knowledge profile already exists: {destination}; choose a new profile name"
        )
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise KnowledgeError("pdftoppm was not found. Install Poppler before indexing PDFs")

    temporary = Path(tempfile.mkdtemp(prefix=f".{profile}-", dir=knowledge_root))
    try:
        pages = temporary / "pages"
        assets = temporary / "assets"
        pages.mkdir()
        assets.mkdir()
        reader = PdfReader(str(pdf))
        try:
            subprocess.run(
                [pdftoppm, "-png", "-r", "120", str(pdf), str(pages / "page")],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            raise KnowledgeError("PDF page rendering timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace").strip()
            raise KnowledgeError(f"PDF page rendering failed: {detail}") from exc
        page_images = sorted(
            pages.glob("page-*.png"),
            key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
        )
        chunks: list[dict[str, Any]] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            image = page_images[index - 1] if index <= len(page_images) else None
            chunks.append(
                {
                    "id": f"pdf.page.{index}",
                    "kind": "pdf_page",
                    "page": index,
                    "name": f"PDF page {index}",
                    "text": text,
                    "image": str(image.relative_to(temporary)) if image else None,
                }
            )

        embedded_manifest = _attachment_bytes(reader, "manual.json")
        manual_id = profile
        manual_title = pdf.stem
        if embedded_manifest:
            try:
                manual = json.loads(embedded_manifest.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KnowledgeError("embedded manual.json is invalid") from exc
            if not isinstance(manual, dict):
                raise KnowledgeError("embedded manual.json must be a JSON object")
            (temporary / "manual.json").write_bytes(embedded_manifest)
            for attachment_name in reader.attachments:
                if attachment_name == "manual.json":
                    continue
                content = _attachment_bytes(reader, attachment_name)
                if content is None:
                    continue
                target = _attachment_path(assets, attachment_name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            chunks.extend(_structured_chunks(manual, assets))
            metadata = manual.get("manual", {})
            if isinstance(metadata, dict):
                manual_id = str(metadata.get("id", manual_id))
                manual_title = str(metadata.get("title", manual_title))

        chunks_path = temporary / "chunks.jsonl"
        chunks_path.write_text(
            "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "profile": profile,
            "manual_id": manual_id,
            "title": manual_title,
            "source_pdf": str(pdf),
            "source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "pages": len(reader.pages),
            "chunks": len(chunks),
            "structured_manual": embedded_manifest is not None,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)
        return {**manifest, "directory": str(destination)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class KnowledgeBase:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        manifest_path = self.directory / "manifest.json"
        chunks_path = self.directory / "chunks.jsonl"
        if not manifest_path.is_file() or not chunks_path.is_file():
            raise KnowledgeError(f"invalid knowledge profile: {self.directory}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            document_frequency.update(set(_tokens(str(chunk.get("text", "")))))
        count = max(len(self.chunks), 1)
        self.idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    @classmethod
    def open(cls, knowledge_root: Path, profile: str) -> "KnowledgeBase":
        return cls(knowledge_root.resolve() / _safe_profile(profile))

    def query(self, query: str, limit: int = 4) -> dict[str, Any]:
        if limit < 1:
            raise KnowledgeError("query limit must be at least 1")
        query_tokens = Counter(_tokens(query))
        normalized_query = " ".join(query_tokens)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for chunk in self.chunks:
            text = str(chunk.get("text", ""))
            terms = Counter(_tokens(text))
            score = sum(
                self.idf.get(token, 1.0) * min(count, terms.get(token, 0))
                for token, count in query_tokens.items()
            )
            normalized_text = " ".join(_tokens(text))
            if normalized_query and normalized_query in normalized_text:
                score += 8.0
            if chunk.get("kind") == "task":
                score *= 1.35
            elif chunk.get("kind") in {"screen", "icon"}:
                score *= 1.15
            if score > 0:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
        chosen: list[tuple[float, dict[str, Any]]] = []
        chosen_ids: set[str] = set()
        # A compact multimodal result is more useful than four near-duplicate PDF
        # pages. Prefer one task, screen, icon, and source page when available.
        for kind in ("task", "screen", "icon", "pdf_page"):
            match = next((item for item in ranked if item[1].get("kind") == kind), None)
            if match and len(chosen) < limit:
                chosen.append(match)
                chosen_ids.add(str(match[1].get("id", "")))
        for item in ranked:
            if len(chosen) >= limit:
                break
            if str(item[1].get("id", "")) not in chosen_ids:
                chosen.append(item)
                chosen_ids.add(str(item[1].get("id", "")))
        chosen.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))

        selected: list[dict[str, Any]] = []
        for score, chunk in chosen:
            result = dict(chunk)
            result["score"] = round(score, 4)
            image = result.get("image")
            if isinstance(image, str):
                result["image"] = str((self.directory / image).resolve())
            selected.append(result)
        return {
            "profile": self.manifest.get("profile", self.directory.name),
            "manual_id": self.manifest.get("manual_id"),
            "query": query,
            "chunks": selected,
        }


def prompt_context(
    result: dict[str, Any], current_subgoal: str | None = None
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    source_chunks = result.get("chunks", [])
    has_structured = any(
        chunk.get("kind") in {"task", "screen", "icon"} for chunk in source_chunks
    )
    for chunk in source_chunks:
        if has_structured and chunk.get("kind") == "pdf_page":
            continue
        compact = {
            key: value
            for key, value in chunk.items()
            if key in {"id", "kind", "name", "text", "data", "page", "score"}
        }
        if (
            current_subgoal
            and compact.get("kind") == "task"
            and isinstance(compact.get("data"), dict)
        ):
            task = compact["data"]
            steps = task.get("steps", [])
            active_step = None
            normalized_subgoal = " ".join(_tokens(current_subgoal))
            normalized_task_goal = " ".join(_tokens(str(task.get("goal", ""))))
            for step in steps if isinstance(steps, list) else []:
                if not isinstance(step, dict):
                    continue
                step_milestone = step.get("milestone", step.get("expected", ""))
                if " ".join(_tokens(str(step_milestone))) == normalized_subgoal:
                    active_step = step
                    break
            if (
                active_step is None
                and normalized_subgoal == normalized_task_goal
                and steps
            ):
                active_step = steps[-1]
            if active_step is not None:
                compact["data"] = {
                    "goal": task.get("goal"),
                    "active_step": active_step,
                    "success": task.get("success", []),
                    "forbidden": task.get("forbidden", []),
                }
                compact["text"] = _item_text(
                    "active task step",
                    {
                        **active_step,
                        "forbidden": task.get("forbidden", []),
                    },
                )
        chunks.append(compact)
    return {
        "profile": result.get("profile"),
        "manual_id": result.get("manual_id"),
        "retrieved_chunks": chunks,
    }


def reference_images(result: dict[str, Any], limit: int = 2) -> list[bytes]:
    images: list[bytes] = []
    seen: set[str] = set()
    # Only isolated icon crops are safe to show beside the live screen. Full manual
    # pages or example screens can make a small vision model act on the reference.
    chunks = [
        item for item in result.get("chunks", []) if item.get("kind") == "icon"
    ]
    chunks.sort(key=lambda item: -float(item.get("score", 0)))
    for chunk in chunks:
        path_value = chunk.get("image")
        if not isinstance(path_value, str) or path_value in seen:
            continue
        path = Path(path_value)
        if path.is_file():
            images.append(path.read_bytes())
            seen.add(path_value)
        if len(images) >= limit:
            break
    return images
