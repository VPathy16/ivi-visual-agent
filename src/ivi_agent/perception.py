from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


BOUNDS_PATTERN = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


@dataclass(frozen=True)
class UIElement:
    id: int
    label: str
    role: str
    package: str
    resource_id: str
    bounds: tuple[int, int, int, int]
    center: tuple[float, float]
    scrollable: bool
    source: str = "ui_tree"
    checkable: bool = False
    checked: bool = False
    stateful: bool = False
    editable: bool = False
    focused: bool = False

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "role": self.role,
            "center": [round(self.center[0], 4), round(self.center[1], 4)],
            "scrollable": self.scrollable,
            "checkable": self.checkable,
            "checked": self.checked,
            "stateful": self.stateful,
            "editable": self.editable,
            "focused": self.focused,
            "source": self.source,
        }


def parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    match = BOUNDS_PATTERN.fullmatch(value)
    if not match:
        return None
    left, top, right, bottom = (int(item) for item in match.groups())
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _label(node: ET.Element) -> str:
    for key in ("text", "content-desc"):
        value = node.attrib.get(key, "").strip()
        if value:
            return value
    for child in node.iter("node"):
        if child is node:
            continue
        for key in ("text", "content-desc"):
            value = child.attrib.get(key, "").strip()
            if value:
                return value
    resource_id = node.attrib.get("resource-id", "")
    return resource_id.rsplit("/", 1)[-1].replace("_", " ").strip()


def extract_ui_elements(ui_dump: str, limit: int = 40) -> list[UIElement]:
    if not ui_dump.strip():
        return []
    try:
        root = ET.fromstring(ui_dump)
    except ET.ParseError:
        return []
    parsed_nodes: list[tuple[ET.Element, tuple[int, int, int, int]]] = []
    for node in root.iter("node"):
        bounds = parse_bounds(node.attrib.get("bounds", ""))
        if bounds:
            parsed_nodes.append((node, bounds))
    if not parsed_nodes:
        return []
    screen_width = max(bounds[2] for _, bounds in parsed_nodes)
    screen_height = max(bounds[3] for _, bounds in parsed_nodes)
    if screen_width <= 0 or screen_height <= 0:
        return []

    elements: list[UIElement] = []
    seen: set[tuple[tuple[int, int, int, int], str]] = set()
    for node, bounds in parsed_nodes:
        role = node.attrib.get("class", "View").rsplit(".", 1)[-1]
        editable = role == "EditText"
        interactive = (
            node.attrib.get("clickable") == "true"
            or node.attrib.get("checkable") == "true"
            or node.attrib.get("scrollable") == "true"
            or editable
        )
        if not interactive:
            continue
        label = _label(node)
        scrollable = node.attrib.get("scrollable") == "true"
        if not label and not scrollable:
            continue
        key = (bounds, label)
        if key in seen:
            continue
        seen.add(key)
        left, top, right, bottom = bounds
        checkable = node.attrib.get("checkable") == "true"
        stateful_label = bool(
            re.match(r"^(?:on|off|enabled|disabled)(?:\b|\s*,)", label, re.IGNORECASE)
        )
        elements.append(
            UIElement(
                id=len(elements) + 1,
                label=label or f"scrollable {role}",
                role=role,
                package=node.attrib.get("package", ""),
                resource_id=node.attrib.get("resource-id", ""),
                bounds=bounds,
                center=(
                    ((left + right) / 2) / screen_width,
                    ((top + bottom) / 2) / screen_height,
                ),
                scrollable=scrollable,
                checkable=checkable,
                checked=node.attrib.get("checked") == "true",
                stateful=checkable or stateful_label,
                editable=editable,
                focused=node.attrib.get("focused") == "true",
            )
        )
        if len(elements) >= limit:
            break
    return elements


def parse_ocr_tsv(
    tsv: str,
    screen_width: int,
    screen_height: int,
    start_id: int = 1,
    minimum_confidence: float = 45.0,
    limit: int = 40,
) -> list[UIElement]:
    if screen_width <= 0 or screen_height <= 0:
        return []
    lines = tsv.splitlines()
    if not lines:
        return []
    headers = lines[0].split("\t")
    elements: list[UIElement] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(headers):
            continue
        row = dict(zip(headers, values))
        text = row.get("text", "").strip()
        try:
            confidence = float(row.get("conf", "-1"))
            left = int(row.get("left", "0"))
            top = int(row.get("top", "0"))
            width = int(row.get("width", "0"))
            height = int(row.get("height", "0"))
        except ValueError:
            continue
        if (
            not text
            or len(re.sub(r"\W", "", text)) < 2
            or not any(character.isalpha() for character in text)
            or confidence < minimum_confidence
            or width <= 0
            or height <= 0
        ):
            continue
        bounds = (left, top, left + width, top + height)
        key = (text.lower(), bounds)
        if key in seen:
            continue
        seen.add(key)
        elements.append(
            UIElement(
                id=start_id + len(elements),
                label=text,
                role="OCRText",
                package="",
                resource_id="",
                bounds=bounds,
                center=(
                    (left + width / 2) / screen_width,
                    (top + height / 2) / screen_height,
                ),
                scrollable=False,
                source="ocr",
            )
        )
        if len(elements) >= limit:
            break
    return elements


def extract_ocr_elements(image: bytes, start_id: int = 1, limit: int = 40) -> list[UIElement]:
    if not shutil.which("tesseract"):
        return []
    with Image.open(io.BytesIO(image)) as source:
        width, height = source.size
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "11", "tsv"],
            input=image,
            capture_output=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return parse_ocr_tsv(
        result.stdout.decode(errors="replace"),
        width,
        height,
        start_id=start_id,
        limit=limit,
    )


def packages_in_ui(ui_dump: str, limit: int = 8) -> list[str]:
    if not ui_dump.strip():
        return []
    try:
        root = ET.fromstring(ui_dump)
    except ET.ParseError:
        return []
    packages: list[str] = []
    for node in root.iter("node"):
        package = node.attrib.get("package", "")
        if package and package not in packages:
            packages.append(package)
        if len(packages) >= limit:
            break
    return packages


def extract_screen_titles(ui_dump: str, limit: int = 4) -> list[str]:
    """Return only controls explicitly identified by the UI as screen titles.

    This intentionally uses semantic resource roles rather than device packages,
    screen coordinates, or product-specific title text.
    """
    if not ui_dump.strip():
        return []
    try:
        root = ET.fromstring(ui_dump)
    except ET.ParseError:
        return []
    title_roles = {
        "action bar title",
        "collapsing toolbar title",
        "header title",
        "screen title",
        "toolbar title",
    }
    titles: list[str] = []
    for node in root.iter("node"):
        resource_name = node.attrib.get("resource-id", "").rsplit("/", 1)[-1]
        role = " ".join(re.findall(r"[a-z0-9]+", resource_name.lower()))
        if role not in title_roles and not any(
            role.endswith(f" {title_role}") for title_role in title_roles
        ):
            continue
        label = _label(node)
        if label and label not in titles:
            titles.append(label)
        if len(titles) >= limit:
            break
    return titles


def extract_visible_text(ui_dump: str, limit: int = 80) -> list[str]:
    """Collect deduplicated text that Android reports as currently visible."""
    if not ui_dump.strip():
        return []
    try:
        root = ET.fromstring(ui_dump)
    except ET.ParseError:
        return []
    values: list[str] = []
    for node in root.iter("node"):
        for key in ("text", "content-desc"):
            value = " ".join(node.attrib.get(key, "").split())
            if value and value not in values:
                values.append(value)
                if len(values) >= limit:
                    return values
    return values


def prepare_model_image(image: bytes, max_dimension: int) -> bytes:
    if max_dimension <= 0:
        return image
    with Image.open(io.BytesIO(image)) as source:
        converted = source.convert("RGB")
        converted.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        converted.save(output, format="JPEG", quality=82, optimize=True)
        return output.getvalue()


def prepare_grounded_model_image(
    image: bytes,
    elements: list[UIElement],
    max_dimension: int,
) -> bytes:
    """Overlay candidate IDs so visual reasoning and structured targets agree."""
    with Image.open(io.BytesIO(image)) as source:
        converted = source.convert("RGB")
        draw = ImageDraw.Draw(converted)
        font_size = max(14, min(converted.size) // 28)
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
        line_width = max(2, min(converted.size) // 300)
        for element in elements:
            left, top, right, bottom = element.bounds
            draw.rectangle((left, top, right, bottom), outline="#ff3b30", width=line_width)
            label = str(element.id)
            text_box = draw.textbbox((left, top), label, font=font, stroke_width=1)
            padding = 3
            background = (
                text_box[0] - padding,
                text_box[1] - padding,
                text_box[2] + padding,
                text_box[3] + padding,
            )
            draw.rectangle(background, fill="#ff3b30")
            draw.text(
                (left, top),
                label,
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill="#ff3b30",
            )
        if max_dimension > 0:
            converted.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        converted.save(output, format="JPEG", quality=84, optimize=True)
        return output.getvalue()


def prepare_grid_grounding_image(
    image: bytes,
    max_dimension: int,
    columns: int = 12,
    rows: int = 6,
) -> bytes:
    """Overlay a device-independent numbered grid for coarse visual grounding.

    Small local vision models are often good at recognizing a control but poor at
    emitting pixel coordinates. A numbered grid turns localization into a simple
    classification problem without encoding any device layout or application route.
    Cells are numbered left-to-right and then top-to-bottom, starting at one.
    """
    with Image.open(io.BytesIO(image)) as source:
        converted = source.convert("RGB")
        if max_dimension > 0:
            converted.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        width, height = converted.size
        draw = ImageDraw.Draw(converted)
        font_size = max(12, min(converted.size) // 34)
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
        line_width = max(1, min(converted.size) // 400)
        for column in range(1, columns):
            x = round(width * column / columns)
            draw.line((x, 0, x, height), fill="#00e5ff", width=line_width)
        for row in range(1, rows):
            y = round(height * row / rows)
            draw.line((0, y, width, y), fill="#00e5ff", width=line_width)
        for row in range(rows):
            for column in range(columns):
                cell = row * columns + column + 1
                x = round(width * column / columns) + 3
                y = round(height * row / rows) + 2
                label = str(cell)
                box = draw.textbbox((x, y), label, font=font, stroke_width=1)
                draw.rectangle(
                    (box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1),
                    fill="#111111",
                )
                draw.text(
                    (x, y),
                    label,
                    fill="#00e5ff",
                    font=font,
                    stroke_width=1,
                    stroke_fill="#111111",
                )
        output = io.BytesIO()
        converted.save(output, format="JPEG", quality=88, optimize=True)
        return output.getvalue()


def perceptual_hash(image: bytes, size: int = 16) -> int:
    with Image.open(io.BytesIO(image)) as source:
        reduced = source.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = list(getattr(reduced, "get_flattened_data", reduced.getdata)())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def hash_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def exact_digest(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()
