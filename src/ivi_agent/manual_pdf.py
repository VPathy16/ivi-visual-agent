from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from PIL import Image as PillowImage


class ManualValidationError(ValueError):
    pass


def _required_text(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManualValidationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _string_list(data: dict[str, Any], key: str, context: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ManualValidationError(f"{context}.{key} must be a list of strings")
    return [item.strip() for item in value]


def _records(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ManualValidationError(f"{key} must be a list of objects")
    return value


def _resolve_image(source: Path, value: str, context: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ManualValidationError(f"{context}.image must be relative to the source folder")
    root = source.resolve()
    path = (source / relative).resolve()
    if not path.is_relative_to(root):
        raise ManualValidationError(f"{context}.image must remain inside the source folder")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ManualValidationError(f"{context}.image has an unsupported file type")
    if not path.is_file():
        raise ManualValidationError(f"{context}.image was not found: {relative}")
    try:
        with PillowImage.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ManualValidationError(f"{context}.image is not readable: {relative}") from exc
    return path


def load_manual_source(source: Path) -> dict[str, Any]:
    source = source.resolve()
    manifest_path = source / "manual.json"
    if not manifest_path.is_file():
        raise ManualValidationError(f"manual.json was not found in {source}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManualValidationError(f"manual.json is invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManualValidationError("manual.json must contain a JSON object")
    if data.get("schema_version") != 1:
        raise ManualValidationError("schema_version must be 1")

    manual = data.get("manual")
    if not isinstance(manual, dict):
        raise ManualValidationError("manual must be an object")
    for key in ("id", "title", "ui_profile", "version", "description"):
        _required_text(manual, key, "manual")

    icons = _records(data, "icons")
    screens = _records(data, "screens")
    tasks = _records(data, "tasks")
    if not screens:
        raise ManualValidationError("screens must contain at least one screen")
    if not tasks:
        raise ManualValidationError("tasks must contain at least one task")

    seen_ids: set[str] = set()
    icon_ids: set[str] = set()
    for index, item in enumerate(icons):
        context = f"icons[{index}]"
        item_id = _required_text(item, "id", context)
        if item_id in seen_ids:
            raise ManualValidationError(f"duplicate id: {item_id}")
        seen_ids.add(item_id)
        icon_ids.add(item_id)
        for key in ("name", "meaning", "opens"):
            _required_text(item, key, context)
        _string_list(item, "synonyms", context)
        item["_image_path"] = str(
            _resolve_image(source, _required_text(item, "image", context), context)
        )

    screen_ids: set[str] = set()
    control_ids: set[str] = set()
    for index, item in enumerate(screens):
        context = f"screens[{index}]"
        item_id = _required_text(item, "id", context)
        if item_id in seen_ids:
            raise ManualValidationError(f"duplicate id: {item_id}")
        seen_ids.add(item_id)
        screen_ids.add(item_id)
        for key in ("name", "description"):
            _required_text(item, key, context)
        _string_list(item, "landmarks", context)
        item["_image_path"] = str(
            _resolve_image(source, _required_text(item, "image", context), context)
        )
        controls = _records(item, "controls")
        for control_index, control in enumerate(controls):
            control_context = f"{context}.controls[{control_index}]"
            control_id = _required_text(control, "id", control_context)
            if control_id in seen_ids:
                raise ManualValidationError(f"duplicate id: {control_id}")
            seen_ids.add(control_id)
            control_ids.add(control_id)
            for key in ("name", "action", "result"):
                _required_text(control, key, control_context)
            icon_id = control.get("icon_id")
            if icon_id is not None and icon_id not in icon_ids:
                raise ManualValidationError(
                    f"{control_context}.icon_id references unknown icon: {icon_id}"
                )

    for index, item in enumerate(tasks):
        context = f"tasks[{index}]"
        item_id = _required_text(item, "id", context)
        if item_id in seen_ids:
            raise ManualValidationError(f"duplicate id: {item_id}")
        seen_ids.add(item_id)
        for key in ("name", "goal"):
            _required_text(item, key, context)
        _string_list(item, "success", context)
        _string_list(item, "forbidden", context)
        steps = _records(item, "steps")
        if not steps:
            raise ManualValidationError(f"{context}.steps must contain at least one step")
        for step_index, step in enumerate(steps):
            step_context = f"{context}.steps[{step_index}]"
            screen_id = _required_text(step, "screen", step_context)
            if screen_id not in screen_ids:
                raise ManualValidationError(
                    f"{step_context}.screen references unknown screen: {screen_id}"
                )
            target = _required_text(step, "target", step_context)
            if target not in control_ids and target not in icon_ids:
                raise ManualValidationError(
                    f"{step_context}.target references unknown control or icon: {target}"
                )
            for key in ("instruction", "expected"):
                _required_text(step, key, step_context)

    data["_source"] = str(source)
    data["_manifest_path"] = str(manifest_path)
    return data


def _escape(value: object) -> str:
    return html.escape(str(value))


def build_manual_pdf(source: Path, output: Path) -> dict[str, Any]:
    data = load_manual_source(source)
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            HRFlowable,
            Image,
            PageBreak,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ManualValidationError(
            "PDF support is not installed. Run: python -m pip install -e '.[docs]'"
        ) from exc

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manual = data["manual"]
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#172033")
    blue = colors.HexColor("#315EFB")
    muted = colors.HexColor("#5E687C")
    line = colors.HexColor("#DCE2EE")
    pale_blue = colors.HexColor("#EEF3FF")
    pale_green = colors.HexColor("#ECFAF4")
    pale_red = colors.HexColor("#FFF0F1")

    styles.add(ParagraphStyle(
        name="ManualTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=29, leading=34, textColor=navy, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="ManualSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=13, leading=18, textColor=muted, spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        name="SectionLabel", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=10, textColor=blue, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=21, leading=25, textColor=navy, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CardTitle", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=navy, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9, leading=13, textColor=navy,
    ))
    styles.add(ParagraphStyle(
        name="MutedSmall", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8, leading=11, textColor=muted,
    ))
    styles.add(ParagraphStyle(
        name="StepNumber", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=colors.white, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="CodeSmall", parent=styles["Code"], fontName="Courier",
        fontSize=7, leading=10, textColor=colors.HexColor("#DCE6FF"),
    ))

    def page_decor(pdf_canvas: Any, doc: Any) -> None:
        pdf_canvas.saveState()
        pdf_canvas.setTitle(str(manual["title"]))
        pdf_canvas.setAuthor("IVI Visual Agent")
        pdf_canvas.setFillColor(muted)
        pdf_canvas.setFont("Helvetica", 7.5)
        pdf_canvas.drawString(18 * mm, 11 * mm, f"MANUAL-ID: {manual['id']}")
        pdf_canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"PAGE {doc.page}")
        pdf_canvas.restoreState()

    def fitted_image(path: str, max_width: float, max_height: float) -> Image:
        with PillowImage.open(path) as source_image:
            width, height = source_image.size
        scale = min(max_width / width, max_height / height)
        return Image(path, width=width * scale, height=height * scale)

    def bullets(values: list[str], color: str = "#172033") -> list[Paragraph]:
        return [
            Paragraph(f'<font color="{color}">-</font> {_escape(value)}', styles["BodySmall"])
            for value in values
        ]

    def section_heading(kind: str, heading: str, description: str) -> list[Any]:
        return [
            Paragraph(_escape(kind.upper()), styles["SectionLabel"]),
            Paragraph(_escape(heading), styles["SectionTitle"]),
            Paragraph(_escape(description), styles["ManualSubtitle"]),
            HRFlowable(width="100%", thickness=0.7, color=line, spaceAfter=16),
        ]

    story: list[Any] = []
    story.extend([
        Spacer(1, 30 * mm),
        Paragraph("CUSTOM IVI MANUAL", styles["SectionLabel"]),
        Paragraph(_escape(manual["title"]), styles["ManualTitle"]),
        Paragraph(_escape(manual["description"]), styles["ManualSubtitle"]),
        Spacer(1, 8 * mm),
    ])
    metadata = [
        [Paragraph("MANUAL ID", styles["SectionLabel"]), Paragraph("UI PROFILE", styles["SectionLabel"]), Paragraph("VERSION", styles["SectionLabel"])],
        [Paragraph(_escape(manual["id"]), styles["CardTitle"]), Paragraph(_escape(manual["ui_profile"]), styles["CardTitle"]), Paragraph(_escape(manual["version"]), styles["CardTitle"])],
    ]
    metadata_table = Table(metadata, colWidths=[55 * mm, 55 * mm, 42 * mm], rowHeights=[9 * mm, 17 * mm])
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale_blue),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#C7D5FF")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D5FF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([
        metadata_table,
        Spacer(1, 16 * mm),
        Paragraph("How the agent uses this manual", styles["SectionTitle"]),
        Paragraph(
            "The PDF teaches control meaning, screen transitions and success evidence. "
            "The live screenshot supplies current position. Retrieved instructions are "
            "provisional: the agent confirms every control before tapping and observes the "
            "result before continuing.",
            styles["ManualSubtitle"],
        ),
        Spacer(1, 42 * mm),
        Paragraph("Fictional example - no production vehicle UI is represented.", styles["MutedSmall"]),
        PageBreak(),
    ])

    icons = data["icons"]
    if icons:
        story.extend(section_heading(
            "Reference", "Proprietary icon glossary",
            "Each icon image is paired with a stable ID, human meaning, synonyms and the screen or action it opens.",
        ))
        cards: list[list[Any]] = []
        row: list[Any] = []
        for item in icons:
            icon_image = fitted_image(item["_image_path"], 24 * mm, 24 * mm)
            synonyms = ", ".join(item.get("synonyms", [])) or "None"
            body = [
                Paragraph(f"ICON-ID: {_escape(item['id'])}", styles["SectionLabel"]),
                Paragraph(_escape(item["name"]), styles["CardTitle"]),
                Paragraph(_escape(item["meaning"]), styles["BodySmall"]),
                Spacer(1, 3),
                Paragraph(f"<b>Synonyms:</b> {_escape(synonyms)}", styles["MutedSmall"]),
                Paragraph(f"<b>Opens:</b> {_escape(item['opens'])}", styles["MutedSmall"]),
                Paragraph(f"<b>Image:</b> {_escape(item['image'])}", styles["MutedSmall"]),
            ]
            inner = Table([[icon_image, body]], colWidths=[30 * mm, 52 * mm])
            inner.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            row.append(inner)
            if len(row) == 2:
                cards.append(row)
                row = []
        if row:
            row.append("")
            cards.append(row)
        table = Table(cards, colWidths=[85 * mm, 85 * mm], rowHeights=[54 * mm] * len(cards), hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.7, line),
            ("INNERGRID", (0, 0), (-1, -1), 0.7, line),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.extend([table, PageBreak()])

    for screen in data["screens"]:
        story.extend(section_heading(
            "Screen reference", f"{screen['id']} - {screen['name']}", screen["description"]
        ))
        screenshot = fitted_image(screen["_image_path"], 170 * mm, 92 * mm)
        image_frame = Table([[screenshot]], colWidths=[174 * mm])
        image_frame.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#101827")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#303A4D")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([
            image_frame,
            Spacer(1, 5 * mm),
            Paragraph(f"IMAGE: {_escape(screen['image'])}", styles["SectionLabel"]),
            Paragraph("Distinctive landmarks", styles["CardTitle"]),
            *bullets(screen.get("landmarks", [])),
            Spacer(1, 4 * mm),
        ])
        controls = screen.get("controls", [])
        if controls:
            rows: list[list[Any]] = [[
                Paragraph("CONTROL", styles["SectionLabel"]),
                Paragraph("ACTION", styles["SectionLabel"]),
                Paragraph("EXPECTED RESULT", styles["SectionLabel"]),
            ]]
            for control in controls:
                icon_note = f"<br/><font color='#5E687C'>icon: {_escape(control.get('icon_id', 'none'))}</font>"
                rows.append([
                    Paragraph(f"<b>{_escape(control['name'])}</b><br/>{_escape(control['id'])}{icon_note}", styles["BodySmall"]),
                    Paragraph(_escape(control["action"]), styles["BodySmall"]),
                    Paragraph(_escape(control["result"]), styles["BodySmall"]),
                ])
            control_table = Table(rows, colWidths=[52 * mm, 56 * mm, 62 * mm], repeatRows=1)
            control_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), pale_blue),
                ("BOX", (0, 0), (-1, -1), 0.7, line),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, line),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            story.append(control_table)
        story.append(PageBreak())

    for task in data["tasks"]:
        story.extend(section_heading("Task guide", task["id"], task["goal"]))
        step_rows: list[list[Any]] = []
        for number, step in enumerate(task["steps"], 1):
            step_rows.append([
                Table([[Paragraph(str(number), styles["StepNumber"])]], colWidths=[9 * mm], rowHeights=[9 * mm], style=[
                    ("BACKGROUND", (0, 0), (-1, -1), blue),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]),
                [
                    Paragraph(f"<b>{_escape(step['instruction'])}</b>", styles["BodySmall"]),
                    Paragraph(f"Screen: {_escape(step['screen'])} | Target: {_escape(step['target'])}", styles["MutedSmall"]),
                    Paragraph(f"Expected: {_escape(step['expected'])}", styles["MutedSmall"]),
                ],
            ])
        steps_table = Table(step_rows, colWidths=[14 * mm, 154 * mm])
        steps_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOX", (0, 0), (-1, -1), 0.7, line),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, line),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        story.extend([steps_table, Spacer(1, 7 * mm)])
        success = [Paragraph("PASS WHEN", styles["SectionLabel"]), *bullets(task["success"], "#35B77A")]
        forbidden = [Paragraph("DO NOT", styles["SectionLabel"]), *bullets(task["forbidden"], "#E85B68")]
        evidence_table = Table([[success, forbidden]], colWidths=[84 * mm, 84 * mm])
        evidence_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), pale_green),
            ("BACKGROUND", (1, 0), (1, 0), pale_red),
            ("BOX", (0, 0), (-1, -1), 0.7, line),
            ("INNERGRID", (0, 0), (-1, -1), 0.7, line),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        summary = {
            "TASK_ID": task["id"],
            "GOAL": task["goal"],
            "TARGETS": [step["target"] for step in task["steps"]],
            "SUCCESS": task["success"],
            "FORBIDDEN": task["forbidden"],
        }
        code = json.dumps(summary, indent=2, ensure_ascii=True)
        code_table = Table([[Preformatted(code, styles["CodeSmall"])]], colWidths=[168 * mm])
        code_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#151B29")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#151B29")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.extend([
            evidence_table,
            Spacer(1, 7 * mm),
            Paragraph("Machine-readable retrieval summary", styles["CardTitle"]),
            code_table,
            PageBreak(),
        ])

    if story and isinstance(story[-1], PageBreak):
        story.pop()
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=str(manual["title"]), author="IVI Visual Agent",
    )
    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return {
        "manual_id": manual["id"],
        "source": data["_source"],
        "output": str(output),
        "icons": len(data["icons"]),
        "screens": len(data["screens"]),
        "tasks": len(data["tasks"]),
    }
