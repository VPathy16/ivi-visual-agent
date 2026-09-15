#!/usr/bin/env python3
"""Create fictional images for examples/custom-ivi-manual."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#101827"
CARD = "#222E43"
TEXT = "#F7F9FC"
MUTED = "#9EABC1"
BLUE = "#3564FF"
CYAN = "#43C7D6"
GREEN = "#35B77A"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def icon(kind: str, color: str, size: int = 180) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    width = max(5, size // 24)
    cx = cy = size // 2
    if kind == "audio_hub":
        for offset, height in ((-38, 72), (0, 122), (38, 90)):
            draw.line((cx + offset, cy - height // 2, cx + offset, cy + height // 2), fill=color, width=width)
            radius = width
            draw.ellipse((cx + offset - radius, cy - height // 2 - radius,
                          cx + offset + radius, cy - height // 2 + radius), fill=color)
            draw.ellipse((cx + offset - radius, cy + height // 2 - radius,
                          cx + offset + radius, cy + height // 2 + radius), fill=color)
    elif kind == "sources":
        nodes = [(cx, cy), (cx - 55, cy - 42), (cx + 55, cy - 42), (cx, cy + 62)]
        for endpoint in nodes[1:]:
            draw.line((nodes[0], endpoint), fill=color, width=width)
        for index, (x, y) in enumerate(nodes):
            radius = 14 if index == 0 else 12
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=color, width=width)
    elif kind == "linkwave":
        draw.ellipse((28, cy - 11, 50, cy + 11), fill=color)
        for radius in (34, 58, 82):
            draw.arc((44, cy - radius, 44 + radius * 2, cy + radius),
                     start=-42, end=42, fill=color, width=width)
    return image


def save_icon_assets(directory: Path) -> dict[str, Image.Image]:
    directory.mkdir(parents=True, exist_ok=True)
    icons = {
        "audio_hub": icon("audio_hub", BLUE),
        "sources": icon("sources", CYAN),
        "linkwave": icon("linkwave", GREEN),
    }
    filenames = {
        "audio_hub": "icon-audio-hub.png",
        "sources": "icon-sources.png",
        "linkwave": "icon-linkwave.png",
    }
    for key, image in icons.items():
        image.save(directory / filenames[key])
    return icons


def base_screen(title_text: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1280, 720), BG)
    draw = ImageDraw.Draw(image)
    draw.text((42, 30), title_text, fill=TEXT, font=font(30, True))
    draw.text((1050, 35), "10:24  Driver", fill=MUTED, font=font(18))
    draw.line((38, 82, 1242, 82), fill="#2E3A50", width=2)
    return image, draw


def paste_icon(screen: Image.Image, asset: Image.Image, box: tuple[int, int, int, int]) -> None:
    width = box[2] - box[0]
    height = box[3] - box[1]
    copy = asset.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = box[0] + (width - copy.width) // 2
    y = box[1] + (height - copy.height) // 2
    screen.paste(copy, (x, y), copy)


def create_home(directory: Path, icons: dict[str, Image.Image]) -> None:
    image, draw = base_screen("Home")
    tiles = [
        ("Audio", "audio_hub", BLUE),
        ("Phone", "sources", CYAN),
        ("Navigation", "linkwave", "#F4B740"),
        ("Vehicle", "audio_hub", GREEN),
        ("Recent", "sources", "#E85B68"),
        ("Connectivity", "linkwave", "#9B7BFF"),
    ]
    for index, (name, icon_id, accent) in enumerate(tiles):
        col, row = index % 3, index // 3
        x = 54 + col * 407
        y = 118 + row * 254
        outline = accent if index == 0 else "#344159"
        draw.rounded_rectangle((x, y, x + 365, y + 210), radius=24,
                               fill=CARD, outline=outline, width=5 if index == 0 else 2)
        paste_icon(image, icons[icon_id], (x + 24, y + 42, x + 150, y + 168))
        draw.text((x + 165, y + 72), name, fill=TEXT, font=font(25, True))
        draw.text((x + 165, y + 118), "Tap to open", fill=MUTED, font=font(18))
    image.save(directory / "home.png")


def create_audio_hub(directory: Path, icons: dict[str, Image.Image]) -> None:
    image, draw = base_screen("Audio Hub")
    draw.rounded_rectangle((55, 125, 755, 650), radius=28, fill=CARD)
    draw.text((95, 168), "Now playing", fill=TEXT, font=font(38, True))
    draw.text((95, 230), "Source: Broadcast", fill=MUTED, font=font(25))
    paste_icon(image, icons["audio_hub"], (260, 300, 550, 590))
    draw.rounded_rectangle((820, 150, 1215, 290), radius=24,
                           fill="#1D3544", outline=CYAN, width=4)
    paste_icon(image, icons["sources"], (842, 173, 950, 267))
    draw.text((970, 195), "Sources", fill=TEXT, font=font(30, True))
    draw.text((820, 330), "Opens available media inputs", fill=MUTED, font=font(20))
    image.save(directory / "audio-hub.png")


def create_bt_selected(directory: Path, icons: dict[str, Image.Image]) -> None:
    image, draw = base_screen("Audio Hub")
    draw.rounded_rectangle((55, 125, 700, 650), radius=28, fill=CARD)
    draw.text((95, 168), "Now playing", fill=TEXT, font=font(38, True))
    draw.text((95, 230), "Source: BT Audio", fill=GREEN, font=font(28, True))
    paste_icon(image, icons["audio_hub"], (235, 305, 520, 590))
    draw.text((770, 130), "Sources", fill=TEXT, font=font(30, True))
    rows = [("Broadcast", "sources", MUTED), ("USB", "sources", MUTED),
            ("BT Audio", "linkwave", GREEN)]
    for index, (name, icon_id, accent) in enumerate(rows):
        y = 190 + index * 135
        selected = index == 2
        draw.rounded_rectangle((755, y, 1215, y + 105), radius=20,
                               fill="#1D3437" if selected else CARD,
                               outline=GREEN if selected else "#344159",
                               width=4 if selected else 2)
        paste_icon(image, icons[icon_id], (775, y + 10, 865, y + 95))
        draw.text((885, y + 34), name, fill=TEXT, font=font(25, selected))
        if selected:
            draw.ellipse((1140, y + 25, 1195, y + 80), fill=GREEN)
            draw.text((1157, y + 30), "v", fill=TEXT, font=font(27, True))
    image.save(directory / "bt-selected.png")


def create_sources_open(directory: Path, icons: dict[str, Image.Image]) -> None:
    image, draw = base_screen("Audio Hub")
    draw.rounded_rectangle((55, 125, 700, 650), radius=28, fill=CARD)
    draw.text((95, 168), "Now playing", fill=TEXT, font=font(38, True))
    draw.text((95, 230), "Source: Broadcast", fill=MUTED, font=font(28))
    paste_icon(image, icons["audio_hub"], (235, 305, 520, 590))
    draw.text((770, 130), "Sources", fill=TEXT, font=font(30, True))
    rows = [("Broadcast", "sources", CYAN), ("USB", "sources", MUTED),
            ("BT Audio", "linkwave", GREEN)]
    for index, (name, icon_id, accent) in enumerate(rows):
        y = 190 + index * 135
        selected = index == 0
        draw.rounded_rectangle((755, y, 1215, y + 105), radius=20,
                               fill="#1D3544" if selected else CARD,
                               outline=CYAN if selected else "#344159",
                               width=4 if selected else 2)
        paste_icon(image, icons[icon_id], (775, y + 10, 865, y + 95))
        draw.text((885, y + 34), name, fill=TEXT, font=font(25, selected))
        if selected:
            draw.ellipse((1140, y + 25, 1195, y + 80), fill=CYAN)
            draw.text((1157, y + 30), "v", fill=TEXT, font=font(27, True))
    image.save(directory / "sources-open.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/custom-ivi-manual/images"),
    )
    args = parser.parse_args()
    icons = save_icon_assets(args.output)
    create_home(args.output, icons)
    create_audio_hub(args.output, icons)
    create_sources_open(args.output, icons)
    create_bt_selected(args.output, icons)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
