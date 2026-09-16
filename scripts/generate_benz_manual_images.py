#!/usr/bin/env python3
"""Create PLACEHOLDER images for examples/benz-mbux-manual.

These are labeled stand-ins so the scaffold builds and indexes out of the box.
Replace every file with real screenshots / icon crops from the target unit
before using the profile against a device.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = "#0E1116"
CARD = "#1C2530"
TEXT = "#F5F7FA"
MUTED = "#93A1B5"
ACCENTS = {
    "climate": "#3AA0FF",
    "seat_massage": "#C77DFF",
    "vehicle_settings": "#43C7A6",
}


def font(size: int, bold: bool = False):
    for name in (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def icon(label: str, color: str, size: int = 180) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 10, size - 10, size - 10), radius=28, outline=color, width=8)
    draw.text((size // 2, size // 2), label, fill=color, font=font(46, True), anchor="mm")
    return image


def base_screen(title: str):
    image = Image.new("RGB", (1280, 720), BG)
    draw = ImageDraw.Draw(image)
    draw.text((42, 30), title, fill=TEXT, font=font(32, True))
    draw.text((1040, 36), "PLACEHOLDER", fill=MUTED, font=font(18))
    draw.line((38, 84, 1242, 84), fill="#2C3542", width=2)
    return image, draw


def tile(draw, x, y, name, accent):
    draw.rounded_rectangle((x, y, x + 360, y + 200), radius=22, fill=CARD, outline=accent, width=3)
    draw.text((x + 26, y + 80), name, fill=TEXT, font=font(28, True))
    draw.text((x + 26, y + 128), "Tap to open", fill=MUTED, font=font(18))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/benz-mbux-manual/images"))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    icon("A/C", ACCENTS["climate"]).save(out / "icon-climate.png")
    icon("~M~", ACCENTS["seat_massage"]).save(out / "icon-seat-massage.png")
    icon("VEH", ACCENTS["vehicle_settings"]).save(out / "icon-vehicle-settings.png")

    home, draw = base_screen("Home")
    tile(draw, 54, 120, "Climate", ACCENTS["climate"])
    tile(draw, 460, 120, "Seat Comfort", ACCENTS["seat_massage"])
    tile(draw, 866, 120, "Vehicle", ACCENTS["vehicle_settings"])
    home.save(out / "home.png")

    veh, draw = base_screen("Vehicle settings")
    for i, row in enumerate(["Driving modes", "Lighting", "Seat comfort", "Displays"]):
        draw.rounded_rectangle((54, 120 + i * 120, 1226, 210 + i * 120), radius=18, fill=CARD)
        draw.text((80, 150 + i * 120), row, fill=TEXT, font=font(26, True))
    veh.save(out / "vehicle-settings.png")

    clim, draw = base_screen("Climate")
    draw.text((80, 150), "Driver  22.0°C", fill=TEXT, font=font(30, True))
    draw.rounded_rectangle((80, 220, 1200, 260), radius=20, fill="#2C3542")
    draw.rounded_rectangle((80, 220, 640, 260), radius=20, fill=ACCENTS["climate"])
    draw.text((80, 300), "Fan   Zones   Auto   A/C", fill=MUTED, font=font(24))
    clim.save(out / "climate.png")

    seat, draw = base_screen("Seat massage")
    for i, prog in enumerate(["Wave", "Lumbar", "Shoulder"]):
        sel = i == 0
        draw.rounded_rectangle(
            (80 + i * 380, 150, 420 + i * 380, 300),
            radius=18,
            fill="#243244" if sel else CARD,
            outline=ACCENTS["seat_massage"] if sel else "#2C3542",
            width=4 if sel else 2,
        )
        draw.text((110 + i * 380, 210), prog, fill=TEXT, font=font(26, True))
    draw.text((80, 360), "Intensity", fill=TEXT, font=font(24, True))
    draw.rounded_rectangle((80, 410, 1200, 450), radius=20, fill="#2C3542")
    draw.rounded_rectangle((80, 410, 500, 450), radius=20, fill=ACCENTS["seat_massage"])
    draw.rounded_rectangle((1040, 520, 1200, 600), radius=18, fill=ACCENTS["seat_massage"])
    draw.text((1120, 560), "Start", fill=TEXT, font=font(24, True), anchor="mm")
    seat.save(out / "seat-massage.png")

    print(out.resolve())


if __name__ == "__main__":
    main()
