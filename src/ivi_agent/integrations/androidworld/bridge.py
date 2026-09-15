"""Pure translation helpers between AndroidWorld and the IVI visual agent.

These functions deliberately avoid importing ``android_world`` so they can be
unit-tested without the benchmark (or an emulator) installed. They operate on
duck-typed objects that expose the same attributes as
``android_world.env.representation_utils.UIElement`` and
``android_world.env.json_action.JSONAction``.

Two coordinate spaces are involved:

* The IVI agent speaks in **normalized** coordinates in ``[0, 1]`` (fractions of
  the screen), which are resolution independent.
* AndroidWorld's :class:`JSONAction` clicks use **absolute logical pixels**.

``screen_size`` throughout is the environment's ``logical_screen_size``:
``(width, height)`` in pixels.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from typing import Any


# IVI gesture direction -> AndroidWorld scroll direction. Both name the content
# that should be revealed, so the mapping is 1:1 (verified against
# android_world/env/actuation.py).
GESTURE_TO_SCROLL = {
    "reveal_above": "up",
    "reveal_below": "down",
    "reveal_left": "left",
    "reveal_right": "right",
}


def pixels_to_png(pixels: Any) -> bytes:
    """Encode an AndroidWorld RGB(A) screenshot array as PNG bytes."""
    from PIL import Image  # local import keeps module import cheap

    image = Image.fromarray(pixels)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _bbox(element: Any) -> tuple[int, int, int, int] | None:
    box = getattr(element, "bbox_pixels", None) or getattr(element, "bbox", None)
    if box is None:
        return None
    try:
        left, top = int(box.x_min), int(box.y_min)
        right, bottom = int(box.x_max), int(box.y_max)
    except (AttributeError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def ui_elements_to_uiautomator_xml(
    ui_elements: list[Any], screen_size: tuple[int, int]
) -> str:
    """Render AndroidWorld UI elements as uiautomator-style XML.

    The IVI agent's perception layer (``perception.extract_ui_elements`` and
    friends) parses the exact XML that ``uiautomator dump`` produces, so the
    cleanest bridge is to synthesize that XML rather than re-implement parsing.
    A full-screen frame node is included so the perception layer infers the
    correct screen dimensions for center normalization.
    """
    width, height = screen_size
    hierarchy = ET.Element("hierarchy", {"rotation": "0"})
    frame = ET.SubElement(
        hierarchy,
        "node",
        {
            "class": "android.widget.FrameLayout",
            "bounds": f"[0,0][{int(width)},{int(height)}]",
        },
    )
    for element in ui_elements:
        bounds = _bbox(element)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        text = getattr(element, "text", None) or ""
        content_desc = getattr(element, "content_description", None) or ""
        class_name = getattr(element, "class_name", None) or "android.view.View"
        resource_id = (
            getattr(element, "resource_id", None)
            or getattr(element, "resource_name", None)
            or ""
        )
        package = getattr(element, "package_name", None) or ""

        def flag(name: str) -> str:
            return "true" if getattr(element, name, None) else "false"

        ET.SubElement(
            frame,
            "node",
            {
                "text": text,
                "content-desc": content_desc,
                "resource-id": resource_id,
                "class": class_name,
                "package": package,
                "clickable": flag("is_clickable"),
                "long-clickable": flag("is_long_clickable"),
                "checkable": flag("is_checkable"),
                "checked": flag("is_checked"),
                "scrollable": flag("is_scrollable"),
                "focused": flag("is_focused"),
                "bounds": f"[{left},{top}][{right},{bottom}]",
            },
        )
    return ET.tostring(hierarchy, encoding="unicode")


def _to_pixel(value: float | None, extent: int) -> int:
    if value is None:
        raise ValueError("action is missing a coordinate for AndroidWorld")
    return int(round(max(0.0, min(1.0, value)) * (extent - 1)))


def ivi_action_to_json_action_kwargs(
    action: Any, screen_size: tuple[int, int]
) -> dict[str, Any]:
    """Translate an IVI :class:`~ivi_agent.types.Action` to JSONAction kwargs.

    Returns a plain ``dict`` (not a ``JSONAction``) so this stays importable
    without ``android_world``. The adapter constructs ``JSONAction(**kwargs)``.
    """
    width, height = screen_size
    kind = action.type

    if kind in ("tap", "double_tap", "long_press"):
        mapping = {"tap": "click", "double_tap": "double_tap", "long_press": "long_press"}
        return {
            "action_type": mapping[kind],
            "x": _to_pixel(action.x, width),
            "y": _to_pixel(action.y, height),
        }
    if kind == "input_text":
        return {
            "action_type": "input_text",
            "text": action.text,
            "x": _to_pixel(action.x, width),
            "y": _to_pixel(action.y, height),
            "clear_text": True,
        }
    if kind == "text":
        return {"action_type": "input_text", "text": action.text}
    if kind == "keyboard_enter":
        return {"action_type": "keyboard_enter"}
    if kind == "gesture":
        direction = GESTURE_TO_SCROLL.get(action.direction)
        if direction is None:
            raise ValueError(f"unsupported gesture direction: {action.direction}")
        return {"action_type": "scroll", "direction": direction}
    if kind == "swipe":
        # AndroidWorld's swipe is coarse (whole-screen, fixed direction); map to
        # the dominant axis of the requested swipe so paging still works.
        dx = (action.x2 or 0.0) - (action.x or 0.0)
        dy = (action.y2 or 0.0) - (action.y or 0.0)
        if abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        return {"action_type": "swipe", "direction": direction}
    if kind == "open_app":
        return {"action_type": "open_app", "app_name": action.app_name or action.target}
    if kind == "back":
        return {"action_type": "navigate_back"}
    if kind == "home":
        return {"action_type": "navigate_home"}
    if kind == "wait":
        return {"action_type": "wait"}
    if kind == "finish":
        status = "complete" if action.outcome == "pass" else "infeasible"
        return {"action_type": "status", "goal_status": status}
    raise ValueError(f"cannot translate action type to AndroidWorld: {kind}")
