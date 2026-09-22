"""Normalise a Windows UI-Automation tree into uiautomator-style XML.

Appium's Windows driver returns the UIA tree via ``driver.page_source`` as XML
whose element tags are control types (``Window``, ``Button``, ``Text``, ``Edit``,
...) with attributes like ``Name``, ``AutomationId``, ``ClassName`` and geometry
(``x``/``y``/``width``/``height``). The shared perception layer, however, expects
Android uiautomator XML — ``<node>`` elements with ``bounds='[x1,y1][x2,y2]'``,
``text``, ``content-desc``, ``resource-id``, ``class`` and ``clickable``.

``uia_to_uiautomator`` maps one to the other, preserving nesting (so the root
window supplies the full-screen bounds the perception heuristics rely on). Get
this mapping right and ``extract_ui_elements`` / ``resolve_target_in_tree`` /
the scene graph read Windows screens exactly like Android ones.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

# Control types that are actionable — WinAppDriver's page_source doesn't reliably
# expose an "invokable" flag, so we mark clickability from the control type (and
# keyboard-focusability when present).
_CLICKABLE_TYPES = {
    "Button", "SplitButton", "MenuItem", "ListItem", "TreeItem", "TabItem",
    "CheckBox", "RadioButton", "ComboBox", "Hyperlink", "Slider", "ToggleButton",
    "Tab", "Menu", "Spinner",
}
_EDIT_TYPES = {"Edit", "Document", "ComboBox"}


def _geometry(attrib: dict[str, str]) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) from a UIA element's attributes, or None."""
    try:
        if all(key in attrib for key in ("x", "y", "width", "height")):
            x = int(float(attrib["x"]))
            y = int(float(attrib["y"]))
            w = int(float(attrib["width"]))
            h = int(float(attrib["height"]))
            if w > 0 and h > 0:
                return (x, y, x + w, y + h)
    except (ValueError, TypeError):
        pass
    # Fallback: a BoundingRectangle attribute as "left,top,width,height".
    raw = attrib.get("BoundingRectangle") or attrib.get("boundingRectangle")
    if raw:
        parts = [p for p in re.split(r"[,\s]+", raw.strip()) if p]
        try:
            nums = [int(float(p)) for p in parts]
        except ValueError:
            nums = []
        if len(nums) == 4:
            left, top, w, h = nums
            if w > 0 and h > 0:
                return (left, top, left + w, top + h)
    return None


def _convert(elem: ET.Element) -> ET.Element:
    node = ET.Element("node")
    attrib = elem.attrib
    control_type = elem.tag
    node.set("text", attrib.get("Name", ""))
    node.set("content-desc", attrib.get("HelpText", ""))
    node.set("resource-id", attrib.get("AutomationId", ""))
    # class doubles as the widget role for the perception layer (EditText ->
    # editable); map UIA edit-like types onto that convention.
    node.set("class", "EditText" if control_type in _EDIT_TYPES else control_type)
    geometry = _geometry(attrib)
    if geometry:
        node.set("bounds", "[{},{}][{},{}]".format(*geometry))
    clickable = (
        control_type in _CLICKABLE_TYPES
        or attrib.get("IsKeyboardFocusable", "").lower() == "true"
    )
    node.set("clickable", "true" if clickable else "false")
    node.set("enabled", attrib.get("IsEnabled", "true").lower())
    node.set("focused", attrib.get("HasKeyboardFocus", "false").lower())
    node.set("package", attrib.get("ProcessId", ""))
    for child in list(elem):
        node.append(_convert(child))
    return node


def uia_to_uiautomator(page_source: str) -> str:
    """Convert an Appium/WinAppDriver ``page_source`` into uiautomator XML.

    Returns "" for empty or unparseable input (the agent then treats the screen
    as tree-less and falls back to CV/vision, exactly as on Android).
    """
    if not page_source or not page_source.strip():
        return ""
    try:
        root = ET.fromstring(page_source)
    except ET.ParseError:
        return ""
    hierarchy = ET.Element("hierarchy")
    # page_source is often wrapped in a synthetic <Root>; unwrap it so the top
    # window becomes the root node (and its bounds set the screen size).
    tops = list(root) if root.tag.lower() in {"root", "xml"} else [root]
    for top in tops or [root]:
        hierarchy.append(_convert(top))
    return ET.tostring(hierarchy, encoding="unicode")
