from __future__ import annotations

import re

from .config import Config
from .types import Action


class PolicyViolation(ValueError):
    pass


def validate_action(action: Action, config: Config, goal: str = "") -> None:
    valid = {
        "tap",
        "input_text",
        "gesture",
        "swipe",
        "back",
        "home",
        "wait",
        "text",
        "finish",
    }
    if action.type not in valid:
        raise PolicyViolation(f"Unsupported action type: {action.type}")
    if not 0.0 <= action.confidence <= 1.0:
        raise PolicyViolation("Confidence must be between 0 and 1")
    if action.type != "finish" and action.confidence < config.minimum_action_confidence:
        raise PolicyViolation(
            f"Action confidence {action.confidence:.2f} is below "
            f"{config.minimum_action_confidence:.2f}"
        )
    if action.type in {"tap", "input_text", "swipe"}:
        coordinates = [action.x, action.y]
        if action.type == "swipe":
            coordinates += [action.x2, action.y2]
        if any(value is None or not 0.0 <= value <= 1.0 for value in coordinates):
            raise PolicyViolation("Visual action coordinates must be normalized between 0 and 1")
    if action.type == "gesture":
        if action.direction not in {
            "reveal_above",
            "reveal_below",
            "reveal_left",
            "reveal_right",
        }:
            raise PolicyViolation(
                "Gesture direction must be reveal_above, reveal_below, reveal_left, or reveal_right"
            )
        if action.region not in {"center", "top", "bottom", "left", "right"}:
            raise PolicyViolation("Gesture region is invalid")
    if action.type in {"text", "input_text"} and not config.allow_text_input:
        raise PolicyViolation("Text input is disabled by policy")
    if action.type == "input_text" and not action.text:
        raise PolicyViolation("Input text cannot be empty")
    if action.type == "finish" and action.outcome not in {"pass", "fail", "inconclusive"}:
        raise PolicyViolation("Finish action must contain a valid outcome")
    if action.type in {"tap", "input_text"}:
        normalized_target = " ".join(
            re.findall(r"[a-z0-9]+", action.target.lower())
        )
        normalized_goal = " ".join(re.findall(r"[a-z0-9]+", goal.lower()))
        permission_action = bool(
            re.search(
                r"\b(?:grant(?: permission)?|allow(?: access| permission)?|"
                r"while using|only this time)\b",
                normalized_target,
            )
        )
        permission_goal = bool(
            re.search(r"\b(?:allow|grant|permission)\b", normalized_goal)
        )
        if permission_action and not permission_goal:
            raise PolicyViolation(
                "Permission-changing actions require an explicit permission goal"
            )
        if re.search(
            r"\b(?:factory reset|erase|delete|uninstall|purchase|buy|"
            r"place order|dial|call)\b",
            normalized_target,
        ):
            raise PolicyViolation(f"Destructive or external action is blocked: {action.target}")
        assert action.x is not None and action.y is not None
        for region in config.protected_regions or []:
            if len(region) == 4 and region[0] <= action.x <= region[2] and region[1] <= action.y <= region[3]:
                raise PolicyViolation(f"Tap target is inside protected region {region}")
