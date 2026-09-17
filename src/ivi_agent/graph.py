"""Scene graph: an expected model from the manual that grows as the agent runs.

The graph is seeded from the shared PDF/manual (screens become nodes, documented
``control.result`` transitions become edges — the *expected* model). As the agent
(or the OpenCV fast-path) reaches screens at runtime, each observation is matched
against the graph:

* a reached screen that matches a documented node **confirms** that node and the
  transition that led to it;
* a reached screen that matches **no** documented node becomes a
  ``pending_review`` node and raises an ``undocumented_screen`` finding;
* a transition between two documented screens that the manual never described
  raises an ``undocumented_transition`` finding.

Findings queue for user approval: approve (legitimate, fold into the model) or
mark as a defect (the live HMI diverges from its specification). The graph is a
living model, persisted per knowledge profile and grown across runs.

This module has no device or model dependency, so it is fully unit-testable.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

TOKEN = re.compile(r"[a-z0-9]+")

# A reached screen matches a node when their name/landmark tokens overlap at least
# this much (Jaccard), or when a perceptual hash is within PHASH_MAX_DISTANCE.
TITLE_MATCH_MIN = 0.5
PHASH_MAX_DISTANCE = 6


def _tokens(*values: Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            for item in value:
                out |= _tokens(item)
        elif value:
            out.update(TOKEN.findall(str(value).lower()))
    return out


def _match_tokens(values: Any) -> set[str]:
    """Tokens usable for screen identity: drop single chars and pure numbers.

    Status-bar clutter in a live title ("Driver 22.0°C", "A/c VEH") tokenizes to
    noise like {driver, 22, 0, c} that would otherwise drown out the screen name.
    """
    return {t for t in _tokens(values) if len(t) >= 2 and not t.isdigit()}


def _hamming(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


@dataclass
class Node:
    id: str
    name: str
    source: str  # "manual" | "observed"
    status: str  # expected | confirmed | pending_review | approved | defect
    tokens: list[str] = field(default_factory=list)
    phashes: list[int] = field(default_factory=list)
    sample_screenshot: str = ""
    times_seen: int = 0
    first_run: str = ""
    last_run: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    via_control: str = ""
    via_action: str = ""
    source: str = "manual"  # "manual" | "observed"
    status: str = "expected"  # expected | confirmed | undocumented
    times_seen: int = 0


@dataclass
class Finding:
    id: str
    kind: str  # undocumented_screen | undocumented_transition | unreachable_screen
    detail: str
    status: str = "pending"  # pending | approved | defect
    node_id: str = ""
    src: str = ""
    dst: str = ""
    run_id: str = ""


@dataclass
class ObserveResult:
    node_id: str
    matched: bool
    is_new_node: bool
    finding: Finding | None = None


class SceneGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str], Edge] = {}
        self.findings: list[Finding] = []
        self._obs_counter = 0
        self._finding_counter = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_manual(cls, manual: dict[str, Any]) -> "SceneGraph":
        """Seed the expected graph from a manual dict (screens + controls)."""
        graph = cls()
        for screen in manual.get("screens", []):
            if not isinstance(screen, dict):
                continue
            sid = str(screen.get("id") or screen.get("name") or "")
            if not sid:
                continue
            graph.nodes[sid] = Node(
                id=sid,
                name=str(screen.get("name", sid)),
                source="manual",
                status="expected",
                # Match tokens come from the screen NAME (plus observed titles as
                # they accumulate), not landmarks -- landmarks name other screens'
                # controls and would cross-match.
                tokens=sorted(_tokens(screen.get("name"))),
            )
        # Documented transitions: a control whose result is another screen.
        for screen in manual.get("screens", []):
            if not isinstance(screen, dict):
                continue
            sid = str(screen.get("id") or screen.get("name") or "")
            for control in screen.get("controls", []) or []:
                if not isinstance(control, dict):
                    continue
                dst = str(control.get("result", ""))
                if dst and dst in graph.nodes and dst != sid:
                    graph.edges[(sid, dst)] = Edge(
                        src=sid,
                        dst=dst,
                        via_control=str(control.get("id", control.get("name", ""))),
                        via_action=str(control.get("action", "")),
                        source="manual",
                        status="expected",
                    )
        return graph

    # -- matching ----------------------------------------------------------
    def match(
        self,
        titles: list[str],
        phash: int | None = None,
        similarity: Callable[[str], float] | None = None,
    ) -> str | None:
        """Return the id of the node this observation matches, or None.

        Matches on name/landmark token overlap first, then perceptual-hash
        proximity (for title-less screens). ``similarity`` is an optional hook
        (e.g. CV screen match by node id) reserved for future use.
        """
        observed = _match_tokens(titles)
        best_id, best_score, best_tie = None, 0.0, 0.0
        for node in self.nodes.values():
            node_tokens = _match_tokens(node.tokens)
            if observed and node_tokens:
                shared = len(observed & node_tokens)
                if shared == 0:
                    continue
                # Score by the BETTER of the two directional coverages, so a
                # verbose manual name ("Home - tile grid" vs on-screen "Home")
                # and a noisy observed title (status-bar clutter) both still
                # match. Tie-break by Jaccard for the tightest fit.
                score = max(shared / len(observed), shared / len(node_tokens))
                tie = shared / len(observed | node_tokens)
                if score > best_score or (score == best_score and tie > best_tie):
                    best_id, best_score, best_tie = node.id, score, tie
        if best_id is not None and best_score >= TITLE_MATCH_MIN:
            return best_id
        # Perceptual-hash fallback ONLY for title-less screens. A screen that has
        # a readable title but matches no node is a genuinely new screen (a
        # candidate defect), so it must not be silently bucketed by pixels.
        if not observed and phash is not None:
            for node in self.nodes.values():
                if any(_hamming(phash, seen) <= PHASH_MAX_DISTANCE for seen in node.phashes):
                    return node.id
        return None

    # -- growth ------------------------------------------------------------
    def observe(
        self,
        titles: list[str],
        phash: int | None = None,
        screenshot: str = "",
        run_id: str = "",
        came_from: str | None = None,
        via_action: str = "",
        via_target: str = "",
    ) -> ObserveResult:
        """Record a reached screen; grow the graph and raise findings on divergence."""
        node_id = self.match(titles, phash)
        finding: Finding | None = None
        is_new = False
        if node_id is None:
            self._obs_counter += 1
            slug = "-".join(sorted(_tokens(titles))[:3]) or f"n{self._obs_counter}"
            node_id = f"observed.{slug}.{self._obs_counter}"
            name = titles[0] if titles else f"Unknown screen {self._obs_counter}"
            self.nodes[node_id] = Node(
                id=node_id,
                name=name,
                source="observed",
                status="pending_review",
                tokens=sorted(_tokens(titles)),
            )
            is_new = True
            finding = self._add_finding(
                "undocumented_screen",
                f"Reached a screen not described in the manual: {name!r} "
                f"(titles={titles})",
                node_id=node_id,
                run_id=run_id,
            )
        node = self.nodes[node_id]
        if node.source == "manual" and node.status == "expected":
            node.status = "confirmed"
        node.times_seen += 1
        node.last_run = run_id or node.last_run
        node.first_run = node.first_run or run_id
        if screenshot:
            node.sample_screenshot = node.sample_screenshot or screenshot
        for token in _tokens(titles):
            if token not in node.tokens:
                node.tokens.append(token)
        if phash is not None and phash not in node.phashes:
            node.phashes.append(phash)

        if came_from and came_from in self.nodes and came_from != node_id:
            key = (came_from, node_id)
            edge = self.edges.get(key)
            if edge is None:
                edge = Edge(
                    src=came_from,
                    dst=node_id,
                    via_control=via_target,
                    via_action=via_action,
                    source="observed",
                    status="undocumented",
                )
                self.edges[key] = edge
                # Only flag as a divergence when BOTH ends are documented screens
                # (an undocumented path between known screens); transitions that
                # merely involve a new observed screen are covered by that node's
                # own finding.
                if (
                    self.nodes[came_from].source == "manual"
                    and node.source == "manual"
                ):
                    finding = finding or self._add_finding(
                        "undocumented_transition",
                        f"Transition {came_from} -> {node_id} via "
                        f"{via_target or via_action!r} is not documented in the manual.",
                        src=came_from,
                        dst=node_id,
                        run_id=run_id,
                    )
            else:
                if edge.source == "manual":
                    edge.status = "confirmed"
            edge.times_seen += 1

        return ObserveResult(node_id=node_id, matched=not is_new, is_new_node=is_new, finding=finding)

    def _add_finding(self, kind: str, detail: str, **fields: Any) -> Finding:
        self._finding_counter += 1
        finding = Finding(id=f"F{self._finding_counter:04d}", kind=kind, detail=detail, **fields)
        self.findings.append(finding)
        return finding

    # -- review ------------------------------------------------------------
    def pending_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "pending"]

    def review(self, finding_id: str, decision: str) -> Finding:
        """Resolve a finding. decision: "approve" (legitimate) or "defect"."""
        if decision not in {"approve", "defect"}:
            raise ValueError("decision must be 'approve' or 'defect'")
        finding = next((f for f in self.findings if f.id == finding_id), None)
        if finding is None:
            raise KeyError(f"no such finding: {finding_id}")
        finding.status = "approved" if decision == "approve" else "defect"
        if finding.node_id and finding.node_id in self.nodes:
            self.nodes[finding.node_id].status = (
                "approved" if decision == "approve" else "defect"
            )
        return finding

    # -- reporting ---------------------------------------------------------
    def coverage(self) -> dict[str, Any]:
        manual = [n for n in self.nodes.values() if n.source == "manual"]
        confirmed = [n for n in manual if n.status == "confirmed"]
        unreached = [n for n in manual if n.status == "expected"]
        observed = [n for n in self.nodes.values() if n.source == "observed"]
        return {
            "manual_screens": len(manual),
            "confirmed_screens": len(confirmed),
            "unreached_screens": [n.id for n in unreached],
            "observed_new_screens": len([n for n in observed if n.status != "defect"]),
            "defects": len([n for n in self.nodes.values() if n.status == "defect"]),
            "pending_findings": len(self.pending_findings()),
        }

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "findings": [asdict(f) for f in self.findings],
            "counters": {"obs": self._obs_counter, "finding": self._finding_counter},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneGraph":
        graph = cls()
        for raw in data.get("nodes", []):
            node = Node(**raw)
            graph.nodes[node.id] = node
        for raw in data.get("edges", []):
            edge = Edge(**raw)
            graph.edges[(edge.src, edge.dst)] = edge
        for raw in data.get("findings", []):
            graph.findings.append(Finding(**raw))
        counters = data.get("counters", {})
        graph._obs_counter = int(counters.get("obs", 0))
        graph._finding_counter = int(counters.get("finding", len(graph.findings)))
        return graph

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SceneGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_or_seed(cls, path: Path, manual: dict[str, Any] | None) -> "SceneGraph":
        """Load a persisted living graph, or seed a fresh one from the manual."""
        if Path(path).is_file():
            return cls.load(path)
        return cls.from_manual(manual or {})
