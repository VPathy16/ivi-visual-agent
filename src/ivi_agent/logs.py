"""Crash / ANR detection from logcat, for HMI validation.

A flow can look visually correct while the app crashed or hung underneath, so a
validation run captures logcat and scans it for fatal events. This module is a
pure text scanner (no adb/device dependency) so it is fully unit-testable; the
agent supplies the logcat text and an optional package to attribute events to.

Noise handling matters on emulators: `screencap`, `binder:*`, and HAL processes
(``android.hardware.*``, the graphics allocator, audio) routinely abort with
SIGABRT — infrastructure churn, not app defects (and partly provoked by the
agent's own screenshotting). Native crashes/process-deaths from those are
filtered out; Java crashes and ANRs (always app-level) are kept.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_JAVA_FATAL = re.compile(r"\bFATAL EXCEPTION\b")
_JAVA_PROCESS = re.compile(r"AndroidRuntime:\s*Process:\s*([^\s,]+)")
_ANR = re.compile(r"\bANR in ([^\s(]+)")
_NATIVE_SIGNAL = re.compile(r"\bFatal signal\s+\d+\s+\(SIG[A-Z]+\)")
_NATIVE_PROC = re.compile(r"\bpid\s+\d+\s*\(([^)]+)\)")
_PROCESS_DIED = re.compile(r"Process\s+([^\s]+)\s+\(pid[^)]*\)\s+has died")
_PKG_ANYWHERE = re.compile(r"([a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,})")

# Substrings marking infrastructure/system processes whose native aborts are
# emulator/system noise, not app-under-test defects.
_NOISE_MARKERS = (
    "screencap",
    "binder:",
    "android.hardware.",
    ".allocator",
    "audioserver",
    ".audio@",
    "hwservicemanager",
    "servicemanager",
    "logcat",
    "perfetto",
    "traced",
    "crash_dump",
    "tombstoned",
    "dumpsys",
    "vendor.qti",
    "cameraserver",
)


def _is_noise(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _NOISE_MARKERS)


@dataclass
class CrashEvent:
    kind: str  # java_crash | anr | native_crash | process_death
    package: str
    summary: str
    line: int  # 1-indexed line in the scanned text


def _package_near(lines: list[str], index: int, window: int = 6) -> str:
    for offset in range(0, window + 1):
        for probe in (index + offset, index - offset):
            if 0 <= probe < len(lines):
                match = _PKG_ANYWHERE.search(lines[probe])
                if match:
                    return match.group(1)
    return ""


def scan_crashes(logcat_text: str, package: str | None = None) -> list[CrashEvent]:
    """Return fatal events (crashes / ANRs) found in logcat text.

    Java crashes and ANRs are always app-level and always reported. Native
    crashes and process deaths from known infrastructure/system processes are
    filtered as noise. When ``package`` is given, only events attributable to it
    are returned. Duplicate events on the same line/kind are collapsed.
    """
    lines = logcat_text.splitlines()
    events: list[CrashEvent] = []

    def add(kind: str, index: int, pkg: str, summary: str) -> None:
        events.append(
            CrashEvent(kind=kind, package=pkg, summary=summary.strip()[:300], line=index + 1)
        )

    for index, line in enumerate(lines):
        if _JAVA_FATAL.search(line):
            pkg = ""
            for probe in range(index, min(index + 4, len(lines))):
                proc = _JAVA_PROCESS.search(lines[probe])
                if proc:
                    pkg = proc.group(1)
                    break
            pkg = pkg or _package_near(lines, index)
            detail = next(
                (
                    lines[probe]
                    for probe in range(index, min(index + 6, len(lines)))
                    if "Exception" in lines[probe] or "Error" in lines[probe]
                ),
                line,
            )
            add("java_crash", index, pkg, detail)
        elif _ANR.search(line):
            add("anr", index, _ANR.search(line).group(1), line)
        elif _NATIVE_SIGNAL.search(line):
            proc = _NATIVE_PROC.search(line)
            name = proc.group(1).strip() if proc else _package_near(lines, index)
            if _is_noise(name):
                continue  # screencap / binder / HAL abort — infrastructure noise
            add("native_crash", index, name, line)
        elif _PROCESS_DIED.search(line):
            name = _PROCESS_DIED.search(line).group(1)
            if _is_noise(name):
                continue
            # Process death is routine churn unless it's the app under test, so it
            # is only surfaced when a target package filter matches (below).
            if package:
                add("process_death", index, name, line)

    if package:
        pkg = package.strip()
        kept: list[CrashEvent] = []
        for event in events:
            if event.package == pkg or pkg in event.summary or pkg in lines[event.line - 1]:
                kept.append(event)
        events = kept

    seen: set[tuple[str, int]] = set()
    unique: list[CrashEvent] = []
    for event in events:
        key = (event.kind, event.line)
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return unique


def crash_summary(events: list[CrashEvent]) -> list[dict]:
    return [asdict(event) for event in events]
