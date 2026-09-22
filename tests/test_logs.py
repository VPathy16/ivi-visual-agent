import unittest

from ivi_agent.logs import scan_crashes

JAVA = """\
01-01 10:00:00.100 1000 1000 I ActivityManager: Displayed com.example.iviwv/.MainActivity
01-01 10:00:01.200 2000 2000 E AndroidRuntime: FATAL EXCEPTION: main
01-01 10:00:01.200 2000 2000 E AndroidRuntime: Process: com.example.iviwv, PID: 2000
01-01 10:00:01.200 2000 2000 E AndroidRuntime: java.lang.NullPointerException: boom
01-01 10:00:01.200 2000 2000 E AndroidRuntime:  at com.example.iviwv.MainActivity.onClick(MainActivity.java:42)
"""

ANR = """\
01-01 10:00:05.000 1000 1000 E ActivityManager: ANR in com.example.iviwv (com.example.iviwv/.MainActivity)
01-01 10:00:05.000 1000 1000 E ActivityManager: Reason: Input dispatching timed out
"""

NATIVE = """\
01-01 10:00:07.000 3000 3000 F libc    : Fatal signal 11 (SIGSEGV), code 1 in tid 3000 (example.iviwv)
01-01 10:00:07.100 3100 3100 F DEBUG   : *** *** *** *** *** *** *** *** *** *** ***
01-01 10:00:07.100 3100 3100 F DEBUG   : pid: 3000, name: com.example.iviwv
"""

CLEAN = """\
01-01 10:00:00.100 1000 1000 I ActivityManager: Displayed com.example.iviwv/.MainActivity
01-01 10:00:00.200 1000 1000 D SomeTag: everything is fine
01-01 10:00:00.300 1000 1000 W SomeTag: a warning, not fatal
"""

# Real emulator noise from a passing run: screencap/binder HAL SIGABRTs.
SCREENCAP_NOISE = """\
09-17 18:35:22.710 F/libc    ( 8204): Fatal signal 6 (SIGABRT), code -1 (SI_QUEUE) in tid 8205 (binder:8204_1), pid 8204 (screencap)
09-17 18:35:26.380 F/libc    ( 8252): Fatal signal 6 (SIGABRT), code -1 (SI_QUEUE) in tid 8253 (binder:8252_1), pid 8252 (screencap)
09-17 18:35:30.365 F/libc    ( 8308): Fatal signal 6 (SIGABRT), code -1 (SI_QUEUE) in tid 8309 (binder:8308_1), pid 8308 (screencap)
"""


class ScanCrashesTests(unittest.TestCase):
    def test_java_crash(self) -> None:
        events = scan_crashes(JAVA)
        self.assertTrue(any(e.kind == "java_crash" for e in events))
        crash = next(e for e in events if e.kind == "java_crash")
        self.assertEqual(crash.package, "com.example.iviwv")
        self.assertIn("NullPointerException", crash.summary)

    def test_anr(self) -> None:
        events = scan_crashes(ANR)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "anr")
        self.assertEqual(events[0].package, "com.example.iviwv")

    def test_native_crash(self) -> None:
        events = scan_crashes(NATIVE)
        self.assertTrue(any(e.kind == "native_crash" for e in events))

    def test_clean_log_has_no_events(self) -> None:
        self.assertEqual(scan_crashes(CLEAN), [])

    def test_screencap_binder_hal_noise_is_ignored(self) -> None:
        # These are emulator infrastructure aborts, not app defects.
        self.assertEqual(scan_crashes(SCREENCAP_NOISE), [])
        self.assertEqual(scan_crashes(SCREENCAP_NOISE, package="com.example.iviwv"), [])

    def test_app_native_crash_still_detected(self) -> None:
        # A native crash whose process is the app under test is a real defect.
        text = (
            "09-17 18:35:22.710 F/libc ( 900): Fatal signal 11 (SIGSEGV), "
            "code 1 in tid 900 (com.example.iviwv), pid 900 (com.example.iviwv)\n"
        )
        events = scan_crashes(text)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "native_crash")
        self.assertEqual(events[0].package, "com.example.iviwv")

    def test_package_filter(self) -> None:
        events = scan_crashes(JAVA, package="com.other.app")
        self.assertEqual(events, [])
        events = scan_crashes(JAVA, package="com.example.iviwv")
        self.assertTrue(events)

    def test_line_numbers_are_1_indexed(self) -> None:
        events = scan_crashes(JAVA)
        self.assertGreaterEqual(events[0].line, 1)


if __name__ == "__main__":
    unittest.main()
