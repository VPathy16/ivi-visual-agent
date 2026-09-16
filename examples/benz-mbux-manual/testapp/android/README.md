# Native WebView build of the sample IVI

The HTML mock in [`../ivi.html`](../ivi.html) is enough to open in a browser,
but a browser is the wrong host for automated grounding: Chrome silently
**drops synthetic `adb input tap` events** on web content, so the agent's taps
never land. Wrapping the same page in a tiny native `WebView` activity fixes
this — synthetic taps reach the view like any other Android app, and the app
runs fullscreen in landscape so the screenshot the agent grounds against has no
browser chrome.

This is what the end-to-end custom-IVI runs in the repo README were recorded
against (local `qwen3-vl:8b`, "Open the climate screen" and "Start the seat
massage" both pass).

## Contents

- `AndroidManifest.xml` — package `com.example.iviwv`, single fullscreen
  landscape `MainActivity`, launcher icon.
- `src/com/example/iviwv/MainActivity.java` — loads
  `file:///android_asset/ivi.html` into a `WebView` (JS + DOM storage on) and
  sets immersive fullscreen.
- `assets/ivi.html` — **not committed**; copy it from `../ivi.html` at build
  time (below) so the page has a single source of truth.

## Build without Gradle

Any Gradle/Android Studio project that points at these two sources and drops
`../ivi.html` into `assets/` will produce the same APK. If you don't want a
Gradle project, the command-line SDK tools build it directly:

```bash
# Prereqs: JDK 17, Android cmdline-tools (sdkmanager), and:
#   sdkmanager "platforms;android-34" "build-tools;34.0.0"
SDK=$HOME/android-sdk
BT=$SDK/build-tools/34.0.0
PLATFORM=$SDK/platforms/android-34/android.jar

# 0. Lay out the app and copy the shared HTML into assets/
mkdir -p build/gen build/obj build/apk assets
cp ../ivi.html assets/ivi.html

# 1. Compile resources (none beyond the manifest) and emit R.java
$BT/aapt2 compile --dir res -o build/res.zip 2>/dev/null || true   # no res/ dir needed
$BT/aapt2 link -o build/apk/base.apk -I "$PLATFORM" \
    --manifest AndroidManifest.xml -A assets --java build/gen \
    --auto-add-overlay

# 2. Compile Java -> classes -> dex
javac -source 17 -target 17 -bootclasspath "$PLATFORM" \
    -d build/obj $(find src build/gen -name '*.java')
$BT/d8 --output build/apk $(find build/obj -name '*.class')

# 3. Add classes.dex into the apk, align, sign with the debug key
(cd build/apk && zip -uj base.apk classes.dex)
$BT/zipalign -f 4 build/apk/base.apk build/ivi-sample-aligned.apk
$BT/apksigner sign --ks ~/.android/debug.keystore \
    --ks-pass pass:android --key-pass pass:android \
    --out ivi-sample.apk build/ivi-sample-aligned.apk
```

(A debug keystore is created automatically the first time you run any Gradle
build, or with:
`keytool -genkey -v -keystore ~/.android/debug.keystore -storepass android -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US"`.)

## Install and run

```bash
adb install -r ivi-sample.apk
adb shell am start -n com.example.iviwv/.MainActivity
```

The activity is landscape + fullscreen. Point the agent at the device as
described in the repo README's **Custom OEM IVI (Benz-style) bench bring-up**
section, using the `benz` knowledge profile built from
[`../../manual.json`](../../manual.json).

> Tip: on an AndroidWorld emulator image you may see a leftover
> `com.google.androidenv.accessibilityforwarder` crash dialog. Disable it with
> `adb shell pm disable-user --user 0 com.google.androidenv.accessibilityforwarder`.
