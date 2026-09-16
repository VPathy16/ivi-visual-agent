# Sample IVI test app

`ivi.html` is a single self-contained "custom IVI" you can drive on an Android
emulator to test the agent **with the `benz` manual profile** end-to-end, without
a real head unit. Its screens and icons match
[`examples/benz-mbux-manual/manual.json`](../manual.json):

- **Home** → Climate / Seat Comfort / Vehicle tiles
- **Climate** → tap-to-set temperature slider (16–28 °C)
- **Seat massage** → program cards (Wave/Lumbar/Shoulder), intensity slider,
  Start→Stop with a visible running-state line
- **Vehicle settings** → list, incl. Seat comfort

Loaded fullscreen in the emulator browser it renders as one WebView (no per-widget
accessibility tree), so it exercises the **visual-grounding + manual** path — the
same situation as a custom OEM IVI.

## 1. Build the benz profile (once)

```bash
pip install -e '.[docs]'
brew install poppler
ivi-agent manual build --source examples/benz-mbux-manual --output output/pdf/benz.pdf
ivi-agent knowledge index output/pdf/benz.pdf --profile benz
```

## 2. Load the UI on the emulator

```bash
adb -s emulator-5554 push examples/benz-mbux-manual/testapp/ivi.html /sdcard/Download/ivi.html
adb -s emulator-5554 shell am start -a android.intent.action.VIEW \
    -d "file:///sdcard/Download/ivi.html" -t "text/html"
```

Chrome's address bar is harmless (the agent ignores it). For a cleaner, truly
fullscreen surface, use Chrome menu → **Add to Home screen** and launch it, or
enable immersive mode.

## 3. Run the agent against it (with the manual)

```bash
ivi-agent --config config.json run --serial emulator-5554 \
    --knowledge-profile benz --goal "Open the climate screen"

ivi-agent --config config.json run --serial emulator-5554 \
    --knowledge-profile benz --goal "Start the seat massage"
```

Compare **with vs without** `--knowledge-profile benz` to see the manual's effect
on recognizing the custom tiles/icons. For stronger recall, enable semantic
retrieval (`use_embeddings` / `icon_matching`) in `config.json` — see the repo
README's Configuration section.

> The rendered screens are captured in this session's chat for reference; they are
> the exact output of `ivi.html`.
