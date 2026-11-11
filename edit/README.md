# Blade / 80% zone reel: edit pipeline

Source: `VID_20261111_094301_018_bsl.mp4` (720x1280, 95.5 s, already cut out onto black).
Output: 1080x1920, 30 fps, H.264 + original audio.

## Steps

1. **Transcribe** for word timings (faster-whisper `small.en`):
   `python transcribe.py audio16k.wav transcript.json`
2. **Matte**: cut Joey out so photos sit behind him (Robust Video Matting, ONNX):
   `python matte.py raw.mp4 alpha_full.mkv "" "" rvm_mobilenetv3_fp32.onnx`
3. **Render**: set `SRC`, `ALPHA`, `ASSETS`, then
   `python render.py stills OUTDIR 5.9 17.6 ...` to check frames, or
   `python render.py video out.mp4` for the full edit.

`ASSETS/user/u1..u6.png` are the Snipes photos Joey supplied; `ASSETS/commons/full/*.jpg` are
the stock photos below. They are not in this repo.

## Timeline (synced to the recorded VO, not the written script)

| Time | VO | Visual |
|---|---|---|
| 0:00 | "physique like this" | Topless Blade 1998 (card) |
| 0:04 | "Wesley Snipes in Blade... live like him" | Sword poster + LIVE LIKE HIM |
| 0:06 | "60% zone" | Bars: Snipes 90% vs most people 60% (flickers) |
| 0:09 | "3, maybe 4 days a week... not even every week" | Week grid: Mon-Thu ticks, Fri-Sun crosses, 2nd worse week |
| 0:15 | "at 61, got called to play Blade again" | Recent Snipes photo + AGE 61 |
| 0:18 | "not Blade ready" | Blade face-down + highlighted quote |
| 0:20 | "one month, not six, not twelve" | Snarling Blade + 1 MONTH / NOT 6 / NOT 12 |
| 0:25 | "because of his baseline" | 1998 vs 2026 card + HIS BASELINE |
| 0:27 | "trains every day: martial arts..." | Fighting stance + chips (martial arts, strength, yoga, mobility, 10k steps) |
| 0:32 | "same meals on repeat... breakfast" | Oats + BREAKFAST |
| 0:36 | lunch | Chicken, rice, broccoli + LUNCH |
| 0:37 | dinner / omega-3 | Salmon + DINNER, then HIGH OMEGA-3 |
| 0:41 | "refined carbohydrates" | Bread + red X |
| 0:44 | "anything fried" | Fried chicken + red X |
| 0:46 | "let's be real... it's his job" | Joey full screen, no background |
| 0:52 | "don't need 90-95%... 80% sweet spot" | Bars: 90-95 grey / 80 green / 60 red |
| 0:57 | "3-2-1 method" | Deadlift photo + 3 / 2 / 1 rows |
| 1:09 | "1-3 hours a week" | Big type |
| 1:12 | "protein every meal / same meals on repeat" | Chicken, then meal-prep boxes |
| 1:17 | "go-to breakfasts, lunches, dinners" | Three meal strips |
| 1:22 | CTA | Joey full screen + red banner "DM me '321'" to the end |

## Caption credit

Photos: Blade (1998) and Blade II publicity stills; Lisa O'Connor/Getty Images;
Wikimedia Commons: Hogan.jac (CC BY-SA 4.0), HaJunkiyada (CC BY-SA 4.0), Daderot (CC0),
Mmangan333 (CC BY-SA 4.0), Brian Chan (CC0), U.S. Air Force/A1C Jovante Johnson (public domain),
Ella Olsson (CC BY 2.0).
