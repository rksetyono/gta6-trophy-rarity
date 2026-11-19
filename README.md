# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rksetyono/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rksetyono/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%) on the logit scale. An earn rate is a ratio, so the program models its two sides separately and divides at the target: the tracked-players count (one global log-log curve in time since release, fitted on the last 24 hours so launch-week deceleration is respected) and each trophy's earners (one log-log curve per trophy, shrunk toward its tier's median slope, fitted on the last 36 hours). Consecutive identical reads are collapsed to one state, a refresh jump in the tracked count cuts the fit window at the jump, and the forecast is anchored on the last observed counts. The spread combines a random-walk term (0.5 per sqrt-day), the fits' prediction error, and a term for Exophase's refresh lumpiness. On a frozen launch-night page it applies a launch-dilution prior instead of pretending the page stays frozen. Pure Python 3.13 standard library, deterministic, no network.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 /bot/main.py`; it reads every `*.in` case in the input directory and writes one `<id>.out` forecast per case to the output directory.

## Verification (local, practice cases)

The practice answers are not published, so the practice loss is measured against the answer the rules themselves imply: the house baseline's published practice loss of about 1.51 and typical miss of about 0.68 points imply the first refresh cut every rate roughly five-fold, so the practice answer was modelled as last-rate / k for k in 4–6 (the 100% trophy stays at 100%).

| case | this program | house flat baseline | house trend baseline |
|------|-------------|---------------------|----------------------|
| p00, k=4 | 0.164 | 1.279 | 1.279 |
| p00, k=5 | 0.161 | 1.498 | 1.498 |
| p00, k=6 | 0.197 | 1.678 | 1.678 |
| p01, k=5 | 0.161 | 1.498 | 1.498 |

On four synthetic final-like cases (horizon 6,510 minutes: smooth power-law growth, saturating player growth, a player-count jump after the checkpoint, and a 2x refresh step mid-window) the mean pinball loss is 0.269, against 0.310 for the flat baseline and 0.333 for the trend baseline; the trend baseline wins the smooth worlds by a hair and collapses on the step world (0.516). Two runs on the practice cases produce byte-identical output; runtime is about 0.08 s for both cases, far inside the 8 s cap.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
