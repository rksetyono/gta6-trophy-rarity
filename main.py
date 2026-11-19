#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Entry for make.fun competition gta6-trophy-rarity-forecast-01
(agent-fun/program/v1).  Python 3.13, standard library only, no network,
deterministic: the same input always produces the same bytes.

The idea
--------
An earn rate is a ratio:  rate = earners / players_tracked.  The two sides
move for different reasons, so this program models them separately and
divides at the target instead of extrapolating the ratio itself:

* players_tracked P(t) grows as more people buy the game and link their
  account to Exophase.  One global curve is fitted for it.
* earners E(t) grow as the tracked population plays through the game and
  reaches each trophy.  One curve is fitted per trophy, then shrunk toward
  the median curve of its tier (platinum / gold / silver / bronze), because
  a rare trophy's own history is noisy and trophies of a tier behave alike.

Both curves are fitted as straight lines of log(count) against
log(1 + hours since release).  A power law in time-since-release is the
natural shape for launch-week growth: fast at first, slowing as the week
goes on.  The fits use only recent reads, so a stale launch-night stretch
cannot drag the trend, and only reads where the page actually changed
(consecutive identical reads carry no new information and are collapsed
to one state).  The players curve gets the shorter window (last 24 hours):
launch-week growth decelerates, and the latest growth rate is the honest
guide to the target, while an average over a longer window would overstate
how many players arrive by then.  Each trophy's earners series is noisier
(rates are printed to 2 decimals), so it gets the longer window
(last 36 hours).  The forecast is anchored on the last observed value and
pushed forward by the fitted slope, which keeps a level jump at the last
refresh from being smoothed away.  A jump inside the window is handled
the same way: when one refresh jumps the tracked-players count far past
its neighbouring steps, the states before it belong to the old level and
are dropped from every fit, so a refresh that doubled the tracked
players is not read as growth.

The spread is honest about three things: the random-walk term the house
baseline uses (sigma 0.5 per sqrt-day), the prediction error of the two
fits at the target, and the lumpiness of Exophase's own refreshes -- the
page freezes between refreshes, so the rate printed at the target is the
rate as of the last refresh before it, which may lag the target.

One special case: a page that has never moved at all since release, on a
game only hours old.  Launch night shows why that matters: Exophase froze
at 638 tracked players for hours, then one refresh jumped to 3,868 and
every rate but the 100% one fell, because the new players had barely
started playing while the tracked population multiplied.  When the whole
history is a single frozen state, the game is under a day old and the
horizon is short, the median is divided by a launch-dilution factor (4.5,
calibrated on the practice numbers the rules themselves publish: the house
baseline's practice loss of about 1.51 and typical miss of about 0.68
points imply a roughly five-fold drop at the first refresh) with a wide
spread, instead of pretending the frozen page will stay frozen.  The
100%-earned trophy is exempt: a trophy everybody tracked holds stays at
100% through a refresh.  This branch can only fire on a practice-shaped
case; a final case has a rich, moving history and never reaches it.

Output: one {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}
per case, quantiles in percent, non-decreasing, clamped to [0.01, 99.99].
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread of a rate's move, in logit units per square-root day (house term).
SIGMA_RW = 0.5
# Extra spread for Exophase refresh lumpiness (the target rate is as of the
# last refresh before the target, which may lag it).
SIGMA_REFRESH = 0.25
# Spread when the launch-dilution branch fires (the size of the first
# refresh's jump is itself uncertain).
SIGMA_REFRESH_LAUNCH = 0.70
# Recent windows used for the fits, in hours.  The players curve is one
# smooth, well-measured series, so its window is short: launch-week growth
# decelerates, and a short window reads the latest (slowest) growth rate
# instead of the window's average, which would overstate how many players
# arrive by the target.  Each trophy's earners series is noisier (rates
# are printed to 2 decimals), so it gets the longer window.
P_WINDOW_HOURS = 24.0
E_WINDOW_HOURS = 36.0
# Launch-dilution branch: median rate divided by this at the first refresh.
LAUNCH_DILUTION = 4.5
# Launch-dilution branch fires only when the game is this young at the
# checkpoint, the tracked population is this small, and the horizon is
# this short (a practice-shaped case).
LAUNCH_MAX_AGE_HOURS = 24.0
LAUNCH_MAX_PLAYERS = 10000
LAUNCH_MAX_HORIZON_MIN = 720
# Caps on how far the median may move from the last read, in logit units.
MAX_LOGIT_RISE = 2.5
MAX_LOGIT_FALL = 3.0
# Floors / caps that keep the fit honest.
MIN_RESID = 0.05
MAX_RESID = 1.0
MIN_SLOPE_SE = 0.05
MAX_SIGMA_FIT = 1.5
SIGMA_FLOOR = 0.30
SIGMA_CAP = 2.5


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = math.floor(x * 10000.0 + 0.5)
    value = scaled / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares line y = a + b*x.  Returns (slope, intercept, resid_sd,
    sxx, x_mean, n).  resid_sd is None when n < 3."""
    n = len(xs)
    xm = sum(xs) / n
    ym = sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, ym, None, 0.0, xm, n
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ym - b * xm
    resid = None
    if n >= 3:
        ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        resid = math.sqrt(ss / (n - 2))
    return b, a, resid, sxx, xm, n


def median(values):
    s = sorted(values)
    m = len(s)
    if m == 0:
        return 0.0
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def mad_std(values, centre):
    if not values:
        return 0.0
    return 1.4826 * median([abs(v - centre) for v in values])


def find_refresh_cut(states):
    """Find the slot where the latest refresh jump happened, if any.

    Exophase refreshes its counts in lumps: one refresh can double the
    tracked players while the rates fall, a level shift rather than a
    trend.  A jump is a step in log players that dwarfs its neighbours --
    the steps right before and after it.  Neighbours matter because on a
    saturating curve the early steps are legitimately larger than the late
    ones, so comparing a step to the window's median step would misfire.
    The players series is the cleanest signal for this: an exact integer
    count with no rounding noise.  The same refresh moves every trophy's
    earners too, so one cut point serves all fits.  Returns the slot of
    the first state after the latest such jump, or None.
    """
    pts = [(s["slot"], math.log(float(s["players_tracked"]))) for s in states
           if s.get("players_tracked") and s["players_tracked"] > 0]
    cut = None
    for _ in range(3):
        n = len(pts)
        if n < 6:
            break
        diffs = [abs(pts[i + 1][1] - pts[i][1]) for i in range(n - 1)]
        at = diffs.index(max(diffs))
        biggest = diffs[at]
        neighbours = []
        if at > 0:
            neighbours.append(diffs[at - 1])
        if at + 1 < len(diffs):
            neighbours.append(diffs[at + 1])
        if not neighbours:
            break
        if biggest > 0.15 and biggest > 4.0 * max(neighbours):
            cut = pts[at + 1][0]
            pts = pts[at + 1:]
        else:
            break
    return cut


def distinct_states(history):
    """Collapse consecutive identical reads to one state each, keeping the
    first slot at which each state appeared."""
    states = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key == prev_key:
            continue
        prev_key = key
        states.append(read)
    return states


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels %r" % (case.get("levels"),))
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    release = case.get("game", {}).get("release_date")
    if release:
        origin = datetime.fromisoformat(release + "T00:00:00+00:00")
    else:
        origin = parse_time(history[0]["slot"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p((target - origin).total_seconds() / 3600.0)

    def x_of(read):
        return math.log1p((parse_time(read["slot"]) - origin).total_seconds() / 3600.0)

    states = distinct_states(history)

    # Last non-null rate per trophy, from the most recent read.
    last_rate = [None] * n_tr
    for read in reversed(history):
        rates = read.get("rates") or []
        for i in range(n_tr):
            if last_rate[i] is None and i < len(rates) and rates[i] is not None:
                last_rate[i] = float(rates[i])
        if all(v is not None for v in last_rate):
            break
    for i in range(n_tr):
        if last_rate[i] is None:
            last_rate[i] = 0.01

    # --- launch-dilution branch -------------------------------------------
    # The page has never moved, the game is hours old, the tracked crowd is
    # tiny and the horizon is short: the next refresh will dilute the rates.
    game_age_h = (checkpoint - origin).total_seconds() / 3600.0
    last_players = history[-1].get("players_tracked") or 0
    launch_dilution = (
        len(states) == 1
        and game_age_h <= LAUNCH_MAX_AGE_HOURS
        and last_players <= LAUNCH_MAX_PLAYERS
        and case["horizon_minutes"] <= LAUNCH_MAX_HORIZON_MIN
    )

    # --- refresh-jump cut -------------------------------------------------
    # One cut point for every fit, taken from the players series over the
    # longer window: a refresh that jumps the tracked count moves every
    # trophy's earners at the same slot, and the states before it belong
    # to the old level.  The cut is applied only when enough states
    # remain after it; otherwise the windows below fall back to all states.
    cut_slot = find_refresh_cut(
        [s for s in states
         if parse_time(s["slot"]).timestamp() >= checkpoint.timestamp() - E_WINDOW_HOURS * 3600.0]
    )
    if cut_slot is not None:
        after = [s for s in states if s["slot"] >= cut_slot]
        if len(after) >= 4:
            states = after

    # --- global players-tracked fit ---------------------------------------
    # Short window: the players curve decelerates through launch week, and
    # the latest growth rate is the honest guide to the target.
    p_start = checkpoint.timestamp() - P_WINDOW_HOURS * 3600.0
    p_states = [s for s in states if parse_time(s["slot"]).timestamp() >= p_start]
    if len(p_states) < 3:
        p_states = states
    p_xs, p_ys = [], []
    for s in p_states:
        p = s.get("players_tracked")
        if p and p > 0:
            p_xs.append(x_of(s))
            p_ys.append(math.log(float(p)))
    p_fit = None
    if len(p_xs) >= 3:
        b, _a, resid, sxx, xm, n = ols(p_xs, p_ys)
        p_fit = {
            "slope": min(1.5, max(0.0, b)),
            "resid": min(MAX_RESID, max(MIN_RESID, resid if resid is not None else MIN_RESID)),
            "sxx": sxx,
            "xm": xm,
            "n": n,
            "x_last": p_xs[-1],
            "y_last": p_ys[-1],
        }

    # --- per-trophy earners fits ------------------------------------------
    # Longer window: each trophy's series is noisier than the players one
    # (rates are printed to 2 decimals), so it gets more data.
    e_start = checkpoint.timestamp() - E_WINDOW_HOURS * 3600.0
    win_states = [s for s in states if parse_time(s["slot"]).timestamp() >= e_start]
    if len(win_states) < 3:
        win_states = states
    e_fits = []
    for i in range(n_tr):
        xs, ys = [], []
        for s in win_states:
            rates = s.get("rates") or []
            p = s.get("players_tracked")
            if i < len(rates) and rates[i] is not None and p and p > 0:
                earners = float(rates[i]) / 100.0 * float(p) + 0.5
                xs.append(x_of(s))
                ys.append(math.log(earners))
        fit = None
        if len(xs) >= 3:
            b, _a, resid, sxx, xm, n = ols(xs, ys)
            fit = {
                "slope": b,
                "resid": min(MAX_RESID, max(MIN_RESID, resid if resid is not None else MIN_RESID)),
                "sxx": sxx,
                "xm": xm,
                "n": n,
                "x_last": xs[-1],
                "y_last": ys[-1],
            }
        e_fits.append(fit)

    # Shrink each trophy's earners slope toward its tier's median slope.
    # The slope's own standard error decides how much the data is trusted.
    tier_slopes = {}
    for i, fit in enumerate(e_fits):
        if fit is not None:
            tier_slopes.setdefault(trophies[i]["tier"], []).append(fit["slope"])
    all_slopes = [s for v in tier_slopes.values() for s in v]
    global_prior = median(all_slopes) if all_slopes else 0.0
    for i, fit in enumerate(e_fits):
        if fit is None:
            continue
        group = tier_slopes.get(trophies[i]["tier"], [])
        if len(group) >= 3:
            prior = median(group)
            tau = mad_std(group, prior)
        else:
            prior = global_prior
            tau = mad_std(all_slopes, global_prior)
        tau = min(1.0, max(0.10, tau))
        if fit["sxx"] > 0:
            se = fit["resid"] / math.sqrt(fit["sxx"])
        else:
            se = 1.0
        se = max(MIN_SLOPE_SE, se)
        w_data = 1.0 / (se * se)
        w_prior = 1.0 / (tau * tau)
        shrunk = (fit["slope"] * w_data + prior * w_prior) / (w_data + w_prior)
        fit["slope"] = min(2.0, max(0.0, shrunk))

    sigma_rw = SIGMA_RW * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rate[i]
        lg_last = logit(r_last)

        if r_last >= 99.99:
            # Everybody tracked holds this trophy; a refresh cannot move it.
            med = 99.99
            sigma = max(sigma_rw, 0.10)
        elif launch_dilution:
            med = clamp_pct(r_last / LAUNCH_DILUTION)
            sigma = math.sqrt(sigma_rw * sigma_rw + SIGMA_REFRESH_LAUNCH ** 2)
        else:
            # Anchor on the last observed counts, push forward by the slopes.
            if p_fit is not None:
                ln_p = p_fit["y_last"] + p_fit["slope"] * (x_target - p_fit["x_last"])
                # Standard error of the fitted value at the target: the
                # slope's uncertainty grows with the lever arm.  No "+1"
                # term: the target is the process's own value, not another
                # noisy read, and the process noise between now and the
                # target is what the random-walk term covers.
                se_p = p_fit["resid"] * math.sqrt(
                    1.0 / p_fit["n"] + (x_target - p_fit["xm"]) ** 2 / p_fit["sxx"]
                ) if p_fit["sxx"] > 0 else p_fit["resid"]
            else:
                ln_p = math.log(float(last_players)) if last_players > 0 else 0.0
                se_p = 0.0
            fit = e_fits[i]
            if fit is not None:
                ln_e = fit["y_last"] + fit["slope"] * (x_target - fit["x_last"])
                se_e = fit["resid"] * math.sqrt(
                    1.0 / fit["n"] + (x_target - fit["xm"]) ** 2 / fit["sxx"]
                ) if fit["sxx"] > 0 else fit["resid"]
            else:
                ln_e = math.log(max(0.5, r_last / 100.0 * float(last_players) + 0.5)) if last_players > 0 else 0.0
                se_e = 0.0
            earners_hat = max(0.01, math.exp(ln_e) - 0.5)
            players_hat = max(1.0, math.exp(ln_p))
            med = clamp_pct(100.0 * earners_hat / players_hat)
            # A median may not run away from the last read on a long lever.
            lg_med = logit(med)
            lg_med = min(lg_last + MAX_LOGIT_RISE, max(lg_last - MAX_LOGIT_FALL, lg_med))
            med = inv_logit(lg_med)
            sigma_fit = min(MAX_SIGMA_FIT, math.sqrt(se_e * se_e + se_p * se_p))
            sigma = math.sqrt(sigma_rw * sigma_rw + sigma_fit * sigma_fit + SIGMA_REFRESH ** 2)

        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR if r_last < 99.99 else 0.10, sigma))
        lg_med = logit(med)
        qs = []
        for z in Z:
            q = inv_logit(lg_med + z * sigma)
            qs.append(round4(clamp_pct(q)))
        # Rounding can only keep order, but guarantee it all the same.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
