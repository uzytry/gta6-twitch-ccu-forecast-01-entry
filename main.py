# GTA VI launch-day Twitch viewers forecaster (gta6-twitch-ccu-forecast-01).
#
# Idea: a launch day's hourly curve is mostly decided by how big the first
# hour already is. A staggered rolling-midnight unlock leaves hour 1 small
# (only the earliest time zones are in) and the curve then explodes as the
# rest of the world wakes up and unlocks; a simultaneous global unlock fills
# hour 1 at once and the curve stays flat. Across past launches the log of
# hour 1's print explains the whole day's shape (correlation about -0.94
# with the peak log-ratio), so this program fits, for each hour, a local
# linear regression of log(viewers_h / viewers_1) on log(viewers_1) over the
# case's own comparables, weighted by a Gaussian kernel in log(viewers_1).
# The spread for the five quantiles is the regression's residual spread.
# Comparables whose release was contaminated by early access (their peak came
# days before release day, so their release-day curve is flattened) and weak
# analogues are left out of the fit; rolling-midnight launches count double,
# since the target launch unlocks that way. Everything is deterministic:
# same case in, same bytes out.
#
# Standard library only, no network, no model calls.

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Kernel bandwidth in log(viewers_1) units for the local fit.
BANDWIDTH = 1.5
# Extra weight for rolling-midnight launches (the target unlocks that way).
ROLLING_BOOST = 1.5
# The residual spread is shrunk a little: the kernel-weighted spread still
# carries some of the trend across launch sizes.
SPREAD_SHRINK = 0.85
# Smallest spread allowed, in log units, so no hour is forecast razor-thin.
SPREAD_FLOOR = 0.12
# Fallback spread for the persistence baseline (the starter's sigma).
BASELINE_SIGMA = 0.15
# Hours of the release day that are forecast.
HOURS = list(range(2, 25))


def round2(x):
    """Round to 2 decimals, half up, the way JavaScript rounds it.

    A whole number comes back as an int, so the JSON writes 1240 rather
    than 1240.0.
    """
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def release_day_hours(hourly, day_start):
    """Hourly viewers of a launch's release day, keyed by hour 1..24.

    A point at `from` covers the hour starting then; hour h of the day is
    the point whose hour is h - 1 in UTC. Points with no viewers are
    skipped.
    """
    day = day_start[:10]
    out = {}
    for point in hourly or []:
        stamp = point.get("from", "")
        viewers = point.get("viewers")
        if not stamp.startswith(day) or viewers is None:
            continue
        try:
            hour = int(stamp[11:13]) + 1
            value = float(viewers)
        except (ValueError, TypeError):
            continue
        if 1 <= hour <= 24 and value > 0 and hour not in out:
            out[hour] = value
    return out


def tier_of(unlock_text):
    """Which research tier a comparable's unlock note puts it in."""
    text = unlock_text or ""
    if "early-access-contaminated" in text:
        return "early_access"
    if "Tier: weak" in text:
        return "weak"
    if "target-history" in text:
        return "target_history"
    return "ok"


def is_rolling(unlock_text):
    """True when the note describes a rolling midnight-local unlock."""
    head = (unlock_text or "").lower().split("tier:")[0]
    if "simultaneous" in head and "rolling" not in head:
        return False
    return "rolling" in head or "midnight local" in head


def comparable_curves(case):
    """The fit's data: one (log hour-1, log-ratio curve) per usable launch.

    Only clean comparables are kept: early-access-contaminated and weak
    ones are dropped, as is the target game's own pre-release history.
    A curve needs hour 1 and at least 20 of the day's 24 hours.
    """
    curves = []
    for comp in case.get("comparables") or []:
        if tier_of(comp.get("unlock", "")) != "ok":
            continue
        hours = release_day_hours(comp.get("hourly"), comp.get("day_start", ""))
        first = hours.get(1)
        if first is None or len(hours) < 20:
            continue
        log_first = math.log(first)
        ratios = []
        for hour in HOURS:
            if hour in hours:
                ratios.append(math.log(hours[hour] / first))
            else:
                ratios.append(None)
        curves.append(
            {
                "log_first": log_first,
                "ratios": ratios,
                "rolling": is_rolling(comp.get("unlock", "")),
            }
        )
    return curves


def local_fit(curves, log_first, hour_index):
    """Local-linear fit of log-ratio at one hour, around log_first.

    Returns (median log-ratio, spread in log units). The kernel is
    Gaussian in log(hour-1) with BANDWIDTH; rolling launches weigh more.
    The effective log(hour-1) is clamped to the pool's range so a launch
    bigger than every comparable does not extrapolate the slope past it.
    """
    if not curves:
        return 0.0, BASELINE_SIGMA * math.sqrt(hour_index)
    lows = [c["log_first"] for c in curves]
    low, high = min(lows), max(lows)
    target = min(max(log_first, low), high)
    points = []
    for curve in curves:
        ratio = curve["ratios"][hour_index]
        if ratio is None:
            continue
        weight = math.exp(-0.5 * ((curve["log_first"] - target) / BANDWIDTH) ** 2)
        if curve["rolling"]:
            weight *= ROLLING_BOOST
        points.append((curve["log_first"], ratio, weight))
    if len(points) < 3:
        values = sorted(ratio for _, ratio, _ in points)
        if not values:
            return 0.0, BASELINE_SIGMA * math.sqrt(hour_index)
        return values[len(values) // 2], 0.4
    total_weight = sum(w for _, _, w in points)
    mean_x = sum(x * w for x, _, w in points) / total_weight
    mean_y = sum(y * w for _, y, w in points) / total_weight
    cov = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points)
    var = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    slope = cov / var if var > 0 else 0.0
    intercept = mean_y - slope * mean_x
    median = intercept + slope * target
    spread = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points)
        / total_weight
    )
    return median, max(spread, SPREAD_FLOOR) * SPREAD_SHRINK


def hour_one_print(case):
    """The case's hour-1 viewers, falling back to the last history point."""
    for known in case.get("known") or []:
        if known.get("hour") == 1 and known.get("viewers"):
            return max(1.0, float(known["viewers"]))
    history = case.get("history_hourly") or []
    for point in reversed(history):
        if point.get("viewers"):
            return max(1.0, float(point["viewers"]))
    return 1.0


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first_print = hour_one_print(case)
    log_first = math.log(first_print)
    curves = comparable_curves(case)
    rows = []
    for index, hour in enumerate(HOURS):
        median, spread = local_fit(curves, log_first, index)
        quantiles = []
        for z in Z:
            value = first_print * math.exp(median + z * spread)
            quantiles.append(round2(min(max(value, 1.0), 1e9)))
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
