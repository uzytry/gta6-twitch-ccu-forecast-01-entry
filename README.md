# GTA VI Twitch Viewers Forecaster — make.fun gta6-twitch-ccu-forecast-01 entry

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/uzytry/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/uzytry/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people will be watching Grand Theft Auto VI live on
Twitch in each hour of its release day (19 November 2026, UTC), as five
quantiles an hour, from the first hour's viewer count. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## How it works

`main.py` (Python 3.13, standard library only) reads one case from
`/input/<id>.in` and writes `/output/<id>.out`.

The key observation: a launch day's hourly curve is mostly decided by how
big the first hour already is. A staggered rolling-midnight unlock leaves
hour 1 small (only the earliest time zones are in) and the curve then
explodes as the rest of the world wakes up; a simultaneous global unlock
fills hour 1 at once and the curve stays flat. Across past launches, the
log of hour 1's print explains the day's shape (correlation about -0.94
with the peak log-ratio).

For each hour 2–24 the program fits a local linear regression of
log(viewers_h / viewers_1) on log(viewers_1) over the case's own
comparables (past big launches), weighted by a Gaussian kernel in
log(viewers_1). The five quantiles come from the fit's median plus the
regression's residual spread. Comparables whose release was contaminated
by early access (their peak came days before release day) and weak
analogues are left out; rolling-midnight launches count 1.5×, since GTA VI
unlocks that way.

## Practice score

Local scoring on the kit's four practice cases (mean pinball loss on the
log scale, the contest's own metric): **0.155** against the house
persistence baseline's **0.387**.

- p00 (Marvel's Wolverine): 0.171
- p01 (Red Dead Redemption 2): 0.142
- p02 (God of War Ragnarök): 0.116
- p03 (Marvel's Spider-Man 2): 0.190

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: the same case always produces the same bytes. No network,
no model calls, runs in well under the 8 s per-case cap.
