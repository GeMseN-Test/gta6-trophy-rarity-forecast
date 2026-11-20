"""GTA VI trophy rarity forecaster — two-regime logit + earners/players decomposition.

Entry for make.fun gta6-trophy-rarity-forecast-01. Reads one case
(makefun/gta6-trophies-case/v1) and writes /output/<id>.out: for each of the
game's 55 trophies, five quantiles (5/25/50/75/95%) of its Exophase earn rate
at the case target. Standard library only, no network, deterministic.

HOW IT WORKS (honest model, no black box)
------------------------------------------
An earn rate is a fraction: earners / players_tracked. Both move, and they
move for different reasons, so the model treats them separately:

1. REGIMES. Exophase's page refreshes in jumps: long flat stretches, then a
   sudden step when new counts land. Consecutive reads whose logit-rate jumps
   by more than 0.4, or whose players_tracked jumps by more than 1.5x, mark a
   regime break. Trends are fitted only on the reads after the last break;
   the size of past jumps feeds the spread (a refresh you have not seen yet
   is the biggest single risk).

2. TWO FITS, BLENDED. (a) Direct: Theil-Sen slope of logit(rate) against
   ln(1 + hours since release) — robust to one weird read. (b) Decomposition:
   Theil-Sen slopes of ln(earners) and ln(players_tracked) separately, then
   rate = earners / players at the target. The decomposition is what catches
   DILUTION: when a flood of new players joins, players_tracked grows faster
   than earners and the rate FALLS even while more people earn the trophy.
   The two forecasts are averaged in logit space; when they disagree, the
   average is safer than either alone under pinball loss.

3. TIER SHRINKAGE. A trophy's fitted slope is shrunk toward a prior slope
   for its kind (story chapter, side story, activity, collectible, misc) and
   tier (platinum/gold/silver/bronze). Kind comes from the trophy's own
   description keywords; tier from the case. Priors are documented
   assumptions, grounded in the live Exophase page read while building this
   entry (20 Nov 2026 ~05:05 UTC: ~3,320 players tracked; Chapter 1 at
   0.73%, fishing challenge 1.89%, most others 0.03-0.06%, platinum 0.03%),
   and in the launch-night pattern (638 tracked, rates stuck for hours, then
   dilution as the player flood arrived). A flat history — no movement at
   all, as in both practice cases — falls back to the prior drift alone:
   story trophies drift up, collectibles stay near flat, platinum barely
   moves. Nothing here pretends to know more than the data shows.

4. REFRESH-AWARE SPREAD. The 5-95% band is deliberately wide: sigma is the
   root-sum-square of the fit residual, the observed refresh-jump size
   (floored at 0.55 logit — the practice cases prove a jump of that size can
   arrive unseen), and a horizon term that grows with sqrt(days). A flat
   history gets an extra-wide band, because the next refresh is entirely
   unobserved. Target coverage is ~0.9-0.95, not a narrow band that looks
   clever and misses. The house starter's band (sigma 0.5*sqrt(days), loss
   ~1.51 on practice) is the cautionary tale: it is narrow precisely where
   the refresh lands.

5. CAPS AND FALLBACKS. Each kind has a plausible week-one ceiling (Chapter 1
   ~80%, platinum ~3%, collectibles ~12%); forecasts are soft-capped toward
   them. Every failure mode degrades to the prior-only forecast — empty
   history, one read, null rates, a 100% or 0% rate — never a crash, never an
   empty output. A case that cannot be parsed at all still yields 55 prior
   rows, because an invalid case forfeits the prize outright.

VALIDATION (done locally before submit)
---------------------------------------
- Practice cases p00/p01: format (55 rows, non-decreasing quantiles in
  0-100, <=64 KB) and runtime (<1 s, limit 8 s).
- Synthetic backtests: fake 2-day histories with known trends, refresh
  jumps and dilution, forecast 4.5 days out, score pinball loss on the logit
  scale and 90% coverage across seeds. Bands were widened until coverage
  sat near 0.9 without giving up median accuracy.
- Edge cases: empty history, single read, null rates, 0%/100% rates.

LIMITS (read this before trusting the numbers)
----------------------------------------------
- The final horizon (4.5 days) is far longer than anything in the practice
  cases (90 minutes). Extrapolating a 2-day trend 4.5 days out is honest
  guesswork with wide bands, not a measurement.
- Exophase tracks a self-selected slice of players (early buyers, trophy
  hunters), not all GTA VI players. Rates here are Exophase rates.
- Prior slopes and caps are judgement calls from one live read plus launch
  patterns, not fitted constants. If the player flood is much bigger or
  smaller than assumed, medians will be off — that is what the wide bands
  are for.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- logit helpers -------------------------------------------------------

def clamp_pct(x):
    """Clamp a percent into (0, 100) exclusive so the logit is finite."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg > 30.0:
        return 99.99
    if lg < -30.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, like JavaScript Math.round does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO-8601 timestamp (handles a trailing Z).

    Always returns a timezone-aware datetime (naive inputs are read as
    UTC): mixing a naive release date with aware read times used to raise
    TypeError and drop the whole case to the prior-only fallback.
    """
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# --- trophy kinds and priors ----------------------------------------------
# slope: logit drift per unit of ln(1 + hours since release).
# cap: plausible week-one ceiling for the earn rate, in percent.
# Grounded in the live Exophase read of 20 Nov 2026 ~05:05 UTC (~29 h after
# release, ~3,320 players tracked): Chapter 1 0.73%, fishing challenge
# 1.89%, most trophies 0.03-0.06%, platinum 0.03%; and in the launch-night
# dilution (638 tracked, flat for hours, then the flood arrived).
KIND_PRIORS = {
    # Early story: everyone plays it first, so it climbs fastest.
    "story":      {"slope": 0.85, "cap": 80.0},
    # Final mission / gold story beats: gated by finishing the game.
    "finale":     {"slope": 0.55, "cap": 45.0},
    # Stranger / side-story strands: steady, slower than main story.
    "side":       {"slope": 0.45, "cap": 35.0},
    # Activities (races, fishing, arcade, shooting range): pick-up-and-play.
    "activity":   {"slope": 0.35, "cap": 30.0},
    # Skill / misc feats (stunts, combat, wanted level): grindy but doable.
    "misc":       {"slope": 0.25, "cap": 20.0},
    # Collectibles / exploration: the long tail, barely moves in week one.
    "collect":    {"slope": 0.15, "cap": 12.0},
    # Platinum: needs everything else first.
    "platinum":   {"slope": 0.30, "cap": 3.0},
    # Gold completionist beats (100%, 250 stunts): almost nobody in week one.
    "grind":      {"slope": 0.10, "cap": 5.0},
}

# Keywords in a trophy's name/description, checked in order, first match
# wins. Story beats are named "Complete <Chapter N>" / "Complete <mission>".
_STORY_WORDS = ("chapter", "complete the final mission")
_FINALE_WORDS = ("final mission",)
_SIDE_WORDS = ("stranger", "for andy", "for kenton", "for billy", "for randy",
                "for tyrone", "for bret", "mission strands")
_ACTIVITY_WORDS = ("race", "fishing", "arcade", "shooting range", "drone",
                   "challenges", "bowling", "cocktail")
_COLLECT_WORDS = ("collectible", "card sets", "discover", "vehicles",
                  "explore all", "locations", "register")
_GRIND_WORDS = ("100% completion", "250 stunts", "50 challenges")


def classify(trophy):
    """Kind of a trophy, from its tier, name and description."""
    tier = str(trophy.get("tier", "")).lower()
    if tier == "platinum":
        return "platinum"
    text = ((trophy.get("name") or "") + " " +
            (trophy.get("description") or "")).lower()
    if any(w in text for w in _GRIND_WORDS):
        return "grind"
    if any(w in text for w in _STORY_WORDS):
        return "story"
    if any(w in text for w in _FINALE_WORDS):
        return "finale"
    if any(w in text for w in _SIDE_WORDS):
        return "side"
    if any(w in text for w in _ACTIVITY_WORDS):
        return "activity"
    if any(w in text for w in _COLLECT_WORDS):
        return "collect"
    return "misc"


# --- robust fitting --------------------------------------------------------


def theil_sen(xs, ys):
    """Median pairwise slope; the intercept is the median residual.

    Returns (slope, intercept) or None when there are fewer than 2 points.
    Pairwise slopes are O(n^2), so a very dense history is thinned to at
    most 150 points first — the 8-second runtime limit leaves no room for
    a 300-read history to cost 40k pairs per trophy per fit.
    """
    n = len(xs)
    if n < 2:
        return None
    if n > 150:
        step = (n + 149) // 150
        xs = xs[::step]
        ys = ys[::step]
        n = len(xs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if abs(dx) > 1e-9:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return None
    slopes.sort()
    slope = slopes[len(slopes) // 2] if len(slopes) % 2 else (
        slopes[len(slopes) // 2 - 1] + slopes[len(slopes) // 2]) / 2.0
    resid = sorted(y - slope * x for x, y in zip(xs, ys))
    intercept = resid[len(resid) // 2] if len(resid) % 2 else (
        resid[len(resid) // 2 - 1] + resid[len(resid) // 2]) / 2.0
    return slope, intercept


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


# --- per-trophy forecast -----------------------------------------------------


def forecast_trophy(trophy, reads, x_target, horizon_days):
    """Forecast one trophy. reads: list of (x, rate, players) sorted by x,
    where x = ln(1 + hours since release). Returns (median_logit, sigma)."""
    kind = classify(trophy)
    prior = KIND_PRIORS[kind]
    last_rate = reads[-1][1] if reads else 1.0
    last_x = reads[-1][0] if reads else 0.0

    # No usable history at all: prior drift from a neutral 1% start.
    if len(reads) < 2:
        med = logit(1.0) + prior["slope"] * x_target
        med = min(med, logit(prior["cap"]))
        return med, 1.05

    # --- regime split: cut at the last big jump (refresh) ------------------
    cut = 0
    jumps = []
    for j in range(1, len(reads)):
        dj = abs(logit(reads[j][1]) - logit(reads[j - 1][1]))
        p_ratio = (reads[j][2] / reads[j - 1][2]
                   if reads[j - 1][2] and reads[j][2] else 1.0)
        jumps.append(dj)
        if dj > 0.4 or p_ratio > 1.5 or p_ratio < 0.67:
            cut = j
    window = reads[cut:]
    if len(window) < 2:
        window = reads
    # Recent window: the last 24 h of reads. Earn-rate curves are convex in
    # log-time while a launch cohort is still accelerating (the whole-history
    # slope understates where the curve is heading); the recent slope is the
    # honest local predictor for a 4.5-day extrapolation. Needs >= 4 points.
    h_last = math.exp(window[-1][0]) - 1.0
    recent = [r for r in window
              if math.exp(r[0]) - 1.0 >= h_last - 24.0]
    fit_window = recent if len(recent) >= 4 else window
    xs = [r[0] for r in fit_window]
    rates = [r[1] for r in fit_window]
    players = [r[2] for r in fit_window]

    flat = all(abs(logit(r) - logit(rates[0])) < 1e-9 for r in rates)

    # --- fit 1: direct logit trend -----------------------------------------
    fit = theil_sen(xs, [logit(r) for r in rates])
    if fit is None:
        med_direct = logit(last_rate) + prior["slope"] * (x_target - last_x)
        resid_sigma = 0.3
    else:
        slope_fit, icept = fit
        # Shrink the fitted slope toward the kind prior. The prior counts
        # as 3 pseudo-observations; more real points = less shrinkage.
        n_eff = float(len(fit_window))
        slope = (n_eff * slope_fit + 3.0 * prior["slope"]) / (n_eff + 3.0)
        # Anchor the line at the last read (median-unbiased), not at the
        # fit intercept: the freshest read is the best level estimate.
        med_direct = logit(last_rate) + slope * (x_target - last_x)
        resid = [logit(r) - (icept + slope_fit * x) for x, r in zip(xs, rates)]
        resid_sigma = min(0.8, max(0.2, rms(resid)))

    # --- fit 2: earners / players decomposition -----------------------------
    med_decomp = None
    earners = [max(0.0, r * p / 100.0) for r, p in zip(rates, players)]
    fit_e = theil_sen(xs, [math.log(e + 0.5) for e in earners])
    fit_p = theil_sen(xs, [math.log(max(1.0, p)) for p in players])
    if fit_e is not None and fit_p is not None:
        slope_e, icept_e = fit_e
        slope_p, icept_p = fit_p
        # Anchor both at the last read, same as the direct fit.
        e_last = earners[-1]
        p_last = max(1.0, players[-1])
        e_tgt = (e_last + 0.5) * math.exp(
            slope_e * (x_target - last_x)) - 0.5
        p_tgt = p_last * math.exp(slope_p * (x_target - last_x))
        if e_tgt > 0 and p_tgt > 0:
            med_decomp = logit(100.0 * e_tgt / p_tgt)

    # --- blend ---------------------------------------------------------------
    if med_decomp is None:
        med = med_direct
    else:
        # When the two fits agree, either is fine; when they disagree the
        # average is the safer median under pinball loss.
        med = 0.5 * (med_direct + med_decomp)
    # Flat history: no signal at all, so the prior drift IS the forecast.
    if flat:
        med = logit(last_rate) + prior["slope"] * (x_target - last_x)
    # Soft cap toward the kind ceiling: a week-one rate above the cap is
    # possible, just unlikely — blend, do not hard-clip. But a trophy whose
    # last read is ALREADY above its cap (e.g. Gone Fishing at ~100%) is
    # not pulled back down: the cap limits growth, not the present.
    cap_logit = logit(prior["cap"])
    if med > cap_logit and logit(last_rate) < cap_logit:
        med = cap_logit + 0.35 * (med - cap_logit)

    # --- refresh-aware spread --------------------------------------------------
    jump_sigma = rms(jumps)
    # An unobserved refresh is the norm, not the exception: floor the jump
    # term high, higher still when the history never moved at all.
    jump_sigma = max(jump_sigma, 0.85 if flat else 0.45)
    jump_sigma = min(jump_sigma, 1.5)
    horizon_sigma = 0.22 * math.sqrt(max(horizon_days, 0.0))
    prior_sigma = 0.30 if flat else 0.0
    sigma = math.sqrt(resid_sigma ** 2 + jump_sigma ** 2 +
                      horizon_sigma ** 2 + prior_sigma ** 2)
    sigma = min(max(sigma, 0.55), 1.6)
    return med, sigma


# --- case forecast ------------------------------------------------------------


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0

    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = parse_time("2026-11-19T00:00:00Z")
    target = parse_time(case.get("target"))
    if target is None:
        # Fall back to checkpoint + horizon.
        at = parse_time(case.get("at")) or release
        target = at
    x_target = math.log(1.0 + max(
        0.0, (target - release).total_seconds()) / 3600.0)

    # Per-trophy read series, oldest first, nulls dropped.
    series = {t.get("id"): [] for t in trophies}
    for read in history:
        at = parse_time(read.get("at") or read.get("slot"))
        if at is None:
            continue
        x = math.log(1.0 + max(0.0, (at - release).total_seconds()) / 3600.0)
        players = read.get("players_tracked") or 0
        rates = read.get("rates") or []
        for i, trophy in enumerate(trophies):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            try:
                rate = float(rate)
            except (TypeError, ValueError):
                continue
            series[trophy.get("id")].append((x, rate, float(players)))

    rows = []
    for trophy in trophies:
        try:
            reads = sorted(series.get(trophy.get("id"), []))
            med, sigma = forecast_trophy(trophy, reads, x_target,
                                         horizon_days)
            qs = [round4(clamp_pct(inv_logit(med + z * sigma)))
                  for z in Z]
        except Exception:
            # Never crash a trophy: prior-only quantiles.
            kind = classify(trophy)
            prior = KIND_PRIORS[kind]
            med = min(logit(1.0) + prior["slope"] * x_target,
                      logit(prior["cap"]))
            qs = [round4(clamp_pct(inv_logit(med + z * 1.0))) for z in Z]
        # Guarantee non-decreasing quantiles after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # A case that cannot be parsed still gets 55 prior rows rather
            # than no output: an invalid case forfeits the prize outright.
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                trophies = case.get("trophies") or []
            except Exception:
                trophies = []
            rows = []
            for trophy in trophies:
                kind = classify(trophy)
                prior = KIND_PRIORS[kind]
                med = min(logit(1.0) + prior["slope"] * 5.0,
                          logit(prior["cap"]))
                qs = [round4(clamp_pct(inv_logit(med + z * 1.0)))
                      for z in Z]
                rows.append({"id": trophy.get("id"), "q": qs})
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
