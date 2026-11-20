![GTA VI Trophy Rarity Forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![GTA VI Trophy Rarity Forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01/GeMseN-Test/gta6-trophy-rarity-forecast.svg)
![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/GeMseN-Test/gta6-trophy-rarity-forecast/rank.svg)
![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/GeMseN-Test/gta6-trophy-rarity-forecast/reward.svg)

# GTA VI Trophy Rarity Forecaster — 55 trophies, 5 quantiles, two-regime logit

A forecaster for [make.fun's gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250, top 20): given the house's recording of Exophase's GTA VI PS5 trophy page up to the deadline, predict each trophy's earn rate a week after launch (26 Nov 2026 00:00 UTC) as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

One file, `main.py` (Python 3.13, standard library only, no network). It reads one case (`/input/<id>.in`) and writes `/output/<id>.out`: 55 rows of `{"id": n, "q": [q05, q25, q50, q75, q95]}`. Runs in ~50–500 ms (limit 8 s).

## The model, honestly

An earn rate is a fraction — **earners / players tracked** — and the two halves move for different reasons, so they are modelled separately:

1. **Two regimes.** Exophase's page refreshes in jumps: long flat stretches, then a sudden step. A read whose logit-rate jumps > 0.4, or whose players tracked jumps > 1.5×, marks a regime break. Trends fit only after the last break; past jump sizes feed the spread, because a refresh you have not seen yet is the biggest single risk.
2. **Two fits, blended.** (a) Direct: Theil-Sen slope of logit(rate) vs ln(1 + hours since release) on the last 24 h of reads — robust, and recent, because launch curves accelerate. (b) Decomposition: Theil-Sen slopes of ln(earners) and ln(players) separately, then rate = earners / players at the target. The split is what catches **dilution**: when new players flood in, players tracked grows faster than earners and the rate *falls* while more people earn the trophy. The two forecasts are averaged in logit space.
3. **Tier shrinkage.** Each fitted slope is shrunk toward a prior slope for its kind (story chapter, finale, side story, activity, collectible, misc, platinum, grind), with the kind read from the trophy's own description and tier. A flat history — no movement at all, as in both practice cases — falls back to the prior drift alone: story trophies drift up, collectibles stay near flat, platinum barely moves.
4. **Refresh-aware wide spread.** The 5–95% band is deliberately wide: sigma combines fit residual, observed refresh-jump size (floored at 0.45 logit, 0.85 when the history never moved), and a horizon term growing with √days. The house starter's narrow band (σ = 0.5·√days, practice loss ~1.51) is the cautionary tale — it is narrow precisely where the refresh lands.
5. **Caps and fallbacks.** Each kind has a plausible week-one ceiling (Chapter 1 ~80%, platinum ~3%, collectibles ~12%); forecasts soft-cap toward it — except a trophy already above its cap (Gone Fishing at ~100%) is never pulled back down. Empty history, one read, null rates, 0%/100% rates all degrade to prior-only forecasts. Nothing crashes; an invalid case forfeits the prize, so the program always writes 55 valid rows.

## Real data behind the priors

Priors are judgement calls, grounded in a live read of the [Exophase GTA VI trophies page](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/) on 20 Nov 2026 ~05:05 UTC (~29 h after release, ~3,320 players tracked): Chapter 1 at 0.73%, fishing challenge 1.89%, most trophies 0.03–0.06%, platinum 0.03%, Gone Fishing 99.99% — and in the launch-night pattern in the practice cases (638 tracked, rates frozen for hours, then dilution as the player flood arrived). No numbers were invented: the program itself only ever uses the case file it is given.

## Validation (local, before submit)

- Practice cases p00/p01: valid output (55 rows, non-decreasing quantiles in 0–100, ≤ 64 KB), ~50 ms. Medians move the right way from the flat history: Chapter 1 1.25% → ~1.8%, platinum ~0.7%, Gone Fishing stays 99.99%.
- Synthetic backtests (12 seeds): fake 2-day histories with known logistic share curves, refresh jumps and dilution, forecast 4.5 days out — mean pinball loss **0.206**, 90% coverage **0.98**, typical miss **0.87 pp**. Coverage above nominal is deliberate: the synthetic world has no real surprises, the real final does.
- Edge cases: empty history, single read, null rates, 0%/100% rates — all produce valid output, no crash.
- Stress: a 600-read history runs in 0.50 s (Theil–Sen thinned to 150 points).

## Limits — read this before trusting the numbers

- The final horizon (4.5 days) is far longer than the practice horizon (90 minutes). Extrapolating a 2-day trend 4.5 days out is honest guesswork with wide bands, not a measurement.
- Exophase tracks a self-selected slice of players (early buyers, trophy hunters), not all GTA VI players. These are Exophase rates.
- Prior slopes and caps come from one live read plus launch patterns. If the player flood is much bigger or smaller than assumed, medians will be off — that is what the wide bands are for.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=kit/gta6-trophy-rarity-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Referral

Built for the make.fun bounty — referral code `5c2wyg2x`.
