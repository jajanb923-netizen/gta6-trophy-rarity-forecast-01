# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jajanb923-netizen/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jajanb923-netizen/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate one week after launch (26 November 2026), from the house's recorded reads of the trophy page. An Exophase rate is a ratio — earners over players tracked — so the program models both sides: the tracked-player count (three growth forms, picked by backtest) and each trophy's earners (a cohort model: players join over time, each cohort earns a trophy with a per-trophy time constant and delay, fitted by model-averaging). Quantile spreads are asymmetric in logit space and calibrated by an out-of-sample backtest on the case's own history, so the 5–95% band stays honest over a 4.5-day horizon. A degenerate history (every read identical, as on launch night before Exophase's first refresh) takes a refresh path instead: the next refresh multiplies the tracked count and new players earn each trophy at a fraction of the old players' rate.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=vttfhgk4&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
