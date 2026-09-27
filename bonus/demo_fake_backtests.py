# -*- coding: utf-8 -*-
"""
Демонстрація: звідки в бектестах беруться "сотні відсотків на місяць".

Та сама щотижнева модель і та сама симуляція портфеля, що й у
weekly_model.py / weekly_portfolio.py, але з двома найпоширенішими
помилками бектесту. Результати цього скрипта НЕ є оцінкою моделі - це
показ того, наскільки легко отримати фантастичні цифри випадково.

1. In-sample: модель вчиться на всіх тижнях 2021-2026 і "торгує" на тих
   самих тижнях. Бустинг запам'ятовує, які монети виросли, і бектест
   показує не прогноз, а пам'ять.
2. Look-ahead на 1 день: до ознак додано дохідність монети за перший
   день після checkpoint. Типова причина в житті - зсув дат на день
   (свічка з open_time, прочитана як close того ж дня) або ознака,
   порахована з даних, які на момент рішення ще не були відомі.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from weekly_model import (CAT_FEATURES, EMBARGO_WEEKS, INITIAL_TRAIN_WEEKS, MARKET_FEATURES,  # noqa: E402
                          RAW_FEATURES, fit_predict, load)
from weekly_portfolio import simulate, stats  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
RAW = ROOT.parent / "data" / "raw"
spec = importlib.util.spec_from_file_location("build_panel", RAW / "build_panel.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

HORIZON = 7
LEAK_RETRAIN_EVERY = 13


def next_day_returns(df):
    out = []
    for symbol, g in df.groupby("symbol"):
        k = bp.load_klines(symbol)
        close = k["close"].asfreq("D").ffill()
        nxt = (close.shift(-1) / close - 1).reindex(g["checkpoint"]).values
        out.append(pd.Series(nxt, index=g.index))
    return pd.concat(out)


def main():
    df = load(HORIZON)
    weeks = sorted(df["checkpoint"].unique())
    test_weeks = weeks[INITIAL_TRAIN_WEEKS:]
    numeric = RAW_FEATURES + MARKET_FEATURES
    ppy = 365.25 / HORIZON
    results = {}

    print("1. In-sample: навчання й перевірка на тих самих тижнях...", flush=True)
    test = df[df["checkpoint"].isin(test_weeks)].copy()
    test["score"] = fit_predict(df, test, numeric)
    results["in_sample"] = stats(simulate(test, "score", universe="liquid"), ppy)

    print("2. Look-ahead на 1 день (walk-forward, перенавчання кожні 13 тижнів)...", flush=True)
    df["leak_next_day_return"] = next_day_returns(df)
    leak_numeric = numeric + ["leak_next_day_return"]
    chunks = []
    for start in range(INITIAL_TRAIN_WEEKS, len(weeks), LEAK_RETRAIN_EVERY):
        tw = weeks[start:start + LEAK_RETRAIN_EVERY]
        train = df[df["checkpoint"].isin(weeks[:start - EMBARGO_WEEKS])]
        t = df[df["checkpoint"].isin(tw)].copy()
        t["score"] = fit_predict(train, t, leak_numeric)
        chunks.append(t)
    leak = pd.concat(chunks)
    results["look_ahead_1d"] = stats(simulate(leak, "score", universe="liquid"), ppy)

    honest_path = ROOT / f"weekly_strategy_stats_h{HORIZON}.csv"
    table = pd.DataFrame(results)
    if honest_path.exists():
        honest = pd.read_csv(honest_path, index_col=0)["long_short_liq"]
        table.insert(0, "чесний walk-forward", honest)
    pd.set_option("display.width", 200)
    print()
    print("Long-short серед ліквідних монет, комісія 0,1% від обороту:")
    print(table.astype(float).round(3).to_string())
    table.to_csv(ROOT / "demo_fake_backtests.csv")
    print("Збережено:", ROOT / "demo_fake_backtests.csv")


if __name__ == "__main__":
    main()
