# -*- coding: utf-8 -*-
"""
Симуляція щотижневого портфеля на out-of-fold прогнозах weekly_model.py.

Кожної неділі портфель перебудовується за прогнозом і тримається рівно
H днів до наступного checkpoint, тож періоди не перекриваються і
результат можна компаундувати. Комісії рахуються від фактичного обороту:
частка портфеля, яку довелось купити або продати при ребалансуванні,
множиться на вартість угоди (0,1% на одиницю обороту: тейкер Binance
Futures 0,05% плюс прослизання 0,05%; сценарій 0,2% - песимістичний).
Фандинг ф'ючерсів не моделюється.

Стратегії:
  long_short     - купити верхні 20%, продати нижні 20%, рівні ваги,
                   нейтрально до ринку (однакові суми в обох ногах)
  long_short_liq - те саме лише серед ліквідної половини монет тижня
                   (обсяг вище медіани) - реалістичніший варіант, бо
                   дрібні монети часто не мають ф'ючерсу для шорту
  long_top       - лише купити верхні 20% (без шорту), результат в USDT
  btc_hold       - просто тримати BTC, для порівняння
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SHARE = 0.2


def leg_weights(g, score, share, side):
    k = max(1, int(round(len(g) * share)))
    picked = g.nlargest(k, score) if side == "long" else g.nsmallest(k, score)
    return pd.Series(1.0 / k, index=picked["symbol"].values)


def simulate(oof, score, universe="all", mode="long_short", share=SHARE, cost=0.001):
    rows = []
    prev = pd.Series(dtype=float)
    for cp, g in oof.groupby("checkpoint"):
        if universe == "liquid":
            g = g[g["avg_quote_volume_30d"] >= g["avg_quote_volume_30d"].median()]
        ret = g.set_index("symbol")["forward_return"]
        long_w = leg_weights(g, score, share, "long")
        if mode == "long_short":
            short_w = leg_weights(g, score, share, "short")
            w = long_w.sub(short_w, fill_value=0.0)
            gross = (long_w * ret.reindex(long_w.index)).sum() - (short_w * ret.reindex(short_w.index)).sum()
        else:
            w = long_w
            gross = (long_w * ret.reindex(long_w.index)).sum()
        turnover = w.sub(prev, fill_value=0.0).abs().sum()
        rows.append({"checkpoint": cp, "gross": gross, "turnover": turnover,
                     "net": gross - turnover * cost, "regime": g["market_regime"].iloc[0]})
        prev = w
    return pd.DataFrame(rows).set_index("checkpoint")


def btc_series(oof):
    s = oof.groupby("checkpoint")["btc_forward_return"].first()
    return pd.DataFrame({"gross": s, "net": s, "turnover": 0.0})


def stats(r, periods_per_year):
    net = r["net"]
    equity = (1 + net).cumprod()
    years = len(net) / periods_per_year
    cagr = equity.iloc[-1] ** (1 / years) - 1
    monthly = (1 + net).groupby(net.index.to_period("M")).prod() - 1
    drawdown = equity / equity.cummax() - 1
    yearly = (1 + net).groupby(net.index.year).prod() - 1
    best = monthly.max()
    return {
        "середній місяць": monthly.mean(),
        "медіанний місяць": monthly.median(),
        "місяців у плюсі": (monthly > 0).mean(),
        "найкращий місяць": best,
        "найгірший місяць": monthly.min(),
        "найкращий місяць у річних": (1 + best) ** 12 - 1,
        "CAGR (фактичний річний)": cagr,
        "разів за весь період": equity.iloc[-1],
        "Sharpe": net.mean() / net.std() * np.sqrt(periods_per_year),
        "макс. просадка": drawdown.min(),
        "оборот за тиждень": r["turnover"].mean(),
        **{f"рік {y}": v for y, v in yearly.items()},
    }


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    oof = pd.read_csv(ROOT / f"weekly_oof_h{horizon}.csv.gz", parse_dates=["checkpoint"])
    ppy = 365.25 / horizon

    runs = {
        "long_short": simulate(oof, "score"),
        "long_short_w": simulate(oof, "score_w"),
        "long_short_liq": simulate(oof, "score", universe="liquid"),
        "long_short_liq_w": simulate(oof, "score_w", universe="liquid"),
        "long_short_liq_cost0.2%": simulate(oof, "score", universe="liquid", cost=0.002),
        "long_top": simulate(oof, "score", mode="long_only"),
        "btc_hold": btc_series(oof),
    }
    table = pd.DataFrame({name: stats(r, ppy) for name, r in runs.items()})
    pd.set_option("display.width", 250)
    print(f"Тестовий період: {oof['checkpoint'].min().date()} - {oof['checkpoint'].max().date()}, "
          f"{oof['checkpoint'].nunique()} тижнів")
    print(table.round(3).to_string())

    table.to_csv(ROOT / f"weekly_strategy_stats_h{horizon}.csv")
    weekly = pd.DataFrame({name: r["net"] for name, r in runs.items()})
    weekly.to_csv(ROOT / f"weekly_strategy_returns_h{horizon}.csv")
    print("Збережено:", ROOT / f"weekly_strategy_stats_h{horizon}.csv")


if __name__ == "__main__":
    main()
