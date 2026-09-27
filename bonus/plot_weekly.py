# -*- coding: utf-8 -*-
"""
Графік росту капіталу для щотижневих стратегій (weekly_portfolio.py):
скільки стало б із $1000 від 2 січня 2022 року, з комісіями, з
реінвестуванням. Вісь Y логарифмічна: однакова відстань означає однакове
відсоткове зростання.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).parent
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
START = 1000

SERIES = [
    ("long_short_liq", "Модель: long-short серед ліквідних монет", "#2a78d6"),
    ("btc_hold", "Просто тримати BTC", "#eb6834"),
    ("long_top", "Модель: лише купівля топ-20% (без шорту)", "#1baf7a"),
]


def main():
    r = pd.read_csv(ROOT / "weekly_strategy_returns_h7.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)

    dates = [r.index[0]] + list(r.index + pd.Timedelta(days=7))
    for key, label, color in SERIES:
        equity = [START] + list(START * (1 + r[key]).cumprod())
        ax.plot(dates, equity, color=color, linewidth=2, label=label)
        ax.annotate(f"${equity[-1]:,.0f}".replace(",", " "), xy=(dates[-1], equity[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=10, color=INK)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}".replace(",", " ")))
    ax.axhline(START, color=INK_2, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Капітал, $ (логарифмічна шкала)", color=INK_2, fontsize=10)
    ax.set_title("Що стало б із $1000: щотижневе ребалансування, комісія 0,1% від обороту, "
                 "січень 2022 - вересень 2026", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK)
    ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=90))
    out = ROOT / "figures" / "weekly_equity_curve.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("Збережено:", out)


if __name__ == "__main__":
    main()
