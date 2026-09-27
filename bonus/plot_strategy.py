# -*- coding: utf-8 -*-
"""
Графіки за out-of-fold прогнозами з strategy_backtest.py (читає
bonus/oof_predictions.csv, нічого не перенавчає).

1. quintile_validation_v2.png - середня реальна надлишкова дохідність
   за квінтилем прогнозу (квінтилі рахуються всередині кожного
   checkpoint, потім усереднюються), три моделі поруч.
2. long_short_by_month.png - результат портфеля "верхні 20% мінус нижні
   20%" для кожного місяця тестового періоду.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategy_backtest import add_scores, bottom_share, top_share

ROOT = Path(__file__).parent
FIG = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# ключ скору, підпис, колір (слоти 1-3 валідованої палітри, перевірено
# validate_palette.js --pairs all; бірюзовий нижче 3:1 до фону, тому
# на барах стоять підписи значень)
MODELS = [
    ("xs_median_blend", "Нова модель (ціль: краща за медіану місяця)", "#2a78d6"),
    ("ens_btc_target", "Попередній блендинг (ціль: краща за BTC)", "#eb6834"),
    ("baseline", "Базова модель курсової", "#1baf7a"),
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def quintile_table(oof):
    rows = []
    for cp, g in oof.groupby("checkpoint"):
        g = add_scores(g)
        for key, _, _ in MODELS:
            q = pd.qcut(g[f"s_{key}"].rank(method="first"), 5, labels=False) + 1
            means = g.groupby(q)["excess_return_60d"].mean()
            for bucket, value in means.items():
                rows.append({"checkpoint": cp, "model": key, "quintile": bucket, "excess": value})
    return (pd.DataFrame(rows).groupby(["model", "quintile"])["excess"].mean()
            .unstack("quintile"))


def long_short_table(oof):
    rows = []
    for cp, g in oof.groupby("checkpoint"):
        g = add_scores(g)
        row = {"checkpoint": cp, "regime": g["market_regime"].iloc[0]}
        for key, _, _ in MODELS:
            s = f"s_{key}"
            row[key] = top_share(g, s) - bottom_share(g, s)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_quintiles(qt):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(1, 6)
    width = 0.27
    for i, (key, label, color) in enumerate(MODELS):
        values = qt.loc[key].values * 100
        pos = x + (i - 1) * width
        ax.bar(pos, values, width, color=color, label=label, edgecolor=SURFACE, linewidth=1.5)
        for px, v in zip(pos, values):
            ax.text(px, v - 0.35 if v < 0 else v + 0.15, f"{v:.1f}", ha="center",
                    va="top" if v < 0 else "bottom", fontsize=7.5, color=INK_2)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["1\nнайнижчий", "2", "3", "4", "5\nнайвищий"], color=INK_2)
    ax.set_xlabel("Квінтиль прогнозу всередині місяця", color=INK_2, fontsize=10)
    ax.set_ylabel("Надлишкова дохідність проти BTC за 60 днів, %", color=INK_2, fontsize=10)
    ax.set_title("Що насправді принесла кожна група монет", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    plt.tight_layout()
    fig.savefig(FIG / "quintile_validation_v2.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_long_short(ls):
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(ls))
    for key, label, color in reversed(MODELS):
        values = ls[key].values * 100
        positive = int((ls[key] > 0).sum())
        ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"{label}: у плюсі {positive} з {len(ls)} місяців, "
                      f"в середньому {values.mean():+.1f}%")
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:%m.%y}\n{r}" for c, r in zip(ls["checkpoint"], ls["regime"].replace({"sideways": "flat"}))],
                       fontsize=7.5, color=INK_2)
    ax.set_ylabel("Верхні 20% мінус нижні 20%, % за 60 днів", color=INK_2, fontsize=10)
    ax.set_title("Long-short портфель, сформований у кожному місяці тестового періоду",
                 color=INK, fontsize=12, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0, -0.16), labelcolor=INK)
    fig.savefig(FIG / "long_short_by_month.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    oof = pd.read_csv(ROOT / "oof_predictions.csv", parse_dates=["checkpoint"])
    qt = quintile_table(oof)
    ls = long_short_table(oof)
    qt.round(4).to_csv(ROOT / "quintile_table.csv")
    ls.round(4).to_csv(ROOT / "long_short_by_month.csv", index=False)
    print("Середня надлишкова дохідність за квінтилем (квінтилі всередині місяця):")
    print((qt * 100).round(1).to_string())
    plot_quintiles(qt)
    plot_long_short(ls)
    print("Збережено:", FIG / "quintile_validation_v2.png", FIG / "long_short_by_month.png")


if __name__ == "__main__":
    main()
