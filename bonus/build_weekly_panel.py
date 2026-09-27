# -*- coding: utf-8 -*-
"""
Щотижнева панель "монета x тиждень" з 2021 року.

Місячна панель курсової має 31 checkpoint за 2024-2026 роки, тобто один
ринковий цикл і 19 тестових місяців. Тут checkpoint щонеділі з січня 2021
року: кілька повних циклів (бичачий 2021, ведмежий 2022, відновлення 2023,
2024-2026), приблизно в 12 разів більше точок і коротший горизонт
прогнозу (за замовчуванням 7 днів), тож тестові періоди не перекриваються
і результати можна чесно компаундувати.

Усі ознаки - каузальні ковзні вікна по денній історії монети: значення в
день t0 рахується лише з даних до t0 включно (close дня t0 відомий у
момент формування портфеля наприкінці цього дня). Ціль - дохідність від
close дня t0 до close дня t0 + H.

Запуск: python3 bonus/build_weekly_panel.py [H]
Результат: bonus/weekly_panel_h{H}.csv.gz (не комітиться, відтворюється
за хвилину-дві).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
spec = importlib.util.spec_from_file_location("build_panel", RAW / "build_panel.py")
bp = importlib.util.module_from_spec(spec)
sys.modules["build_panel"] = bp
spec.loader.exec_module(bp)

sys.path.insert(0, str(Path(__file__).parent))
from build_features_v2 import bollinger_pctb, macd_hist, rsi  # noqa: E402

MIN_AGE_DAYS = 98
FIRST_CHECKPOINT = "2021-01-03"


def daily_features(df, btc_ret, horizon):
    d = df.asfreq("D")
    close = d["close"].ffill()
    qv = d["quote_volume"]
    ret1 = close.pct_change()
    btc_ret = btc_ret.reindex(d.index)
    vq30 = qv.rolling(30, min_periods=20).mean()
    vq7 = qv.rolling(7, min_periods=5).mean()
    win = dict(window=90, min_periods=60)

    f = pd.DataFrame(index=d.index)
    f["close_t0"] = close
    f["trailing_return_7d"] = close / close.shift(7) - 1
    f["trailing_return_30d"] = close / close.shift(30) - 1
    f["trailing_return_90d"] = close / close.shift(90) - 1
    f["trailing_volatility_7d"] = ret1.rolling(7, min_periods=5).std()
    f["trailing_volatility_30d"] = ret1.rolling(30, min_periods=20).std()
    f["avg_quote_volume_30d"] = vq30
    f["trailing_volume_trend"] = vq30 / vq30.shift(30) - 1
    f["volume_7d_vs_30d"] = vq7 / vq30 - 1
    f["drawdown_from_high_180d"] = close / close.rolling(180, min_periods=1).max() - 1
    f["max_daily_return_30d"] = ret1.rolling(30, min_periods=20).max()
    f["rsi_14"] = rsi(close)
    f["macd_hist_12_26_9"] = macd_hist(close) / close
    f["bollinger_pctb_20"] = bollinger_pctb(close)
    f["btc_corr_90d"] = ret1.rolling(**win).corr(btc_ret)
    f["btc_beta_90d"] = ret1.rolling(**win).cov(btc_ret) / btc_ret.rolling(**win).var()
    f["coin_age_days"] = (d.index - d.index[0]).days

    fwd = close.shift(-horizon) / close - 1
    last_real = d["close"].last_valid_index()
    fwd[d.index + pd.Timedelta(days=horizon) > last_real] = np.nan
    f[f"forward_return_{horizon}d"] = fwd
    return f


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    manifest = bp.load_manifest()
    symbol_to_cgid = bp.build_symbol_to_cgid()
    cgid_to_sector = bp.build_cgid_to_sector()

    btc = bp.load_klines("BTCUSDT")
    btc_close = btc["close"].asfreq("D").ffill()
    btc_ret = btc_close.pct_change()
    btc_f = daily_features(btc, btc_ret, horizon)

    last_day = btc_close.index.max()
    checkpoints = pd.date_range(FIRST_CHECKPOINT, last_day - pd.Timedelta(days=horizon), freq="W-SUN")
    fwd_col = f"forward_return_{horizon}d"

    btc_at = btc_f.loc[checkpoints]
    btc_regime = np.where(btc_at["trailing_return_90d"] > 0.10, "bull",
                          np.where(btc_at["trailing_return_90d"] < -0.10, "bear", "sideways"))
    btc_info = pd.DataFrame({
        "checkpoint": checkpoints,
        "market_regime": btc_regime,
        "btc_trailing_return_30d": btc_at["trailing_return_30d"].values,
        "btc_trailing_return_90d": btc_at["trailing_return_90d"].values,
        "btc_rsi_14": btc_at["rsi_14"].values,
        "btc_forward_return": btc_at[fwd_col].values,
    })

    frames = []
    for m in manifest:
        symbol = m["symbol"]
        if m.get("error") or symbol == "BTCUSDT":
            continue
        cgid = symbol_to_cgid.get(symbol[:-4].lower())
        if not cgid:
            continue
        df = bp.load_klines(symbol)
        if df is None or len(df) < MIN_AGE_DAYS + horizon:
            continue
        f = daily_features(df, btc_ret, horizon)
        f = f.reindex(checkpoints)
        f = f[(f["coin_age_days"] >= MIN_AGE_DAYS) & f[fwd_col].notna()
              & f["trailing_return_90d"].notna()]
        if f.empty:
            continue
        f = f.rename_axis("checkpoint").reset_index()
        f["symbol"] = symbol
        f["coingecko_id"] = cgid
        f["sector"] = cgid_to_sector.get(cgid) or "Other"
        frames.append(f)

    panel = pd.concat(frames, ignore_index=True).merge(btc_info, on="checkpoint")
    panel = panel[panel["btc_forward_return"].notna()]
    panel["relative_strength_vs_btc_90d"] = panel["trailing_return_90d"] - panel["btc_trailing_return_90d"]
    panel["relative_strength_vs_btc_30d"] = panel["trailing_return_30d"] - panel["btc_trailing_return_30d"]
    panel["excess_return"] = panel[fwd_col] - panel["btc_forward_return"]
    panel["outperform_market"] = (panel["excess_return"] > 0).astype(int)
    by_cp = panel.groupby("checkpoint")["excess_return"]
    panel["beat_checkpoint_median"] = (panel["excess_return"] > by_cp.transform("median")).astype(int)
    panel["rank_target_pct"] = by_cp.rank(pct=True)
    panel = panel.rename(columns={fwd_col: "forward_return"})
    panel = panel.sort_values(["checkpoint", "symbol"]).reset_index(drop=True)

    n_per_cp = panel.groupby("checkpoint").size()
    print(f"Горизонт {horizon} днів. Рядків: {len(panel)}, монет: {panel['symbol'].nunique()}, "
          f"checkpoint: {panel['checkpoint'].nunique()} "
          f"({panel['checkpoint'].min().date()} - {panel['checkpoint'].max().date()})")
    print(f"Монет на checkpoint: від {n_per_cp.min()} до {n_per_cp.max()}")
    print("Режими ринку (частка checkpoint):")
    print(btc_info.set_index("checkpoint").loc[n_per_cp.index, "market_regime"]
          .value_counts(normalize=True).round(3).to_string())
    print("Пропуски в ознаках:")
    na = panel.isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "немає")

    out = Path(__file__).parent / f"weekly_panel_h{horizon}.csv.gz"
    panel.to_csv(out, index=False, compression="gzip")
    print("Збережено:", out)


if __name__ == "__main__":
    main()
