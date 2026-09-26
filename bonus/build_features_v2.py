# -*- coding: utf-8 -*-
"""
Бонусне розширення панелі (поза межами оцінювання курсової).

Бере ту саму логіку зіставлення symbol -> coingecko_id -> sector і той
самий набір checkpoint, що й `data/raw/build_panel.py`, але рахує ширший
набір трейлінгових ознак: до вже наявних (return 30/90, volatility,
volume trend, drawdown, age, relative strength) додає RSI-14, MACD
(гістограма 12/26/9) і позицію в смугах Боллінджера (%B, 20 днів) - усе
виключно з історії до t0, без витоку.

Результат зберігається окремо (`bonus/panel_v2.csv`), не чіпає артефакти
самої курсової (`data/raw`, `data/structurally_cleaned`, `data/analytical`).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
KLINES_DIR = RAW / "binance_klines"

spec = importlib.util.spec_from_file_location("build_panel", RAW / "build_panel.py")
bp = importlib.util.module_from_spec(spec)
sys.modules["build_panel"] = bp
spec.loader.exec_module(bp)

CHECKPOINTS = bp.CHECKPOINTS
HORIZON_DAYS = bp.HORIZON_DAYS
TRAIL_SHORT = bp.TRAIL_SHORT
TRAIL_LONG = bp.TRAIL_LONG
MIN_HISTORY_BEFORE_CHECKPOINT = bp.MIN_HISTORY_BEFORE_CHECKPOINT


def rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_hist(closes, fast=12, slow=26, signal=9):
    ema_fast = closes.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line - signal_line


def bollinger_pctb(closes, window=20, n_std=2):
    mean = closes.rolling(window).mean()
    std = closes.rolling(window).std()
    upper = mean + n_std * std
    lower = mean - n_std * std
    span = (upper - lower).replace(0, np.nan)
    return (closes - lower) / span


def trailing_features(df, t0):
    """Копія логіки з build_panel.py (там вона вкладена в main(), тому
    недоступна як атрибут модуля) - трейлінгові ознаки без витоку."""
    window = df.loc[:t0]
    if len(window) < MIN_HISTORY_BEFORE_CHECKPOINT:
        return None
    close_t0 = window["close"].iloc[-1]

    def close_days_before(n):
        target = t0 - pd.Timedelta(days=n)
        sub = window.loc[:target]
        return sub["close"].iloc[-1] if len(sub) else np.nan

    c30 = close_days_before(TRAIL_SHORT)
    c90 = close_days_before(TRAIL_LONG)
    ret30 = close_t0 / c30 - 1 if c30 and not np.isnan(c30) else np.nan
    ret90 = close_t0 / c90 - 1 if c90 and not np.isnan(c90) else np.nan

    last_30 = window.loc[t0 - pd.Timedelta(days=TRAIL_SHORT):t0]
    daily_ret = last_30["close"].pct_change().dropna()
    vol30 = daily_ret.std() if len(daily_ret) >= 5 else np.nan

    vol_recent = window.loc[t0 - pd.Timedelta(days=TRAIL_SHORT):t0, "quote_volume"].mean()
    vol_prior = window.loc[t0 - pd.Timedelta(days=2 * TRAIL_SHORT):t0 - pd.Timedelta(days=TRAIL_SHORT), "quote_volume"].mean()
    vol_trend = (vol_recent / vol_prior - 1) if vol_prior and vol_prior > 0 else np.nan

    high_window = window.loc[t0 - pd.Timedelta(days=bp.TRAIL_HIGH_WINDOW):t0, "close"]
    local_high = high_window.max() if len(high_window) else np.nan
    drawdown = close_t0 / local_high - 1 if local_high and local_high > 0 else np.nan

    return {
        "close_t0": close_t0,
        "trailing_return_30d": ret30,
        "trailing_return_90d": ret90,
        "trailing_volatility_30d": vol30,
        "trailing_volume_trend": vol_trend,
        "drawdown_from_high_180d": drawdown,
        "avg_quote_volume_30d": vol_recent,
    }


def trailing_features_v2(df, t0):
    base = trailing_features(df, t0)
    if base is None:
        return None
    window = df.loc[:t0]
    closes = window["close"]
    # daily indicators need daily-frequency data - forward-fill gaps in the
    # trailing window only (no look-ahead: ffill uses only past values)
    daily = closes.asfreq("D").ffill()
    base["rsi_14"] = rsi(daily).iloc[-1]
    base["macd_hist_12_26_9"] = macd_hist(daily).iloc[-1]
    base["bollinger_pctb_20"] = bollinger_pctb(daily).iloc[-1]
    # momentum acceleration: is the 30d trend stronger than the 90d trend
    r30, r90 = base["trailing_return_30d"], base["trailing_return_90d"]
    base["momentum_acceleration"] = (
        r30 - r90 / 3 if r30 is not None and r90 is not None and not np.isnan(r30) and not np.isnan(r90) else np.nan
    )
    return base


def trailing_btc_corr(df, btc, t0, window=90):
    start = t0 - pd.Timedelta(days=window)
    coin_ret = df.loc[start:t0, "close"].pct_change().dropna()
    btc_ret = btc.loc[start:t0, "close"].pct_change().dropna()
    joined = pd.concat([coin_ret, btc_ret], axis=1, join="inner")
    if len(joined) < 20:
        return np.nan
    return joined.iloc[:, 0].corr(joined.iloc[:, 1])


def forward_return(df, t0):
    target = t0 + pd.Timedelta(days=HORIZON_DAYS)
    window = df.loc[t0:target]
    if len(window) < 2:
        return None
    last_available = window.index[-1]
    if (target - last_available).days > 5:
        return None
    return window["close"].iloc[-1] / window["close"].iloc[0] - 1


def main():
    manifest = bp.load_manifest()
    symbol_to_cgid = bp.build_symbol_to_cgid()
    cgid_to_sector = bp.build_cgid_to_sector()

    btc = bp.load_klines("BTCUSDT")
    if btc is None:
        raise RuntimeError("Немає історії BTCUSDT")

    eligible = [
        m for m in manifest
        if not m.get("error") and m.get("rows", 0) >= MIN_HISTORY_BEFORE_CHECKPOINT + HORIZON_DAYS
    ]

    kline_cache = {}
    for m in eligible:
        symbol = m["symbol"]
        base = symbol[:-4].lower()
        cgid = symbol_to_cgid.get(base)
        if not cgid:
            continue
        sector = cgid_to_sector.get(cgid)
        df = bp.load_klines(symbol)
        if df is not None:
            kline_cache[symbol] = (df, sector, cgid)

    btc_feature_cache = {}
    for t0 in CHECKPOINTS:
        btc_feat = trailing_features_v2(btc, t0)
        btc_fwd = forward_return(btc, t0)
        if btc_feat is None or btc_fwd is None:
            continue
        btc_feature_cache[t0] = (btc_feat, btc_fwd)

    rows = []
    for symbol, (df, sector, cgid) in kline_cache.items():
        first_day = df.index.min()
        for t0 in CHECKPOINTS:
            if t0 not in btc_feature_cache:
                continue
            if (t0 - first_day).days < MIN_HISTORY_BEFORE_CHECKPOINT:
                continue
            feat = trailing_features_v2(df, t0)
            if feat is None:
                continue
            fwd = forward_return(df, t0)
            if fwd is None:
                continue
            btc_feat, btc_fwd = btc_feature_cache[t0]
            btc_regime = (
                "bull" if btc_feat["trailing_return_90d"] > 0.10
                else "bear" if btc_feat["trailing_return_90d"] < -0.10
                else "sideways"
            )
            row = {
                "symbol": symbol,
                "coingecko_id": cgid,
                "checkpoint": t0,
                "sector": sector if sector else np.nan,
                "market_regime": btc_regime,
                "coin_age_days": (t0 - first_day).days,
                **feat,
                "relative_strength_vs_btc_90d": feat["trailing_return_90d"] - btc_feat["trailing_return_90d"],
                "btc_rsi_14": btc_feat["rsi_14"],
                "btc_corr_90d": trailing_btc_corr(df, btc, t0),
                "forward_return_60d": fwd,
                "btc_forward_return_60d": btc_fwd,
                "excess_return_60d": fwd - btc_fwd,
            }
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel["outperform_market"] = (panel["excess_return_60d"] > 0).astype(int)

    print("Розмір розширеної панелі:", panel.shape)
    print("Нові стовпці порівняно з основною панеллю: rsi_14, macd_hist_12_26_9,",
          "bollinger_pctb_20, momentum_acceleration, btc_rsi_14, btc_corr_90d")
    print("Пропуски по нових ознаках:")
    new_cols = ["rsi_14", "macd_hist_12_26_9", "bollinger_pctb_20", "momentum_acceleration",
                "btc_rsi_14", "btc_corr_90d"]
    print(panel[new_cols].isna().sum().to_string())

    out_path = Path(__file__).parent / "panel_v2.csv"
    panel.sort_values(["symbol", "checkpoint"]).reset_index(drop=True).to_csv(
        out_path, index=False, encoding="utf-8")
    print("Збережено:", out_path)


if __name__ == "__main__":
    main()
