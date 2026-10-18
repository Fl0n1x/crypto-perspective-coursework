# -*- coding: utf-8 -*-
"""
Зводить три сирі джерела (Binance klines, CoinGecko markets, CoinGecko
categories) в одну панель "монета x місячний checkpoint" і зберігає
проміжні артефакти для аудиту join'у.

Логіка розв'язання колізій тікерів і виключення нон-крипто активів:
базовий актив Binance (наприклад "BTC" з "BTCUSDT") зіставляється з
монетою CoinGecko з тим самим символом, що має найбільшу капіталізацію
серед топ-2500. Якщо збігу немає взагалі - актив або відсутній у топ-2500
CoinGecko (замалий чи новий), або це не криптовалюта (токенізована акція).
Обидва випадки виключаються з панелі, кількість і приклади фіксуються.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent
KLINES_DIR = RAW / "binance_klines"
CATEGORIES_DIR = RAW / "coingecko_categories"
# місячна панель - перша версія роботи, збережена для історії в legacy/monthly;
# курсова використовує тижневу панель з build_weekly_panel.py
OUT_DIR = RAW.parent.parent / "legacy" / "monthly"

HORIZON_DAYS = 60
TRAIL_SHORT = 30
TRAIL_LONG = 90
TRAIL_HIGH_WINDOW = 180
MIN_HISTORY_BEFORE_CHECKPOINT = 98  # трохи більше за TRAIL_LONG про запас

CHECKPOINTS = pd.date_range("2024-01-31", "2026-08-31", freq="ME")

# пріоритет секторів для вибору ОДНІЄЇ первинної категорії, коли монета
# входить у кілька; від найспецифічнішої/змістовнішої до найзагальнішої
SECTOR_PRIORITY = [
    "meme-token", "dog-themed-coins", "gaming", "metaverse",
    "artificial-intelligence", "non-fungible-tokens-nft", "privacy-coins",
    "oracle", "decentralized-exchange", "centralized-exchange-token-cex",
    "storage", "real-world-assets-rwa", "yield-farming", "wrapped-tokens",
    "layer-2", "layer-1", "smart-contract-platform",
    "decentralized-finance-defi", "infrastructure", "stablecoins",
]
SECTOR_LABELS = {
    "meme-token": "Meme", "dog-themed-coins": "Meme",
    "gaming": "Gaming", "metaverse": "Metaverse/Gaming",
    "artificial-intelligence": "AI", "non-fungible-tokens-nft": "NFT",
    "privacy-coins": "Privacy", "oracle": "Oracle/Infra",
    "decentralized-exchange": "DEX", "centralized-exchange-token-cex": "Exchange Token",
    "storage": "Infrastructure", "real-world-assets-rwa": "RWA",
    "yield-farming": "DeFi", "wrapped-tokens": "Wrapped",
    "layer-2": "Layer-2", "layer-1": "Layer-1",
    "smart-contract-platform": "Layer-1",
    "decentralized-finance-defi": "DeFi", "infrastructure": "Infrastructure",
    "stablecoins": "Stablecoin",
}


def load_manifest():
    return json.load(open(KLINES_DIR / "_manifest.json", encoding="utf-8"))


def load_klines(symbol):
    path = KLINES_DIR / f"{symbol}.json"
    if not path.exists():
        return None
    rows = json.load(open(path, encoding="utf-8"))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    for col in ("close", "quote_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "close", "quote_volume"]].drop_duplicates("date").sort_values("date")
    df = df.set_index("date")
    return df


def build_symbol_to_cgid():
    markets = json.load(open(RAW / "coingecko_markets_by_mcap.json", encoding="utf-8"))
    by_symbol = {}
    for m in markets:
        sym = (m.get("symbol") or "").lower()
        mcap = m.get("market_cap") or 0
        if sym not in by_symbol or mcap > by_symbol[sym][1]:
            by_symbol[sym] = (m["id"], mcap)
    return {sym: cgid for sym, (cgid, _) in by_symbol.items()}


def build_cgid_to_sector():
    cgid_sectors = {}
    for sector in SECTOR_PRIORITY:
        path = CATEGORIES_DIR / f"{sector}.json"
        if not path.exists():
            continue
        for coin in json.load(open(path, encoding="utf-8")):
            cgid_sectors.setdefault(coin["id"], set()).add(sector)
    primary = {}
    for cgid, sectors in cgid_sectors.items():
        for pri in SECTOR_PRIORITY:
            if pri in sectors:
                primary[cgid] = SECTOR_LABELS[pri]
                break
    return primary


def main():
    manifest = load_manifest()
    symbol_to_cgid = build_symbol_to_cgid()
    cgid_to_sector = build_cgid_to_sector()

    print(f"Символів у маніфесті Binance: {len(manifest)}")
    print(f"Base asset -> CoinGecko id відомостей: {len(symbol_to_cgid)}")
    print(f"CoinGecko id з визначеним первинним сектором: {len(cgid_to_sector)}")

    btc = load_klines("BTCUSDT")
    if btc is None:
        raise RuntimeError("Немає історії BTCUSDT - неможливо порахувати бенчмарк")

    join_audit = []
    kline_cache = {}
    resolved, unresolved = 0, 0
    unresolved_examples = []

    eligible = [m for m in manifest if not m.get("error") and m.get("rows", 0) >= MIN_HISTORY_BEFORE_CHECKPOINT + HORIZON_DAYS]
    print(f"Символів з достатньою історією для панелі: {len(eligible)} з {len(manifest)}")

    for m in eligible:
        symbol = m["symbol"]
        base = symbol[:-4].lower()  # USDT завжди 4 символи
        cgid = symbol_to_cgid.get(base)
        sector = cgid_to_sector.get(cgid) if cgid else None
        status = "resolved" if cgid else "unresolved_no_cgid"
        if cgid and sector is None:
            status = "resolved_no_sector"
        join_audit.append({"symbol": symbol, "base_asset": base, "coingecko_id": cgid,
                            "sector": sector, "status": status})
        if cgid:
            resolved += 1
            df = load_klines(symbol)
            if df is not None:
                kline_cache[symbol] = (df, sector, cgid)
        else:
            unresolved += 1
            if len(unresolved_examples) < 15:
                unresolved_examples.append(symbol)

    print(f"Розв'язано (base asset -> CoinGecko id): {resolved}")
    print(f"Не розв'язано (виключено з панелі): {unresolved}")
    print("Приклади невирішених символів:", unresolved_examples)

    join_audit_df = pd.DataFrame(join_audit)
    join_audit_df.to_csv(OUT_DIR / "join_audit.csv", index=False, encoding="utf-8")

    def trailing_features(df, t0):
        window = df.loc[:t0]
        if len(window) < MIN_HISTORY_BEFORE_CHECKPOINT:
            return None
        close_t0 = window["close"].iloc[-1]
        idx = window.index

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

        high_window = window.loc[t0 - pd.Timedelta(days=TRAIL_HIGH_WINDOW):t0, "close"]
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

    def forward_return(df, t0):
        target = t0 + pd.Timedelta(days=HORIZON_DAYS)
        window = df.loc[t0:target]
        if len(window) < 2:
            return None
        last_available = window.index[-1]
        if (target - last_available).days > 5:
            return None  # монету, ймовірно, делістовано - forward невідомий
        return window["close"].iloc[-1] / window["close"].iloc[0] - 1

    rows = []
    btc_feature_cache = {}
    for t0 in CHECKPOINTS:
        btc_feat = trailing_features(btc, t0)
        btc_fwd = forward_return(btc, t0)
        if btc_feat is None or btc_fwd is None:
            continue
        btc_feature_cache[t0] = (btc_feat, btc_fwd)

    for symbol, (df, sector, cgid) in kline_cache.items():
        first_day = df.index.min()
        for t0 in CHECKPOINTS:
            if t0 not in btc_feature_cache:
                continue
            if (t0 - first_day).days < MIN_HISTORY_BEFORE_CHECKPOINT:
                continue
            feat = trailing_features(df, t0)
            if feat is None:
                continue
            fwd = forward_return(df, t0)
            if fwd is None:
                continue
            btc_feat, btc_fwd = btc_feature_cache[t0]
            btc_regime = ("bull" if btc_feat["trailing_return_90d"] > 0.10
                          else "bear" if btc_feat["trailing_return_90d"] < -0.10
                          else "sideways")
            row = {
                "symbol": symbol,
                "coingecko_id": cgid,
                "checkpoint": t0,
                "sector": sector if sector else np.nan,
                "market_regime": btc_regime,
                "coin_age_days": (t0 - first_day).days,
                **feat,
                "relative_strength_vs_btc_90d": feat["trailing_return_90d"] - btc_feat["trailing_return_90d"],
                "forward_return_60d": fwd,
                "btc_forward_return_60d": btc_fwd,
                "excess_return_60d": fwd - btc_fwd,
            }
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel["outperform_market"] = (panel["excess_return_60d"] > 0).astype(int)

    print()
    print("Розмір сирої панелі:", panel.shape)
    print("Унікальних монет у панелі:", panel["symbol"].nunique())
    print("Унікальних checkpoint:", panel["checkpoint"].nunique())
    print("Період checkpoint:", panel["checkpoint"].min(), "-", panel["checkpoint"].max())
    print("Баланс цільової змінної outperform_market:")
    print(panel["outperform_market"].value_counts(normalize=True).round(3).to_string())
    print("Пропуски по стовпцях:")
    print(panel.isna().sum()[panel.isna().sum() > 0].to_string())

    panel_out = panel.sort_values(["symbol", "checkpoint"]).reset_index(drop=True)
    panel_out.to_csv(OUT_DIR / "raw_crypto_perspective_panel.csv", index=False, encoding="utf-8")
    print()
    print("Збережено:", OUT_DIR / "raw_crypto_perspective_panel.csv")


if __name__ == "__main__":
    main()
