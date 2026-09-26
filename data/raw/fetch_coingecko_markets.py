# -*- coding: utf-8 -*-
"""
Крок 1 зіставлення символів. Тягне список монет CoinGecko, впорядкований за
капіталізацією (найбільша спочатку). Використовується для розв'язання
колізій тікерів: коли символ Binance збігається з кількома id в довіднику
CoinGecko, обирається монета з найбільшою капіталізацією - евристика,
задокументована в звіті як рішення проблеми структурного очищення.
"""
import json
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "coingecko_markets_by_mcap.json"


PAGES_DIR = Path(__file__).parent / "coingecko_pages"
PAGES_DIR.mkdir(exist_ok=True)


def http_get_json(url, retries=6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            wait = 20 * (attempt + 1)
            print(f"  {exc} -> чекаю {wait}с і повторюю")
            time.sleep(wait)
    raise RuntimeError(f"не вдалося отримати {url}")


def main():
    TARGET_PAGES = 10  # 10 * 250 = топ 2500 монет за капіталізацією
    for page in range(1, TARGET_PAGES + 1):
        page_file = PAGES_DIR / f"page_{page:02d}.json"
        if page_file.exists():
            print(f"сторінка {page}: вже завантажена, пропуск")
            continue
        url = (f"https://api.coingecko.com/api/v3/coins/markets"
               f"?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}")
        chunk = http_get_json(url)
        with open(page_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False)
        print(f"сторінка {page}: {len(chunk)} монет збережено")
        time.sleep(15)

    all_coins = []
    for page_file in sorted(PAGES_DIR.glob("page_*.json")):
        all_coins.extend(json.load(open(page_file, encoding="utf-8")))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_coins, f, ensure_ascii=False)
    print("Об'єднано:", OUT, "|", len(all_coins), "монет")


if __name__ == "__main__":
    main()
