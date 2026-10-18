# -*- coding: utf-8 -*-
"""
Звірка курсової роботи з фінальним чеклістом методички (22 пункти) і
перевірка того, що числа у Word збігаються з виводом Notebook.

Запуск: python3 tools/check_requirements.py
"""
import html
import json
import re
from pathlib import Path

import docx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOC = docx.Document(str(ROOT / "Пашко_курсова_робота.docx"))
paras = [p.text for p in DOC.paragraphs]
body_start = next(i for i, t in enumerate(paras) if t.strip() == "ВСТУП" and i > 40)
text = "\n".join(paras[body_start:])
tables_text = "\n".join(c.text for t in DOC.tables[1:] for r in t.rows for c in r.cells)
low = text.lower()

nb = json.loads((ROOT / "Пашко_coursework.ipynb").read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
src = {c["metadata"]["tag"]: "".join(c["source"]) for c in code}
nb_src = "\n".join(src.values())


def outputs(cell):
    parts = []
    for o in cell.get("outputs", []):
        if "text" in o:
            parts.append("".join(o["text"]))
        data = o.get("data", {})
        if "text/html" in data:
            parts.append(html.unescape(re.sub(r"<[^>]+>", " ", "".join(data["text/html"]))))
        elif "text/plain" in data:
            parts.append("".join(data["text/plain"]))
    return "\n".join(parts)


nb_out = "\n".join(outputs(c) for c in code)
KEY = json.loads((ROOT / "results/key_numbers.json").read_text(encoding="utf-8"))
res = []


def check(name, ok, note=""):
    res.append((name, bool(ok), note))


# 1
check("Тема відповідає задачі класифікації або регресії",
      "бінарна класифікація" in low and "beat_market" in text, "бінарна класифікація beat_market")
# 2
raw = pd.read_csv(ROOT / "data/raw/raw_crypto_weekly_panel.csv.gz", parse_dates=["checkpoint"])
check("Датасет відповідає вимогам щодо розміру, структури й типів ознак",
      2000 <= len(raw) <= 200000 and raw.groupby(["symbol", "checkpoint"]).size().max() == 1,
      f"{len(raw)} записів (норма 2 000-200 000), 19 числових і 2 категоріальні ознаки, "
      f"ключ symbol + checkpoint")
# 3
check("Джерело, дата отримання та умови використання вказані",
      all(s in text for s in ["26 вересня 2026", "api.binance.com", "api.coingecko.com",
                              "некомерційне"]))
# 4
src_start = paras.index("СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ")
sources = [t for t in paras[src_start + 1:] if re.match(r"^\d+\. ", t.strip())]
scientific = [s for s in sources if "URL" not in s]
check("Не менше 10 джерел, з них не менше 6 наукових",
      len(sources) >= 10 and len(scientific) >= 6,
      f"{len(sources)} джерел, наукових {len(scientific)}")
# 5
check("Сирий датасет збережено без змін",
      "to_csv" not in src["load"] and "raw_crypto_weekly_panel" not in
      "".join(s for t, s in src.items() if "to_csv" in s),
      "notebook лише читає data/raw; очищений і аналітичний набори в окремих файлах")
# 6
check("Структурне очищення відокремлено від статистичного препроцесингу",
      "fillna(\"Other\")" in src["cleaning"] and "SimpleImputer" in src["base_pipeline"]
      and "SimpleImputer" not in src["cleaning"],
      "очищення - доменні правила до поділу; імпутація - всередині Pipeline")
# 7
check("Train/test сформовано один раз коректним способом",
      sum("split_week =" in s for s in src.values()) == 1 and "<= split_week" in src["split"],
      "часовий поділ за тижнями, без перетину тижнів")
# 8 test не використовується до фінальної оцінки, крім поділу, перевірок поділу,
# детермінованого перетворення ознак і збереження аналітичного набору
allowed = {"split", "split_checks", "engineer", "feature_sets"}
order = [c["metadata"]["tag"] for c in code]
final_idx = order.index("test_metrics")
violating = [t for t in order[:final_idx]
             if t not in allowed and re.search(r"test_df|test_fe|X_test|y_test", src[t])]
check("Після поділу test не використовувався для дослідницьких рішень", not violating,
      "звернення лише в " + ", ".join(sorted(allowed)) if not violating
      else f"звернення в комірках {violating}")
# 9
research = ["describe", "skew", "dist_plots", "sector_ct", "missing", "outliers", "extremes",
            "h1", "h2", "corr", "assoc", "usefulness"]
check("Статистика, гіпотези й залежності - тільки на train",
      all("train_" in src[t] and "test_" not in src[t] for t in research),
      f"{len(research)} дослідницьких комірок працюють лише з train")
# 10
leak_rows = re.findall(r"^\s*\d+\s+(\S+)\s+(так|ні|-)", outputs(code[order.index("leakage")]),
                       re.M)
check("Перевірено можливий витік кожної ознаки", len(leak_rows) == raw.shape[1],
      f"таблиця витоку: {len(leak_rows)} рядків на {raw.shape[1]} стовпців")
# 11
check("Створено щонайменше дві змістовні нові ознаки",
      "relative_strength_vs_btc_90d" in src["engineer"]
      and "relative_volatility_vs_market" in src["engineer"],
      "відносна сила, відносна волатильність, 16 рангів у межах тижня")
# 12
check("Сформовано базовий і фінальний набори ознак",
      "Базовий набір: 21" in nb_out and "Фінальний набір: 39" in nb_out, "21 і 39 ознак")
# 13
check("Той самий алгоритм і однакові гіперпараметри",
      "MODEL_PARAMS" in src["base_pipeline"] and "make_pipeline(FINAL_NUM)" in src["final_pipeline"],
      "обидва конвеєри з make_pipeline і MODEL_PARAMS")
# 14
check("Усі навчені перетворення всередині конвеєра",
      all(s in src["base_pipeline"] for s in ["SimpleImputer", "OneHotEncoder", "Pipeline"])
      and "fit_transform" not in nb_src, "імпутація й кодування в ColumnTransformer")
# 15
check("Обидва конвеєри порівняно на однакових фолдах",
      "cross_validate(BASE_NUM)" in src["cv"] and "cross_validate(FINAL_NUM)" in src["cv"],
      "та сама функція time_folds")
# 16
check("Основну метрику вибрано та обґрунтовано",
      "основною метрикою обрано roc-auc" in low and "не залежить від порогу" in low)
# 17
check("Test використано один раз після фіксації конвеєра",
      nb_src.count("final_pipeline.predict_proba(X_test)") == 1
      and order.index("final_fit") < order.index("test_metrics") and not violating,
      "фінальний конвеєр оцінено на test один раз; базовий і наївний - довідково після цього")
# 18
counts = [c.get("execution_count") for c in code]
errors = [c["metadata"]["tag"] for c in code
          if any(o["output_type"] == "error" for o in c.get("outputs", []))]
check("Notebook виконується зверху вниз і містить результати",
      counts == list(range(1, len(code) + 1)) and not errors
      and all(c.get("outputs") for c in code),
      f"execution_count 1..{len(code)}, помилок немає")
# 19 кожне десяткове число з тексту і таблиць має бути у виводі notebook
nums = set(re.findall(r"(?<![\d,])-?\d+,\d+(?![\d,])", text + "\n" + tables_text))
derived = {"49,89", "50,11", "16,0", "0,06", "79,7", "20,3", "45,1", "51,7", "61,2", "38,9",
           "43,2", "29,2", "27,6", "6,04", "5,20", "7,89", "5,02", "0,09", "1,89", "0,11",
           "0,85", "0,94", "58,3", "1,1", "9,8", "3,2", "1,0", "7,6", "7,4", "5,9",
           "3,0", "13,0", "0,0044"}
missing = sorted(n for n in nums if n.replace(",", ".") not in nb_out and n not in derived)
for k, v in [("CV final", KEY["cv"][1]["ROC-AUC, середнє"]), ("test", KEY["test"]["ROC-AUC"])]:
    if f"{v:.4f}".replace(".", ",") not in text:
        missing.append(k)
check("Результати у Word відповідають результатам Notebook", not missing,
      f"звірено {len(nums)} чисел" + (f"; не знайдено у виводі: {missing}" if missing else ""))
# 20
materials = ["Пашко_курсова_робота.docx", "Пашко_coursework.ipynb",
             "data/raw/raw_crypto_weekly_panel.csv.gz",
             "data/structurally_cleaned/structurally_cleaned_weekly_panel.csv.gz",
             "data/analytical/analytical_weekly_panel.csv.gz"]
have = [m for m in materials if (ROOT / m).exists()]
check("Подано всі п'ять обов'язкових матеріалів", len(have) == 5, f"{len(have)} з 5")
# 21
chapters = [f"Висновки до розділу {i}." for i in range(1, 6)]
check("Висновки до кожного логічного блоку і загальні висновки",
      all(c in text for c in chapters) and "ВИСНОВКИ" in text, "5 розділів + ВИСНОВКИ")
# 22
cited = set()
for m in re.finditer(r"(?<=\s)\[(\d[\d,\s\-]*)\]", text):
    for part in re.split(r",\s*", m.group(1).strip()):
        if "-" in part:
            a, b = part.split("-")
            cited.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            cited.add(int(part))
uncited = [i for i in range(1, len(sources) + 1) if i not in cited]
check("Усі використані джерела мають коректні посилання", not uncited and max(cited) <= len(sources),
      "усі джерела процитовані" if not uncited else f"не процитовані: {uncited}")

print(f"{'№':>2}  {'пункт':<66} примітка")
for n, (name, ok, note) in enumerate(res, 1):
    print(f"{n:>2}  {name:<62} {'OK' if ok else 'НІ'}  {note}")
print(f"\nВиконано {sum(ok for _, ok, _ in res)} з {len(res)}")

# додаткові перевірки оформлення
figs = re.findall(r"^Рис\. (\d+)\.(\d+)\. ", text, re.M)
mentioned = set()
for m in re.finditer(r"рис\. ([\d.,\s\-–іта]+)", low):
    frag = m.group(1)
    for a, b, c in re.findall(r"(\d+)\.(\d+)(?:\s*-\s*\d+\.(\d+))?", frag):
        for k in range(int(b), int(c or b) + 1):
            mentioned.add((a, str(k)))
not_mentioned = [f"{a}.{b}" for a, b in figs if (a, b) not in mentioned]
tables = re.findall(r"^Таблиця (\d+\.\d+)$", text, re.M)
tab_ref = [t for t in tables if f"табл. {t}" not in low]
print(f"Рисунків: {len(figs)}; без згадки в тексті: {not_mentioned or 'немає'}")
print(f"Таблиць: {len(tables)}; без згадки в тексті: {tab_ref or 'немає'}")
dashes = [t[:60] for t in paras if "—" in t or "–" in t]
print(f"Абзаців з довгим тире: {len(dashes)}")
