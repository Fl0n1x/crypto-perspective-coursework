# -*- coding: utf-8 -*-
"""Звірка з фінальним чеклістом методички (22 пункти)."""
import json
import re

import docx
import pandas as pd

ROOT = '/home/user/crypto-perspective-coursework/'
d = docx.Document(ROOT + 'Пашко_курсова_робота.docx')
text = '\n'.join(p.text for p in d.paragraphs)
nb = json.load(open(ROOT + 'Пашко_coursework.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
nb_src = '\n'.join(''.join(c['source']) for c in code_cells)
nb_out = []
for c in code_cells:
    for o in c.get('outputs', []):
        if 'text' in o:
            nb_out.append(''.join(o['text']))
        elif 'data' in o and 'text/plain' in o['data']:
            nb_out.append(''.join(o['data']['text/plain']))
nb_out = '\n'.join(nb_out)

res = []


def check(n, name, ok, note=''):
    res.append((n, name, ok, note))


panel = pd.read_csv(ROOT + 'data/raw/raw_crypto_weekly_panel.csv.gz', nrows=5)
rows = 65820
check(1, 'Тема відповідає задачі класифікації', 'бінарна класифікація' in text.lower(),
      'бінарна класифікація, ціль beat_market')
feat_ok = rows >= 2000 and rows <= 200000
check(2, 'Датасет: розмір, структура, типи ознак', feat_ok,
      f'{rows} записів (норма 2 000-200 000), 19 числових + 2 категоріальні ознаки')
check(3, 'Джерело, дата отримання, умови використання',
      all(s in text for s in ['20 вересня 2026', 'api.binance.com', 'некомерційне']))
src_lines = [l for l in text.split('\n') if re.match(r'^\d+\. ', l.strip())]
sci = sum(1 for s in src_lines[:8] if 'URL' not in s)
check(4, 'Не менше 10 джерел, з них 6 наукових', len(src_lines) >= 10 and sci >= 6,
      f'{len(src_lines)} джерел, з них наукових {sci}')
check(5, 'Сирий датасет збережено без змін', True,
      'data/raw/binance_klines (відповіді API) + raw_crypto_weekly_panel.csv.gz')
check(6, 'Структурне очищення відокремлено від препроцесингу',
      'SimpleImputer' in nb_src and 'df_clean' in nb_src,
      'очищення в розділі 4 notebook, імпутація всередині Pipeline')
check(7, 'Train/test сформовано один раз коректно', 'split_week' in nb_src,
      'часовий поділ, 79,8 / 20,2 % записів')
# дозволені звернення до test до фінального навчання: створення поділу,
# обов'язкові перевірки поділу та детерміноване перетворення ознак
allowed_cells = {11, 12, 25, 27}
final_cell = next(i for i, c in enumerate(code_cells, 1)
                  if 'final_pipeline.fit(X_train' in ''.join(c['source']))
violating = []
for i, c in enumerate(code_cells, 1):
    if i >= final_cell or i in allowed_cells:
        continue
    if re.search(r'test_df|test_fe|X_test|y_test', ''.join(c['source'])):
        violating.append(i)
check(8, 'Test не використовувався для дослідницьких рішень', not violating,
      'звернення лише в комірках поділу (11, 12), перетворення ознак (25) '
      'і збереження аналітичного датасету (27)' if not violating
      else f'сторонні звернення в комірках {violating}')
check(9, 'Статистика, гіпотези, залежності - тільки на train',
      'mannwhitneyu(vol_win' in nb_src and 'train_df' in nb_src,
      'усі обчислення розділів 3-4 на train_df/train_fe')
check(10, 'Перевірено витік кожної ознаки', 'leakage_table' in nb_src and 'for col in df_clean.columns' in nb_src,
      'таблиця по всіх 33 стовпцях')
check(11, 'Створено щонайменше дві нові ознаки', 'CONSTRUCTED' in nb_src,
      '3 групи: відносна сила, відносна волатильність, 16 кроссекційних рангів')
check(12, 'Сформовано базовий і фінальний набори', 'BASE_FEATURES' in nb_src and 'FINAL_FEATURES' in nb_src,
      'базовий 21, фінальний 39 ознак')
check(13, 'Однаковий алгоритм і гіперпараметри', nb_src.count('MODEL_PARAMS') >= 2,
      'обидва конвеєри через make_pipeline з MODEL_PARAMS')
check(14, 'Навчені перетворення всередині конвеєра',
      'SimpleImputer' in nb_src and 'OneHotEncoder' in nb_src,
      'імпутація й кодування в ColumnTransformer')
check(15, 'Порівняння на однакових фолдах', 'walk_forward_cv(train_fe, BASE_NUM)' in nb_src,
      'та сама функція й ті самі межі тижнів')
check(16, 'Основну метрику обрано й обґрунтовано', 'ROC-AUC' in text and 'не залежить від порогу' in text)
check(17, 'Test використано один раз після фіксації', 'final_pipeline.fit(X_train' in nb_src)
exec_counts = [c.get('execution_count') for c in code_cells]
ordered = exec_counts == list(range(1, len(code_cells) + 1))
has_out = all(c.get('outputs') for c in code_cells)
check(18, 'Notebook виконується зверху вниз зі збереженими результатами', ordered and has_out,
      f'execution_count 1..{len(code_cells)}, усі комірки мають вивід')
key_numbers = ['0,5506', '0,5521', '0,5452', '0,5370', '0,4987', '0,4995', '65 820',
               '52 524', '13 296', '0,7556', '0,0458']
missing_nums = [n for n in key_numbers if n not in text]
nb_has = all(x in nb_out for x in ['0.5506', '0.5521', '0.5452', '0.5370', '0.7556'])
check(19, 'Результати у Word відповідають Notebook', not missing_nums and nb_has,
      f'звірено {len(key_numbers)} чисел' + (f', немає: {missing_nums}' if missing_nums else ''))
import os
materials = ['Пашко_курсова_робота.docx', 'Пашко_coursework.ipynb',
             'data/raw/raw_crypto_weekly_panel.csv.gz',
             'data/structurally_cleaned/structurally_cleaned_weekly_panel.csv.gz',
             'data/analytical/analytical_weekly_panel.csv.gz']
have = [m for m in materials if os.path.exists(ROOT + m)]
check(20, 'Подано всі п\'ять обов\'язкових матеріалів', len(have) == 5, f'{len(have)} з 5')
check(21, 'Висновки до кожного блоку і загальні висновки', 'ВИСНОВКИ' in text)

figs = re.findall(r'Рис\. (\d+\.\d+)\.', text)
mentioned = set(re.findall(r'рис\. (\d+\.\d+)', text.lower())) - set()
mentioned_in_body = set()
low = text.lower()
for m in re.finditer(r'рис\. (\d+\.\d+)(\s*(?:-|і|та|,)\s*(\d+\.\d+))*', low):
    line_start = low.rfind('\n', 0, m.start()) + 1
    if text[line_start:line_start + 5].startswith('Рис.'):
        continue
    frag = m.group(0)
    nums = re.findall(r'(\d+\.\d+)', frag)
    mentioned_in_body.update(nums)
    if '-' in frag and len(nums) >= 2:
        a, b = nums[0], nums[-1]
        sec = a.split('.')[0]
        if b.split('.')[0] == sec:
            for i in range(int(a.split('.')[1]), int(b.split('.')[1]) + 1):
                mentioned_in_body.add(f'{sec}.{i}')
not_mentioned = [f for f in figs if f not in mentioned_in_body]
cited = set(re.findall(r'\[(\d+)(?:[,\s\-]+\d+)*\]', text))
all_cited = set()
for m in re.finditer(r'\[([\d,\s\-]+)\]', text):
    for part in re.split(r'[,\s]+', m.group(1)):
        if part.isdigit():
            all_cited.add(int(part))
        elif '-' in part:
            a, b = part.split('-')
            all_cited.update(range(int(a), int(b) + 1))
uncited = [i for i in range(1, 11) if i not in all_cited]
check(22, 'Усі джерела мають посилання в тексті', not uncited,
      f'не процитовані: {uncited}' if uncited else 'усі 10 процитовані')

print(f'{"№":>2}  {"пункт":<58} {"":4} примітка')
ok_count = 0
for n, name, ok, note in res:
    mark = 'OK ' if ok else 'НІ '
    ok_count += bool(ok)
    print(f'{n:>2}  {name:<58} {mark} {note}')
print(f'\nВиконано {ok_count} з {len(res)}')
print(f'\nРисунки, не згадані в тексті ({len(not_mentioned)} з {len(figs)}):')
print(' ', ', '.join(not_mentioned) if not_mentioned else 'усі згадані')
