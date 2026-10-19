# -*- coding: utf-8 -*-
"""
Спільні функції збирання тексту курсової роботи.

Оформлення відповідає docs/Вимоги до оформлення курсової.docx і зразку
docs/Зразок оформлення курсової СА-32.docx: Times New Roman, текст 14 пт з
інтервалом 1,5 і відступом 1 см, код 12 пт з одинарним інтервалом без відступу,
таблиці 13 пт, рисунки й підписи по центру, інтервали до і після 0 пт, усі
літери чорні, номер сторінки праворуч у верхньому колонтитулі, крім титульної.
"""
import json
import re
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "Пашко_курсова_робота.docx"
NOTEBOOK = ROOT / "Пашко_coursework.ipynb"
FIG = ROOT / "figures"
FIRST_LINE = Cm(1)
BLACK = RGBColor(0, 0, 0)
TNR = "Times New Roman"

_SIZES = json.loads((FIG / "sizes.json").read_text(encoding="utf-8"))
_NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
_CODE = {c["metadata"]["tag"]: "".join(c["source"]) for c in _NB["cells"]
         if c["cell_type"] == "code"}
TOPIC_TAIL = "МАШИННОГО НАВЧАННЯ (НА ПРИКЛАДІ ПРОГНОЗУВАННЯ ВІДНОСНОЇ ТИЖНЕВОЇ ДИНАМІКИ ЦІН КРИПТОВАЛЮТ)"
PLAN_DATES = ["26.09.2026", "02.10.2026", "10.10.2026", "20.10.2026", "30.10.2026",
              "09.11.2026", "18.11.2026", "25.11.2026", "28.11.2026"]
FIG_COUNTER = {}
USED_TAGS = []


def _font(run, size, italic=False, bold=False):
    run.font.name = TNR
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), TNR)
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    run.italic = italic
    run.bold = bold


_POWER = re.compile(r"(?<=10)\^(-?\d+)")


def _add_text(p, text, size, italic=False, bold=False):
    """Додає текст; запис 10^-65 перетворюється на степінь верхнім індексом."""
    parts = _POWER.split(text)
    for i, part in enumerate(parts):
        if not part:
            continue
        run = p.add_run(part.replace("-", "\u2212") if i % 2 else part)
        _font(run, size, italic, bold)
        if i % 2 == 1:
            run.font.superscript = True


def _zero_spacing(pf, line=1.5):
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = line
    pf.left_indent = Cm(0)
    pf.right_indent = Cm(0)


def _fix_styles(doc):
    for name in ["Normal", "Heading 1", "Heading 2"]:
        st = doc.styles[name]
        _zero_spacing(st.paragraph_format)
        st.font.name = TNR
        st.font.color.rgb = BLACK
        st.font.size = Pt(14)
        if name != "Normal":
            st.font.bold = True
            st.paragraph_format.keep_with_next = True


def _page_numbers(doc):
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True
    for part in (sec.header, sec.first_page_header):
        for p in part.paragraphs[1:]:
            p._p.getparent().remove(p._p)
    sec.first_page_header.paragraphs[0].text = ""
    p = sec.header.paragraphs[0]
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _zero_spacing(p.paragraph_format, line=1.0)
    for kind, text in (("begin", None), (None, "PAGE \\* MERGEFORMAT"), ("separate", None),
                       (None, "2"), ("end", None)):
        run = p.add_run()
        _font(run, 14)
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
            run._r.append(el)
        elif text.startswith("PAGE"):
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
            run._r.append(el)
        else:
            run.text = text


def _update_fields_on_open(doc):
    # зміст заповнюється готовими рядками (tools/docx/toc.py), тож запит Word на
    # оновлення полів при відкритті не потрібен
    settings = doc.settings.element
    for el in settings.findall(qn("w:updateFields")):
        settings.remove(el)
    for zoom in settings.findall(qn("w:zoom")):
        if zoom.get(qn("w:percent")) is None:
            zoom.set(qn("w:percent"), "100")


def _fix_title_page(doc, body_start):
    """Титульна сторінка і план за зразком кафедри."""
    paras = list(doc.paragraphs)
    for p in paras:
        if p._p is body_start:
            break
        t = p.text.strip()
        if t == "Кафедра інформаційних систем та мереж":
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if t.startswith("Таблиця") and "0.1" in t or t == "Календарний план виконання курсової роботи":
            p._p.getparent().remove(p._p)
            continue
        if t.startswith("Завдання прийнято до виконання") or t.startswith("Керівник роботи"):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if t.startswith("МАШИННОГО НАВЧАННЯ (НА ПРИКЛАДІ"):
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            p.runs[0].text = TOPIC_TAIL
        for r in p.runs:
            r.font.color.rgb = BLACK
            if "РОЗВ'ЯЗАННЯ" in r.text:
                r.text = r.text.replace("РОЗВ'ЯЗАННЯ", "РОЗВ’ЯЗАННЯ")
        _zero_spacing(p.paragraph_format, line=p.paragraph_format.line_spacing or 1.5)
    for t in doc.tables[:1]:
        for row, date in zip(t.rows[1:], PLAN_DATES):
            cell = row.cells[1].paragraphs[0]
            cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if cell.runs:
                cell.runs[0].text = date
            else:
                cell.add_run(date)
        for row in t.rows:
            for cell in row.cells:
                for cp in cell.paragraphs:
                    _zero_spacing(cp.paragraph_format, line=1.0)
                    for r in cp.runs:
                        _font(r, 13, bold=r.bold)


def load_doc(path=DOC_PATH):
    doc = docx.Document(str(path))
    body = doc.element.body
    # усе після початку поля змісту перебудовується; саме поле повертається до
    # заглушки, яку потім заповнює tools/docx/toc.py
    toc_start = next(p._p for p in doc.paragraphs
                     if any((el.text or "").startswith("TOC ") for el in p._p.iter(qn("w:instrText"))))
    children = list(body)
    for el in children[children.index(toc_start):]:
        if not el.tag.endswith("}sectPr"):
            body.remove(el)
    toc = doc.add_paragraph()
    _zero_spacing(toc.paragraph_format)
    run = toc.add_run()
    for kind, instr in (("begin", None), (None, 'TOC \\o "1-2" \\h \\z \\u'),
                        ("separate", None)):
        el = OxmlElement("w:fldChar" if kind else "w:instrText")
        if kind:
            el.set(qn("w:fldCharType"), kind)
        else:
            el.set(qn("xml:space"), "preserve")
            el.text = instr
        run._r.append(el)
    _font(toc.add_run("Натисніть правою кнопкою і виберіть «Оновити поле», щоб побачити зміст."), 14)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    toc.add_run()._r.append(end)
    toc_placeholder = toc._p
    _fix_title_page(doc, toc_placeholder)
    _fix_styles(doc)
    _page_numbers(doc)
    _update_fields_on_open(doc)
    return doc


def para(doc, text, first_line=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=14,
         italic=False, bold=False, line=1.5):
    p = doc.add_paragraph()
    p.alignment = align
    _zero_spacing(p.paragraph_format, line)
    p.paragraph_format.first_line_indent = FIRST_LINE if first_line else Cm(0)
    if text:
        _add_text(p, text, size, italic, bold)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    _zero_spacing(p.paragraph_format)
    p.paragraph_format.first_line_indent = Cm(0)
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.page_break_before = True
        if text.startswith("РОЗДІЛ") and ". " in text:
            head, tail = text.split(". ", 1)
            r = p.add_run(head + ".")
            _font(r, 14, bold=True)
            r.add_break(WD_BREAK.LINE)
            _font(p.add_run(tail), 14, bold=True)
            return p
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.first_line_indent = FIRST_LINE
    _font(p.add_run(text), 14, bold=True)
    return p


def code_block(doc, tag):
    """Код комірки notebook текстом: TNR 12, ліворуч, без відступу, інтервал 1."""
    for line in _CODE[tag].rstrip("\n").split("\n"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _zero_spacing(p.paragraph_format, line=1.0)
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.keep_with_next = True
        if line.strip():
            indent = len(line) - len(line.lstrip(" "))
            _font(p.add_run("\u00a0" * indent + line.strip()), 12)
    para(doc, "", first_line=False).paragraph_format.keep_with_next = True


def figure(doc, chapter, tag, caption, with_code=True):
    """Код комірки (за потреби) і рисунок з її виводом, з підписом Рис. N.M."""
    if with_code:
        code_block(doc, tag)
    FIG_COUNTER[chapter] = FIG_COUNTER.get(chapter, 0) + 1
    number = f"{chapter}.{FIG_COUNTER[chapter]}"
    size = _SIZES[tag]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _zero_spacing(p.paragraph_format, line=1.0)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / f"{tag}.png"), width=Cm(size["width_cm"]))
    para(doc, f"Рис. {number}. {caption}", first_line=False, align=WD_ALIGN_PARAGRAPH.CENTER,
         italic=True)
    para(doc, "", first_line=False)
    USED_TAGS.append(tag)
    return number


def table(doc, number, title, rows, widths_cm):
    """Таблиця 13 пт: номер праворуч курсивом, назва по центру жирним, шапка
    повторюється на кожній сторінці, рядки не розриваються, коротка таблиця
    тримається разом з назвою на одній сторінці."""
    label = para(doc, f"Таблиця {number}", first_line=False, align=WD_ALIGN_PARAGRAPH.RIGHT,
                 italic=True)
    label.paragraph_format.keep_with_next = True
    cap = para(doc, title, first_line=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.autofit = False
    t.alignment = 1
    margins = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(int(Cm(0.1).twips)))
        m.set(qn("w:type"), "dxa")
        margins.append(m)
    t._tbl.tblPr.append(margins)
    for grid_col, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths_cm):
        grid_col.set(qn("w:w"), str(int(Cm(w).twips)))
    keep_together = len(rows) <= 10
    for r_i, row in enumerate(rows):
        tr_pr = t.rows[r_i]._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if r_i == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for c_i, value in enumerate(row):
            cell = t.rows[r_i].cells[c_i]
            cell.width = Cm(widths_cm[c_i])
            cp = cell.paragraphs[0]
            _zero_spacing(cp.paragraph_format, line=1.0)
            cp.paragraph_format.first_line_indent = Cm(0)
            cp.paragraph_format.keep_with_next = keep_together and r_i < len(rows) - 1
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER if r_i == 0 else WD_ALIGN_PARAGRAPH.LEFT
            _add_text(cp, str(value), 13, bold=(r_i == 0))
    para(doc, "", first_line=False)
    return t


def unused_figures():
    return sorted(set(_SIZES) - set(USED_TAGS))
