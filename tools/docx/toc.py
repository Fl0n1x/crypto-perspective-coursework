# -*- coding: utf-8 -*-
"""
Заповнює зміст курсової роботи готовими рядками з номерами сторінок.

Поле TOC у документі залишається полем (у Word його можна оновити), але його
збережений результат містить реальні рядки змісту, тому зміст видно одразу,
без ручного оновлення. Номери сторінок визначаються з PDF, отриманого через
LibreOffice (Liberation Serif метрично сумісний з Times New Roman). Після
заповнення документ конвертується повторно, доки номери не стабілізуються.

Запуск: python3 tools/docx/toc.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import docx
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

sys.path.insert(0, str(Path(__file__).parent))
from common import DOC_PATH, _font, _zero_spacing  # noqa: E402

TEXT_WIDTH_CM = 18.0


def headings(doc):
    out = []
    for p in doc.paragraphs:
        if p.style.name in ("Heading 1", "Heading 2") and p.text.strip():
            out.append((int(p.style.name[-1]), " ".join(p.text.split())))
    return out


def toc_paragraph(doc):
    for p in doc.paragraphs:
        if any((el.text or "").startswith("TOC ") for el in p._p.iter(qn("w:instrText"))):
            return p
    raise RuntimeError("поле TOC не знайдено")


def _field_run(p, kind=None, instr=None):
    run = p.add_run()
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = instr
    run._r.append(el)


def fill(doc, pages):
    """Замінює поле TOC полем з рядками змісту; pages - номери сторінок у порядку заголовків."""
    old = toc_paragraph(doc)
    # прибрати попередньо заповнений зміст: усі абзаци від початку поля до його кінця
    nodes, el = [old._p], old._p
    while not any(f.get(qn("w:fldCharType")) == "end" for f in el.iter(qn("w:fldChar"))):
        el = el.getnext()
        nodes.append(el)
    anchor = nodes[-1].getnext()
    for n in nodes:
        n.getparent().remove(n)

    entries = headings(doc)
    for i, ((level, text), page) in enumerate(zip(entries, pages)):
        p = doc.add_paragraph()
        anchor.addprevious(p._p)
        _zero_spacing(p.paragraph_format)
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.left_indent = Cm(0 if level == 1 else 0.5)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT,
                                                  WD_TAB_LEADER.DOTS)
        if i == 0:
            _field_run(p, "begin")
            _field_run(p, instr='TOC \\o "1-2" \\h \\z \\u')
            _field_run(p, "separate")
        _font(p.add_run(f"{text}\t{page}"), 14)
        if i == len(entries) - 1:
            _field_run(p, "end")


def pdf_pages(path):
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "doc.docx"
    shutil.copy(path, src)
    subprocess.run(["soffice", f"-env:UserInstallation=file://{tmp}/lo", "--headless",
                    "--convert-to", "pdf", "--outdir", str(tmp), str(src)],
                   check=True, capture_output=True, timeout=600)
    text = subprocess.run(["pdftotext", "-layout", str(tmp / "doc.pdf"), "-"], check=True,
                          capture_output=True, text=True).stdout
    shutil.rmtree(tmp)
    return text.split("\f")


def find_pages(doc, pages_text):
    lines = [[" ".join(l.split()) for l in page.splitlines() if l.strip()] for page in pages_text]
    start = next(i for i, page in enumerate(lines) if "ЗМІСТ" in page)
    last_entry = headings(doc)[-1][1]
    start = next(i for i in range(start, len(lines)) if last_entry in " ".join(lines[i])) + 1
    found = []
    for level, text in headings(doc):
        key = text.split(". ")[0] + "." if text.startswith("РОЗДІЛ") else text[:30]
        page = next(i for i in range(start, len(lines))
                    if any(l.startswith(key) for l in lines[i]))
        found.append(page + 1)
        start = page
    return found, len(pages_text) - (pages_text[-1].strip() == "")


def main():
    pages = None
    for attempt in range(4):
        doc = docx.Document(str(DOC_PATH))
        fill(doc, pages or ["0"] * len(headings(doc)))
        doc.save(str(DOC_PATH))
        new, total = find_pages(doc, pdf_pages(DOC_PATH))
        if new == pages:
            print(f"Зміст заповнено: {len(new)} рядків, сторінок у документі: {total}")
            for (level, text), page in zip(headings(doc), new):
                print(("  " if level == 2 else "") + f"{text[:70]} ... {page}")
            return
        pages = new
    raise RuntimeError("номери сторінок змісту не стабілізувались")


if __name__ == "__main__":
    main()
