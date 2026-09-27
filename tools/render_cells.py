# -*- coding: utf-8 -*-
"""
Рендерить кожну кодову комірку виконаного notebook в окремий PNG для
вставки у текст курсової роботи.

Notebook конвертується в HTML, сторінка відкривається у Chromium через
Playwright, і кожна кодова комірка знімається окремим скриншотом у
figures/cell_XX.png. Нумерація йде підряд за кодовими комірками.

Запуск: python3 tools/render_cells.py
"""
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "Пашко_coursework.ipynb"
FIGURES = ROOT / "figures"
HTML = Path("/tmp/coursework_render.html")


def to_html():
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--no-input",
                    "--no-prompt", str(NOTEBOOK), "--output", str(HTML)],
                   check=True, capture_output=True)


def to_html_with_input():
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html",
                    str(NOTEBOOK), "--output", str(HTML)], check=True, capture_output=True)


async def shoot():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
            args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1500, "height": 1200},
                                      device_scale_factor=2)
        await page.goto(HTML.as_uri())
        await page.wait_for_timeout(1500)
        cells = await page.query_selector_all(".jp-Cell.jp-CodeCell")
        print("Знайдено кодових комірок:", len(cells))
        saved = 0
        for i, cell in enumerate(cells, start=1):
            box = await cell.bounding_box()
            if not box or box["height"] < 20:
                continue
            saved += 1
            out = FIGURES / f"cell_{saved:02d}.png"
            await cell.screenshot(path=str(out))
        await browser.close()
        return saved


def main():
    if FIGURES.exists():
        for old in FIGURES.glob("cell_*.png"):
            old.unlink()
    FIGURES.mkdir(exist_ok=True)
    to_html_with_input()
    saved = asyncio.run(shoot())
    print(f"Збережено {saved} зображень у {FIGURES}")


if __name__ == "__main__":
    main()
