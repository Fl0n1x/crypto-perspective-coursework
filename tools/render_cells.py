# -*- coding: utf-8 -*-
"""
Рендерить вивід кожної кодової комірки виконаного notebook в окремий PNG.

Код комірок у Word вставляється текстом (вимога до оформлення коду), тому тут
знімається лише вивід. Увесь текст виводу перефарбовується в чорний, порожні
поля обрізаються. Масштаб єдиний для всіх рисунків: 1 CSS-піксель = CM_PER_PX
см у документі, тож моноширинний текст виводу (13 px) друкується приблизно
шрифтом 10 пт. Розміри для вставки записуються у figures/sizes.json.

Запуск: python3 tools/render_cells.py
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "Пашко_coursework.ipynb"
FIGURES = ROOT / "figures"
HTML = Path("/tmp") / "coursework_render.html"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
SCALE = 2
VIEWPORT = 760
CM_PER_PX = 0.0271
MAX_W_CM, MAX_H_CM = 17.0, 21.0

CSS = """
.jp-InputArea, .jp-InputPrompt, .jp-OutputPrompt, .jp-Collapser { display: none !important; }
body, .jp-Notebook { background: #fff !important; padding: 0 !important; margin: 0 !important; }
.jp-Cell { padding: 0 !important; margin: 0 !important; }
.jp-OutputArea-output, .jp-OutputArea-output * { color: #000 !important; }
.jp-OutputArea-output pre { font-size: 13px !important; line-height: 1.3 !important;
  white-space: pre !important; }
.jp-RenderedHTMLCommon table { font-size: 12.5px !important; }
.jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { white-space: normal !important;
  padding: 2px 6px !important; }
.jp-RenderedHTMLCommon tbody tr:nth-child(odd) { background: #f2f2f2 !important; }
"""


def trim(path):
    img = Image.open(path).convert("RGB")
    diff = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255)))
    box = diff.point(lambda v: 255 if v > 12 else 0).getbbox()
    if box:
        pad = 6 * SCALE
        box = (max(0, box[0] - pad), max(0, box[1] - pad),
               min(img.width, box[2] + pad), min(img.height, box[3] + pad))
        img = img.crop(box)
    img.save(path)
    return img.size


async def shoot(tags):
    from playwright.async_api import async_playwright

    sizes = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": VIEWPORT, "height": 1200},
                                      device_scale_factor=SCALE)
        await page.goto(HTML.as_uri())
        await page.add_style_tag(content=CSS)
        await page.wait_for_timeout(1200)
        cells = await page.query_selector_all(".jp-Cell.jp-CodeCell")
        assert len(cells) == len(tags), f"комірок у HTML {len(cells)}, у notebook {len(tags)}"
        for tag, cell in zip(tags, cells):
            out = await cell.query_selector(".jp-OutputArea")
            if out is None or not (await out.inner_text()).strip() and \
                    not await out.query_selector("img"):
                continue
            # широкий вивід не повинен обрізатися прокруткою: розширюємо вікно до вмісту
            need = await out.evaluate(
                "e => Math.max(...[e, ...e.querySelectorAll('*')].map(x => x.scrollWidth))")
            if need > VIEWPORT + 1:
                await page.set_viewport_size({"width": need + 16, "height": 1200})
            path = FIGURES / f"{tag}.png"
            await out.screenshot(path=str(path))
            await page.set_viewport_size({"width": VIEWPORT, "height": 1200})
            w, h = trim(path)
            w_cm = (w / SCALE) * CM_PER_PX
            h_cm = (h / SCALE) * CM_PER_PX
            k = min(1.0, MAX_W_CM / w_cm, MAX_H_CM / h_cm)
            sizes[tag] = {"width_cm": round(w_cm * k, 2), "height_cm": round(h_cm * k, 2),
                          "font_pt_approx": round(10 * k, 1)}
        await browser.close()
    return sizes


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tags = [c["metadata"]["tag"] for c in nb["cells"] if c["cell_type"] == "code"]
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NOTEBOOK),
                    "--output", str(HTML)], check=True, capture_output=True)
    FIGURES.mkdir(exist_ok=True)
    for old in FIGURES.glob("*.png"):
        old.unlink()
    sizes = asyncio.run(shoot(tags))
    (FIGURES / "sizes.json").write_text(json.dumps(sizes, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    small = {t: s["font_pt_approx"] for t, s in sizes.items() if s["font_pt_approx"] < 8}
    print(f"Збережено {len(sizes)} рисунків у {FIGURES}")
    print("Рисунки з текстом дрібнішим за 8 пт:", small or "немає")


if __name__ == "__main__":
    main()
