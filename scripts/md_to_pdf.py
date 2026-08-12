#!/usr/bin/env python3
"""Convert markdown to PDF via HTML + Edge headless printing.

Usage: python scripts/md_to_pdf.py <input.md> <output.pdf>
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 22mm 18mm;
    @bottom-center { content: "- " counter(page) " -"; font-size: 9px; color: #999; }
}
body {
    font-family: "Microsoft YaHei", "SimHei", "SimSun", sans-serif;
    font-size: 10.5pt;
    line-height: 1.7;
    color: #222;
    max-width: 100%;
}
h1 { font-size: 20pt; text-align: center; margin: 0 0 6pt 0; padding-bottom: 8pt;
     border-bottom: 2px solid #333; }
h2 { font-size: 15pt; margin: 18pt 0 8pt 0; padding-bottom: 4pt;
     border-bottom: 1px solid #ccc; color: #1a1a1a; }
h3 { font-size: 12.5pt; margin: 14pt 0 6pt 0; color: #222; }
h4 { font-size: 11pt; margin: 10pt 0 4pt 0; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #aaa; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
code { font-family: "Cascadia Code", "Consolas", "Microsoft YaHei", monospace;
       background: #f4f4f4; padding: 1pt 3pt; border-radius: 2pt; font-size: 9pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; border-radius: 4pt;
      padding: 8pt 10pt; overflow-x: auto; font-size: 8.5pt; line-height: 1.5;
      white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #ccc; margin: 8pt 0; padding: 4pt 12pt;
             color: #555; background: #fafafa; }
ul, ol { margin: 4pt 0; padding-left: 20pt; }
li { margin: 2pt 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }
strong { color: #111; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def find_edge():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError("Microsoft Edge not found")


def md_to_pdf(md_path: str, pdf_path: str):
    md_text = Path(md_path).read_text(encoding="utf-8")

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = HTML_TEMPLATE.format(css=CSS, body=body)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False
    ) as f:
        f.write(html)
        html_path = f.name

    edge = find_edge()
    cmd = [
        edge,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf=" + pdf_path,
        "--print-to-pdf-no-header",
        html_path,
    ]
    print(f"Rendering with Edge headless...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    Path(html_path).unlink(missing_ok=True)

    if Path(pdf_path).exists():
        size_kb = Path(pdf_path).stat().st_size / 1024
        print(f"PDF saved: {pdf_path} ({size_kb:.0f} KB)")
    else:
        print(f"ERROR: PDF not created. stderr: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.md> <output.pdf>")
        sys.exit(1)
    md_to_pdf(sys.argv[1], sys.argv[2])
