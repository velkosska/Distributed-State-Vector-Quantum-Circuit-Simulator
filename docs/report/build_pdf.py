#!/usr/bin/env python3
"""Render docs/report/final_report.md to PDF (Markdown -> HTML -> headless Chrome).

  python3 docs/report/build_pdf.py [out.pdf]

Needs the `markdown` package and Google Chrome or Chromium.
"""
import os, re, subprocess, sys, shutil
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "final_report.md")
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Group05_Part2_Final.pdf"))

text = open(SRC, encoding="utf-8").read()
meta = {}
m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
if m:
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":"); meta[k.strip()] = v.strip().strip('"')
    text = text[m.end():]
text = text.replace("\\newpage", '<div class="pb"></div>')
# ![cap](src){width=60%} -> figure with caption
def fig(mm):
    cap, src, w = mm.group(1), mm.group(2), mm.group(3) or "70%"
    return f'<figure style="width:{w}"><img src="{src}"><figcaption>{cap}</figcaption></figure>'
text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)(?:\{width=([0-9.]+%)\})?", fig, text)
body = markdown.markdown(text, extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 15mm 16mm 15mm 16mm; }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #18181b; }
h1.t { font-size: 17pt; margin: 0 0 2pt; } .sub { color: #52525b; margin: 0 0 8pt; font-size: 10.5pt; }
h1 { font-size: 14pt; margin: 10pt 0 4pt; } h2 { font-size: 11.5pt; margin: 9pt 0 3pt; border-bottom: 0.5pt solid #d4d4d8; padding-bottom: 1pt; }
p { margin: 3pt 0 5pt; text-align: justify; } ul { margin: 2pt 0 4pt; }
table { border-collapse: collapse; margin: 4pt 0 7pt; font-size: 8.4pt; width: 100%; }
th, td { border-bottom: 0.5pt solid #d4d4d8; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 1pt solid #71717a; font-weight: 600; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.2pt; background: #f4f4f5; padding: 0 2px; }
pre { background: #f4f4f5; padding: 5pt 7pt; font-size: 7.8pt; line-height: 1.25; overflow: hidden; }
pre code { background: none; padding: 0; }
figure { margin: 5pt auto 7pt; text-align: center; page-break-inside: avoid; }
figure img { width: 100%; } figcaption { font-size: 8.2pt; color: #52525b; margin-top: 2pt; }
.pb { page-break-after: always; }
"""
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
<h1 class="t">{meta.get('title','')}</h1><p class="sub">{meta.get('subtitle','')}</p>
{body}</body></html>"""
tmp = os.path.join(HERE, ".report_tmp.html")
open(tmp, "w", encoding="utf-8").write(html)
chrome = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("chromium") or "", shutil.which("google-chrome") or "",
                           "/opt/pw-browsers/chromium"] if c and os.path.exists(c)), None)
if not chrome:
    sys.exit("no Chrome/Chromium found")
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={OUT}", "file://" + tmp], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(tmp)
print("wrote", OUT)
