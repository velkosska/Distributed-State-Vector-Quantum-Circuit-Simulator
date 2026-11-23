#!/usr/bin/env python3
"""Convert docs/report/final_report.md to a standalone LaTeX file (Overleaf-ready).

  python3 docs/report/md2tex.py            # writes docs/report/latex/final_report.tex
Handles only what the report uses: front matter, #/## headings, paragraphs, **bold**, *italic*,
`code`, pipe tables, fenced code, ![cap](src){width=..%} figures and \\newpage.
Figures are copied next to the .tex so the folder can be zipped and uploaded.
"""
import os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "final_report.md")
OUTDIR = os.path.join(HERE, "latex")
os.makedirs(os.path.join(OUTDIR, "figures"), exist_ok=True)
text = open(SRC, encoding="utf-8").read()

meta = {}
m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
if m:
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":"); meta[k.strip()] = v.strip().strip('"')
    text = text[m.end():]

UNI = {"≈": r"$\approx$", "×": r"$\times$", "→": r"$\rightarrow$", "─": "-", "▶": ">", "≤": r"$\le$",
       "≥": r"$\ge$", "⁻": "", "ⁿ": "", "ᵖ": "", "π": r"$\pi$", "–": "--", "—": "---", "·": r"$\cdot$",
       "ψ": r"$\psi$", "Δ": r"$\Delta$", "├": "|", "└": "`", "…": r"\ldots{}", "²": r"$^2$", "§": r"\S{}", "Σ": r"$\Sigma$", "₀": r"$_0$"}

def esc(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s

def sup(s):  # 2ⁿ, 10⁻¹⁰, 2ⁿ⁻¹, P = 2ᵖ
    s = re.sub(r"10⁻¹⁰", r"$10^{-10}$", s); s = re.sub(r"10⁻¹⁴", r"$10^{-14}$", s)
    s = re.sub(r"10⁻⁸", r"$10^{-8}$", s); s = re.sub(r"10⁴", r"$10^{4}$", s)
    s = s.replace("2ⁿ⁻¹", r"$2^{n-1}$").replace("2ⁿ", r"$2^{n}$").replace("2ᵖ", r"$2^{p}$")
    return s

def inline(s):
    out, parts = [], re.split(r"(`[^`]+`)", s)
    for p in parts:
        if p.startswith("`") and p.endswith("`"):
            out.append(r"\texttt{" + esc(p[1:-1]) + "}")
        else:
            p = sup(p)
            segs = re.split(r"(\$[^$]*\$)", p)
            buf = []
            for g in segs:
                if g.startswith("$") and g.endswith("$") and len(g) > 1:
                    buf.append(g); continue
                g = esc(g)
                for a, b in UNI.items(): g = g.replace(a, b)
                g = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", g)
                g = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", g)
                buf.append(g)
            out.append("".join(buf))
    return "".join(out)

lines, i, body = text.splitlines(), 0, []
while i < len(lines):
    l = lines[i]
    if l.strip() == "\\newpage":
        body.append(r"\newpage"); i += 1; continue
    if l.startswith("```"):
        j = i + 1; code = []
        while not lines[j].startswith("```"): code.append(lines[j]); j += 1
        c = "\n".join(code)
        for a, b in {"─": "-", "▶": ">", "├": "+", "└": "+"}.items(): c = c.replace(a, b)
        body.append("\\begin{verbatim}\n" + c + "\n\\end{verbatim}"); i = j + 1; continue
    if l.startswith("# "):
        body.append(r"\section*{" + inline(l[2:]) + "}"); i += 1; continue
    if l.startswith("## "):
        h = re.sub(r"^\d+\.\s*", "", l[3:])
        body.append(r"\section{" + inline(h) + "}"); i += 1; continue
    mm = re.match(r"!\[([^\]]*)\]\(([^)]+)\)(?:\{width=([0-9.]+)%\})?", l.strip())
    if mm:
        cap, src, w = mm.group(1), mm.group(2), float(mm.group(3) or 70) / 100
        name = os.path.basename(src); shutil.copy(os.path.normpath(os.path.join(HERE, src)), os.path.join(OUTDIR, "figures", name))
        cap = re.sub(r"^Figure \d+\.\s*", "", cap)
        body.append("\\begin{figure}[ht]\\centering\n\\includegraphics[width=%.2f\\linewidth]{figures/%s}\n\\caption{%s}\n\\end{figure}" % (w, name, inline(cap)))
        i += 1; continue
    if l.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
        head, rows = rows[0], [r for r in rows[2:]]
        n = len(head)
        spec = "@{}" + "".join("X" if k in (0, n - 1) or n <= 3 else "l" for k in range(n)) + "@{}"
        t = ["\\begin{table}[ht]\\centering\\small", "\\begin{tabularx}{\\linewidth}{%s}" % spec, "\\toprule",
             " & ".join(r"\textbf{" + inline(c) + "}" for c in head) + r" \\", "\\midrule"]
        t += [" & ".join(inline(c) for c in r) + r" \\" for r in rows]
        t += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
        body.append("\n".join(t)); continue
    if l.strip() == "":
        body.append(""); i += 1; continue
    par = [l]; i += 1
    while i < len(lines) and lines[i].strip() and not re.match(r"(#|\||```|!\[|\\newpage)", lines[i]):
        par.append(lines[i]); i += 1
    body.append(inline(" ".join(par)))

tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,graphicx,booktabs,tabularx,float,xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{8pt}{3pt}
\setlength{\parskip}{3pt}\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}
\title{\vspace{-12mm}%s\\[2pt]\large %s}
\date{}
\begin{document}
\maketitle
\vspace{-12mm}
%s
\end{document}
""" % (inline(meta.get("title", "")), inline(meta.get("subtitle", "")), "\n\n".join(body))
out = os.path.join(OUTDIR, "final_report.tex")
open(out, "w", encoding="utf-8").write(tex)
print("wrote", out)
