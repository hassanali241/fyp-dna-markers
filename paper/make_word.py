"""Builds Word versions of both papers: python3 paper/make_word.py
Needs pandoc and a finished LaTeX build (the .aux files give figure/table numbers)."""
import os, re, subprocess, sys
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "word")
os.makedirs(os.path.join(W, "png"), exist_ok=True)

for f in os.listdir(os.path.join(HERE, "figures")):
    if f.endswith(".pdf"):
        out = os.path.join(W, "png", f[:-4])
        if not os.path.exists(out + ".png"):
            subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", os.path.join(HERE, "figures", f), out], check=True)

TITLE = "Where to Put the Anchor: Synchronization Markers for Transformer-Based DNA Trace Reconstruction"
AUTHORS = "Hassan Ali*, Muhammad Haad*, Mahek Kalani*, and Muhammad Ismail Mangrio"
AFFIL = "Department of Computer Science, Sukkur IBA University, Sukkur, Pakistan"
EQUAL = "*These authors contributed equally."
KEYWORDS = "DNA data storage, trace reconstruction, synchronization markers, deletion errors, transformers, explainable AI"


def read(name):
    return open(os.path.join(HERE, name)).read()


def resolve_if(text, extended):
    # \ifextended A \else B \fi  (no nesting in our sources)
    pat = re.compile(r"\\ifextended(\{\})?(.*?)(?:\\else(.*?))?\\fi", re.S)
    while pat.search(text):
        text = pat.sub(lambda m: (m.group(2) if extended else (m.group(3) or "")), text, count=1)
    return text



SYM = {r"\times": "×", r"\le": "≤", r"\ge": "≥", r"\in": "∈", r"\rho": "ρ", r"\beta": "β", r"\Delta": "Δ",
       r"\cdot": "·", r"\pm": "±", r"\approx": "≈", r"\,": " "}


def math_to_text(m):
    x = m.group(1)
    x = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", x)
    for k in sorted(SYM, key=len, reverse=True):
        x = x.replace(k, SYM[k])
    x = re.sub(r"\^\{([^}]*)\}", r"\\textsuperscript{\1}", x)
    x = re.sub(r"\^(\S)", r"\\textsuperscript{\1}", x)
    x = re.sub(r"_\{([^}]*)\}", r"\\textsubscript{\1}", x)
    x = re.sub(r"_(\w)", r"\\textsubscript{\1}", x)
    x = x.replace("-", "−") if re.fullmatch(r"[\s\d\-\.]+", x or "x") is None else x
    return x


def plain_math(t):
    t = re.sub(r"\\begin\{equation\}\s*(.*?)\s*\\end\{equation\}",
               lambda m: "\n\n\\begin{center}" + math_to_text(re.match(r"(.*)", m.group(1).strip().rstrip(","), re.S)) +
               "  (1)\\end{center}\n\n", t, flags=re.S)
    t = re.sub(r"\$([^$]+)\$", math_to_text, t)
    return t


def labels(version):
    lab = {}
    for line in open(os.path.join(HERE, "paper_%s.aux" % version)):
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", line)
        if m:
            lab[m.group(1)] = m.group(2)
    return lab


def build(version):
    ext = version == "ieee"
    lab = labels(version)
    t = read("body.tex")
    t = resolve_if(t, ext)
    t = t.replace("\\input{table_results.tex}", read("table_results.tex"))
    t = t.replace("\\input{table_designs.tex}", read("table_designs.tex"))
    t = re.sub(r"\\resizebox\{[^}]*\}\{!\}\{(\\begin\{tabular\}.*?\\end\{tabular\})\s*\}", r"\1", t, flags=re.S)
    t = re.sub(r"\\begin\{(widefigure|pfigure)\}(\[[^\]]*\])?", r"\\begin{figure}", t)
    t = re.sub(r"\\end\{(widefigure|pfigure)\}", r"\\end{figure}", t)
    t = re.sub(r"\\begin\{(widetable|ptable)\}(\[[^\]]*\])?", r"\\begin{table}", t)
    t = re.sub(r"\\end\{(widetable|ptable)\}", r"\\end{table}", t)
    t = re.sub(r"\\includegraphics\[[^\]]*\]\{figures/([^}]*)\.pdf\}", lambda m: "\\includegraphics{%s}" %
               os.path.join(W, "png", m.group(1) + ".png"), t)
    figword, tabword = ("Fig.", "TABLE") if ext else ("Fig.", "Table")

    # number captions from the LaTeX build
    def cap(env_text, kind):
        m = re.search(r"\\label\{([^}]*)\}", env_text)
        num = lab.get(m.group(1), "?") if m else "?"
        return env_text.replace("\\caption{", "\\caption{%s %s. " % (figword if kind == "figure" else tabword, num), 1)
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", lambda m: cap(m.group(0), "figure"), t, flags=re.S)
    t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", lambda m: cap(m.group(0), "table"), t, flags=re.S)
    t = re.sub(r"\\ref\{([^}]*)\}", lambda m: lab.get(m.group(1), "?"), t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    t = t.replace("\\setlength{\\tabcolsep}{3pt}", "").replace("\\setlength{\\tabcolsep}{3.2pt}", "")
    t = t.replace("\\setlength{\\tabcolsep}{2.6pt}", "").replace("\\footnotesize", "")
    t = plain_math(t)
    abstract = plain_math(read("abstract.tex").strip())
    doc = ("\\documentclass{article}\n\\begin{document}\n\\begin{abstract}\n%s\n\\end{abstract}\n%s\n\\end{document}\n"
           % (abstract, t))
    src = os.path.join(W, "flat_%s.tex" % version)
    open(src, "w").write(doc)
    out = os.path.join(W, "paper_%s.docx" % version)
    csl = os.path.join(W, "ieee.csl" if ext else "lncs.csl")
    subprocess.run(["pandoc", src, "-f", "latex", "-t", "docx", "--citeproc", "--bibliography", os.path.join(HERE, "refs.bib"),
                    "--csl", csl, "-M", "reference-section-title=References", "-M", "link-citations=true", "--number-sections",
                    "-o", out], check=True, cwd=W)
    style(out, ext)
    return out


def style(path, ext):
    d = Document(path)
    base = "Times New Roman"
    for s in d.styles:
        try:
            s.font.name = base
        except Exception:
            pass
    d.styles["Normal"].font.size = Pt(10)
    for name, size in [("Heading 1", 12), ("Heading 2", 10.5), ("Heading 3", 10)]:
        try:
            st = d.styles[name]
        except KeyError:
            continue
        st.font.size = Pt(size); st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
    sec = d.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.75 if ext else 1.2)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    # title block at the top
    first = d.paragraphs[0]
    for text, size, bold, italic in [(TITLE, 18, True, False), (AUTHORS, 11, False, False), (AFFIL, 10, False, True),
                                     (EQUAL, 9, False, False)]:
        p = first.insert_paragraph_before()
        r = p.add_run(text); r.font.size = Pt(size); r.bold = bold; r.italic = italic; r.font.name = base
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # keywords after the abstract
    for i, p in enumerate(d.paragraphs):
        if p.style.name == "Abstract" or (i > 3 and p.text.startswith("Deletions are the hardest")):
            k = d.paragraphs[i + 1].insert_paragraph_before() if i + 1 < len(d.paragraphs) else d.add_paragraph()
            r = k.add_run(("Index Terms—" if ext else "Keywords: ") + KEYWORDS); r.italic = True; r.font.size = Pt(9)
            break
    for p in d.paragraphs:
        if p.text.startswith(("Fig.", "TABLE", "Table")) and len(p.text) < 600:
            for r in p.runs:
                r.font.size = Pt(8.5)
    for tbl in d.tables:
        for row in tbl.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(8)
    for p in d.paragraphs:
        if p.style.name.lower().startswith(("heading", "title", "abstract title")):
            for r in p.runs:
                r.font.color.rgb = RGBColor(0, 0, 0); r.font.name = base
    d.save(path)


if __name__ == "__main__":
    for v in ["ieee", "lncs"]:
        print("wrote", build(v))
