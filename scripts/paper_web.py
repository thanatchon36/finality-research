"""Build the web reader docs/paper/ from paper/paper.tex (same text and numbers as the PDF).

pandoc renders the LaTeX source, with the generated number macros of paper/tables/numbers.tex, into
paper/reader.html: masthead, abstract, a results strip whose numbers come from the same macros, contents,
numbered sections, references and the appendices. Cross-references (\\cref, \\ref), equation, figure, table
and appendix numbers are resolved here so they match the PDF. The research-design figure is TikZ in the
PDF; for the web it is rendered once to paper/figures/fig1_design.png (tectonic + pdftoppm). The PDF and
the figures the page uses are copied next to it; GitHub Pages serves docs/ at
https://thanatchon36.github.io/finality-research/. Requires pandoc on PATH.

Usage: ``python scripts/paper_web.py``.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "docs" / "paper"
REPO_URL = "https://github.com/thanatchon36/finality-research"
PAPER_URL = "https://thanatchon36.github.io/finality-research/paper/"
INSTRUMENT_URL = "https://thanatchon36.github.io/finality-research/instrument/"
DATA_URL = "https://huggingface.co/datasets/thanatchon36/finality-research"
DESIGN_PNG = PAPER / "figures" / "fig1_design.png"

# (label, style, finding, detail) with \macro placeholders resolved from paper/tables/numbers.tex.
# style: "ref" = study result under the illustrative calibration, "null" = pending.
RESULTS = [
    ("Finality", "ref",
     r"Finality explains \shareFinality% of the log payoff gap between a bank core and a cross-chain bridge",
     r"the bank becomes worth attacking only below a recovery probability of \breakEven"),
    ("Coverage · 30 years", "ref",
     r"Nominal 95% delta-method intervals cover the 1-in-1,000-year loss \covBThirtyDeltaK% of the time",
     r"parametric bootstrap \covBThirtyBootstrapK%; every miss lies below the truth"),
    ("Bias", "ref",
     r"Median point estimate \medRatioBThirtyK times the true 1-in-1,000-year loss",
     "a 30-year record rarely contains the events that reveal how heavy the tail is"),
    ("Profile likelihood", "ref",
     r"Coverage rises to \covBThirtyProfileK% from 30-year records and \covAThreeHundredProfileK% from 300",
     r"interval score \scoreBThirtyProfileK against \scoreBThirtyDeltaK for the delta method (30 years)"),
    ("Contagion", "ref",
     r"Phase transition at an exposure of \critExp times equity, where the spectral radius crosses one",
     r"systemic importance tracks interconnection: \topTwoName, with \topTwoAssets of assets, is second most systemic"),
    ("Calibration", "null",
     "Real-data calibration pending",
     "all parameters are illustrative; recalibrate with incident data before citing any number"),
]

THEOREMS = ("proposition", "lemma", "corollary", "approximation", "definition", "remark")
KIND = {"sec": "Section", "fig": "Figure", "tab": "Table", "eq": "Equation", "prop": "Proposition",
        "lem": "Lemma", "cor": "Corollary", "rem": "Remark", "def": "Definition"}


def macros() -> dict[str, str]:
    text = (PAPER / "tables" / "numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", text, re.M))


def plain(tex: str, m: dict[str, str]) -> str:
    """Resolve \\macro placeholders to plain text (for the results strip)."""
    def sub(mm: re.Match) -> str:
        if mm.group(1) not in m:
            raise RuntimeError(f"unknown macro \\{mm.group(1)}")
        return m[mm.group(1)]
    s = re.sub(r"\\(\w+)(?:\{\})?", sub, tex)
    return s.replace(r"\$", "$").replace("$-$", "\u2212").replace(r"\%", "%")


def read_inputs(tex: str) -> str:
    """Inline \\input{sections/...} and \\tabinput{tables/...} so pandoc sees one document."""
    tex = re.sub(r"\\input\{(sections/[\w-]+)\}",
                 lambda mm: (PAPER / f"{mm.group(1)}.tex").read_text(encoding="utf-8"), tex)
    return re.sub(r"\\tabinput\{(tables/[\w.-]+)\}",
                  lambda mm: (PAPER / mm.group(1)).read_text(encoding="utf-8"), tex)


def render_design(tex: str, labels: dict[str, str]) -> None:
    """Render the TikZ research-design figure to PNG once (the PDF draws it natively)."""
    if DESIGN_PNG.exists():
        return
    pic = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, re.S).group(0)
    colors = "\n".join(re.findall(r"^\\definecolor.*$", (PAPER / "paper.tex").read_text(), re.M))
    doc = (
        "\\documentclass[border=6pt]{standalone}\n\\usepackage{fontspec}\\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]\n"
        "\\usepackage{amsmath,xcolor,tikz}\n\\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}\n"
        f"{colors}\n\\newcommand{{\\xiTrue}}{{{macros()['xiTrue']}}}\n"
        "\\begin{document}\n" + re.sub(r"\\ref\{([^}]*)\}", lambda mm: labels[mm.group(1)], pic) + "\n\\end{document}\n"
    )
    for tool in ("tectonic", "pdftoppm"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found; needed once to render {DESIGN_PNG.name}")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "design.tex").write_text(doc, encoding="utf-8")
        subprocess.run(["tectonic", "design.tex"], cwd=tmp, check=True, capture_output=True)
        subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", "design.pdf", "design"],
                       cwd=tmp, check=True)
        shutil.copyfile(Path(tmp) / "design.png", DESIGN_PNG)


def front_matter(tex: str) -> dict[str, str]:
    title = re.search(r"\{\\LARGE\\bfseries (.*?):?\\par\}", tex).group(1).rstrip(":")
    subtitle = re.search(r"\{\\Large\\bfseries (.*?)\\par\}", tex).group(1)
    subtitle = re.sub(r"\\\\(\[[^]]*\])?\s*", " ", subtitle)
    return {
        "title": title,
        "subtitle": subtitle,
        "date": re.search(r"\{\\small (\w+ \d{4})\\par\}", tex).group(1),
        "abstract": re.search(r"\\begin\{quote\}\\small\n(.*?)\n\s*\\vspace", tex, re.S).group(1),
        "keywords": ", ".join(k.strip() for k in re.search(
            r"\\textbf\{Keywords:\} (.*?)\\\\", tex).group(1).split("$\\cdot$")),
        "jel": re.search(r"\\textbf\{JEL codes:\} (.*?)\n", tex).group(1).strip(),
        "thanks": re.search(r"\\footnotetext\[1\]\{(.*?)\}\}", tex).group(1),
    }


def body_tex(tex: str) -> str:
    body = tex[tex.index("\\input{sections/01_introduction}"):tex.index("\\end{document}")]
    body = read_inputs(body)
    body = re.sub(r"\\bibliographystyle\{\w+\}\s*\\bibliography\{\w+\}", "", body)
    body = body.replace("\\appendix", "")
    body = re.sub(r"\\includegraphics(\[[^]]*\])?\{figures/([\w-]+)\.pdf\}", r"\\includegraphics{figures/\2.png}", body)
    body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                  f"\\\\includegraphics{{figures/{DESIGN_PNG.name}}}", body, flags=re.S)
    # numbered equations carry a marker that survives into MathML; resolved in number_equations()
    body = re.sub(r"\\begin\{equation\}(\\label\{([^}]*)\})?",
                  lambda mm: "\\begin{equation}\\text{EQMARK" + (mm.group(2) or "") + "}", body)
    # threeparttable notes: pandoc drops them inside a table, so they become a paragraph right after it,
    # marked TNOTES ... ENDTNOTES, which build() moves back under the table
    def table(mm: re.Match) -> str:
        env = mm.group(0)
        notes = re.search(r"\\begin\{tablenotes\}[^\n]*\n(.*?)\\end\{tablenotes\}", env, re.S)
        env = re.sub(r"\\(begin|end)\{threeparttable\}", "", env)
        if notes is None:
            return env
        env = env.replace(notes.group(0), "")
        text = re.sub(r"\\item\s*", " ", notes.group(1)).strip()
        return f"{env}\n\nTNOTES {text} ENDTNOTES\n\n"
    body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)  # pandoc misreads the (lr) trim argument
    return re.sub(r"\\begin\{table\}.*?\\end\{table\}", table, body, flags=re.S)


def header_tex(tex: str) -> str:
    """Notation macros and the generated numbers, so pandoc expands them like LaTeX does."""
    notation = tex[tex.index("% ---------------------------------------------------------------- notation"):
                   tex.index("\\makeatletter")]
    return notation + (PAPER / "tables" / "numbers.tex").read_text(encoding="utf-8")


def pandoc(tex: str, *extra: str) -> str:
    return subprocess.run(
        ["pandoc", "--from=latex", "--to=html5", "--mathml", "--wrap=none", *extra],
        input=tex, capture_output=True, text=True, check=True, cwd=str(PAPER),
    ).stdout


def letter_appendices(page: str, first: int) -> str:
    """Sections from `first` on are appendices A, B, ... as in the PDF."""
    def letter(num: str) -> str:
        head, *rest = num.split(".")
        return ".".join([chr(ord("A") + int(head) - first), *rest]) if int(head) >= first else num
    page = re.sub(r'data-number="([\d.]+)"', lambda mm: f'data-number="{letter(mm.group(1))}"', page)
    return re.sub(r'class="(header|toc)-section-number">([\d.]+)<',
                  lambda mm: f'class="{mm.group(1)}-section-number">{letter(mm.group(2))}<', page)


def number_equations(page: str, labels: dict[str, str]) -> str:
    count = 0

    def eq(mm: re.Match) -> str:
        nonlocal count
        math = mm.group(0)
        label = re.search(r"<mtext[^>]*>EQMARK([^<]*)</mtext>", math)
        if label is None:
            return math
        count += 1
        math = math.replace(label.group(0), "")
        math = re.sub(r"\\text\{EQMARK[^}]*\}", "", math)
        ident = f' id="{label.group(1)}"' if label.group(1) else ""
        if label.group(1):
            labels[label.group(1)] = str(count)
        return f'<span class="eq"{ident}>{math}<span class="eqno">({count})</span></span>'

    page = re.sub(r'<math display="block".*?</math>', eq, page, flags=re.S)
    if "EQMARK" in page:
        raise RuntimeError("unresolved equation marker")
    return page


def resolve_refs(page: str, labels: dict[str, str], appendix: set[str]) -> str:
    def name(key: str, n: int, cap: bool) -> str:
        prefix = key.split(":")[0]
        word = "Appendix" if key in appendix else KIND.get(prefix, prefix)
        if prefix == "app" and key not in appendix:
            word = "Approximation"
        if n > 1:
            word += "es" if word.endswith("x") else "s"
        return word if cap else word.lower()

    def show(key: str) -> str:
        if key not in labels:
            raise RuntimeError(f"unresolved reference {key}")
        return f"({labels[key]})" if key.startswith("eq:") else labels[key]

    def ref(mm: re.Match) -> str:
        kind, keys = mm.group(1), mm.group(2).split(",")
        nums = [show(k) for k in keys]
        if len(nums) == 1:
            text = nums[0]
        elif len(nums) == 2:
            text = f"{nums[0]} and {nums[1]}"
        else:
            text = ", ".join(nums[:-1]) + f" and {nums[-1]}"
        if kind != "ref":
            text = f"{name(keys[0], len(keys), True)} {text}"
        return f'<a href="#{keys[0]}">{text}</a>'

    page = re.sub(r'<a href="#[^"]*" data-reference-type="([^"]+)" data-reference="([^"]+)">[^<]*</a>', ref, page)
    left = re.findall(r'data-reference="[^"]+"', page)
    if left:
        raise RuntimeError(f"unresolved cross-references: {left[:5]}")
    return page


def build() -> Path:
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found; install pandoc to build the web reader")
    tex = (PAPER / "paper.tex").read_text(encoding="utf-8")
    m = macros()
    fm = front_matter(tex)
    head = header_tex(tex)
    page = pandoc(
        head + "\n\\begin{document}\n" + body_tex(tex) + "\n\\end{document}\n",
        "--standalone", "--template=reader.html", "--number-sections", "--toc", "--toc-depth=1",
        "--citeproc", "--bibliography=refs.bib", "--metadata=reference-section-title:References",
        "--metadata=link-citations:true",
        f"--metadata=pagetitle:{fm['title']}",
        "--metadata=description:Preprint: finality, extreme-value uncertainty and contagion in banking-sector cyber risk.",
        f"--metadata=title:{fm['title']}", f"--metadata=subtitle:{fm['subtitle']}",
        "--metadata=author:Thanatchon Chongmankhong", f"--metadata=date:{fm['date']}",
        f"--metadata=jel:{fm['jel']}", *(f"--metadata=keywords:{k.strip()}" for k in fm["keywords"].split(",")),
    )
    abstract = pandoc(head + "\n\\begin{document}\n" + fm["abstract"] + "\n\\end{document}\n")
    thanks = pandoc("\\begin{document}\n" + fm["thanks"] + "\n\\end{document}\n").strip()

    # the reference list sits between the main text and the appendices, as in the PDF
    refs_start = page.index('<h1 class="unnumbered" id="bibliography">')
    refs_end = page.index("</article>")
    refs = page[refs_start:refs_end]
    page = page[:refs_start] + page[refs_end:]
    page = page.replace('<h1 data-number="9" id="app:derivation">', refs + '<h1 data-number="9" id="app:derivation">', 1)
    toc_refs = re.search(r'<li><a href="#bibliography"[^>]*>.*?</li>\n?', page)
    page = page.replace(toc_refs.group(0), "", 1)
    page = re.sub(r'(<li><a href="#app:derivation")', lambda mm: toc_refs.group(0) + mm.group(1), page, count=1)

    first_app = int(re.search(r'<h1 data-number="(\d+)" id="app:derivation"', page).group(1))
    page = letter_appendices(page, first_app)

    labels: dict[str, str] = {}
    for num, ident in re.findall(r'<h\d data-number="([\w.]+)" id="([^"]+)"', page):
        labels[ident] = num
    appendix = {k for k, v in labels.items() if v[0].isalpha()}
    for i, ident in enumerate(re.findall(r'<figure id="(fig:[^"]+)"', page), 1):
        labels[ident] = str(i)
    for i, ident in enumerate(re.findall(r'<div id="(tab:[^"]+)"', page), 1):
        labels[ident] = str(i)
    # theorem-like blocks: pandoc keeps only an anchor, so number them per kind (as amsthm does) with their titles
    body = body_tex(tex)
    seen = {k: 0 for k in THEOREMS}
    for kind, title, ident in re.findall(
            rf"\\begin\{{({'|'.join(THEOREMS)})\}}(?:\[([^\]]*)\])?\s*\\label\{{([^}}]*)\}}", body):
        seen[kind] += 1
        labels[ident] = str(seen[kind])
        head = f"<strong>{kind.capitalize()} {seen[kind]}</strong>" + (f" (<em>{html.escape(title)}</em>)." if title else ".")
        anchor = re.compile(rf'<div class="{kind}">\s*<p><span id="{re.escape(ident)}" data-label="[^"]*"></span>\s*')
        page, n = anchor.subn(f'<div class="{kind}" id="{ident}">\n<p>{head} ', page, count=1)
        if n != 1:
            raise RuntimeError(f"theorem block {ident} not found")
    page = number_equations(page, labels)
    render_design(read_inputs(tex), labels)
    page = resolve_refs(page, labels, appendix)

    # captions: "Figure n." / "Table n." in document order, as in the PDF
    count = {"figcaption": 0, "caption": 0}

    def caption(mm: re.Match) -> str:
        count[mm.group(1)] += 1
        word = "Figure" if mm.group(1) == "figcaption" else "Table"
        return f"<{mm.group(1)}{mm.group(2)}>{mm.group(3) or ''}<b>{word} {count[mm.group(1)]}.</b> "
    page = re.sub(r"<(caption|figcaption)(\b[^>]*)>(\s*<p>)?", caption, page)
    page = re.sub(r"<img\b[^>]*>", lambda mm: re.sub(r'\s(?:style|width|height)="[^"]*"', "", mm.group(0))
                  .replace("<img ", '<img loading="lazy" ', 1), page)
    page = re.sub(r"<table\b[^>]*>\s*<caption>(.*?)</caption>(.*?)</table>",
                  r'<div class="tbl"><p class="tcap">\1</p><div class="tscroll"><table>\2</table></div></div>',
                  page, flags=re.S)
    page = re.sub(r"<colgroup>.*?</colgroup>\s*", "", page, flags=re.S)
    page = re.sub(r"<p>TNOTES (.*?) ENDTNOTES</p>", r'<div class="tnotes"><p>\1</p></div>', page, flags=re.S)
    page = re.sub(r"TNOTES(.*?)ENDTNOTES", r'<div class="tnotes"><p>\1</p></div>', page, flags=re.S)
    if "TNOTES" in page:
        raise RuntimeError("unresolved table notes")

    items = "\n".join(
        f'<li class="r r-{style}"><span class="r-id">{html.escape(label)}</span>'
        f'<span class="r-text">{html.escape(plain(finding, m))}<small>{html.escape(plain(detail, m))}</small></span></li>'
        for label, style, finding, detail in RESULTS
    )
    actions = (
        '<li><a href="paper.pdf">Download PDF</a></li>'
        f'<li><a href="{REPO_URL}">Code and replication</a></li>'
        f'<li><a href="{INSTRUMENT_URL}">Interactive instrument</a></li>'
        f'<li><a href="{DATA_URL}">Data (Hugging Face)</a></li>'
    )
    full_title = f"{fm['title']}: {fm['subtitle']}"
    bib = (
        "@misc{chongmankhong2026false,\n"
        "  author = {Chongmankhong, Thanatchon},\n"
        f"  title  = {{{full_title}}},\n"
        "  year   = {2026},\n"
        "  note   = {Preprint},\n"
        f"  url    = {{{PAPER_URL}}}\n"
        "}"
    )
    colophon = (
        "<p>Rendered from <code>paper/paper.tex</code> by <code>scripts/paper_web.py</code> "
        "(<code>make web</code>). Every number comes from the macros that <code>paper/analysis.py</code> "
        "writes to <code>paper/tables/numbers.tex</code>, the same ones the PDF uses; the PDF is authoritative. "
        "All model parameters are illustrative.</p>"
    )
    for marker, value in (
        ("<!--ABSTRACT-->", abstract),
        ("<!--ACTIONS-->", actions),
        ("<!--RESULTS-NOTE-->", (
            "Headline numbers, generated from the study's results. The coverage results are methodological and "
            "hold against a planted truth; the attacker and contagion numbers depend on illustrative parameters. "
            "Amber: study results; grey: pending."
        )),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"),
        ("<!--COLOPHON-->", thanks + "\n" + colophon),
    ):
        if marker not in page:
            raise RuntimeError(f"reader.html lacks {marker}")
        page = page.replace(marker, value)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(PAPER / "paper.pdf", OUT / "paper.pdf")
    (OUT / "figures").mkdir(exist_ok=True)
    for old in (OUT / "figures").glob("*"):
        old.unlink()
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page))):
        shutil.copyfile(PAPER / "figures" / name, OUT / "figures" / name)
    return OUT / "index.html"


if __name__ == "__main__":
    out = build()
    print(f"wrote {out.relative_to(ROOT)}")
    sys.exit(0)
