"""Turn a EUR-Lex HTML page into a list of provisions.

EUR-Lex publishes two HTML layouts: the Official Journal layout (classes such
as ``oj-ti-art``) and the consolidated-text layout (``title-article-norm``).
Both mark provisions with the same ELI ids (``art_61``, ``rct_12``,
``anx_I``), so the parser keys on those ids and ignores the class names
wherever it can. That is what lets the same code read the original act, any
later consolidated version, and other directives.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

try:  # pages from the Publications Office repository start with an XML declaration
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

ARTICLE_ID = re.compile(r"^art_(\d+[a-z]*)$")
RECITAL_ID = re.compile(r"^rct_(\d+[a-z]*)$")
ANNEX_ID = re.compile(r"^anx_([^.]+)$")
DIVISION_ID = re.compile(r"(?:^|\.)(prt|tis|cpt|sct|sbs)_[^.]+$")

BLOCK_TAGS = {
    "p", "div", "table", "tbody", "thead", "tr", "td", "th", "li", "ul", "ol",
    "dl", "dt", "dd", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
}
# Markers EUR-Lex adds to consolidated texts ("▼B", "►M1 ... ◄").
CONSOLIDATION_MARKS = re.compile(r"[▼►][A-Z]\d*\s?|◄")
PARAGRAPH_LABEL = re.compile(r"^(\d+[a-z]?)\.\s")
POINT_LABEL = re.compile(r"^\((\d+[a-z]?)\)\s")
SPLIT_THRESHOLD = 1800


@dataclass
class Block:
    """One paragraph-sized unit of a provision."""

    text: str
    para: str = ""  # paragraph number, e.g. "2"; empty when unnumbered
    point: str = ""  # numbered point, e.g. definition "(5)" -> "5"


@dataclass
class Provision:
    id: str  # ELI id, also the anchor on the EUR-Lex page
    kind: str  # "article" | "recital" | "annex"
    number: str
    label: str  # as printed in this language, e.g. "Article 61"
    title: str = ""
    path: list[str] = field(default_factory=list)  # Part / Title / Chapter
    blocks: list[Block] = field(default_factory=list)
    source_celex: str = ""

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Provision":
        blocks = [Block(**b) for b in data.get("blocks", [])]
        return cls(**{**data, "blocks": blocks})


def normalise(text: str) -> str:
    """Collapse all whitespace (including non-breaking spaces) to single spaces."""
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def text_of(node: Tag) -> str:
    """Text of a node with inline elements joined and block elements separated."""
    parts: list[str] = []

    def walk(n) -> None:
        if isinstance(n, NavigableString):
            parts.append(str(n))
            return
        if not isinstance(n, Tag):
            return
        is_block = n.name in BLOCK_TAGS
        if is_block:
            parts.append(" ")
        for child in n.children:
            walk(child)
        if is_block:
            parts.append(" ")

    walk(node)
    return normalise(CONSOLIDATION_MARKS.sub("", "".join(parts)))


def _classes(tag: Tag) -> str:
    return " ".join(tag.get("class") or [])


def _is_heading(tag: Tag) -> bool:
    """Article / annex / division heading lines, in either layout."""
    cls = _classes(tag)
    return bool(
        re.search(r"\b(oj-)?(ti-art|sti-art|doc-ti|ti-section-\d)\b", cls)
        or re.search(r"\b(s?title-article|title-annex|title-division)", cls)
        or "eli-title" in cls
    )


def _clean(soup: BeautifulSoup) -> None:
    """Drop footnote call-outs and consolidation marker lines."""
    for anchor in soup.find_all("a", id=re.compile(r"^ntc")):
        prev = anchor.previous_sibling
        if isinstance(prev, NavigableString):
            prev.replace_with(str(prev).rstrip("  \t\n"))
        anchor.decompose()
    for marker in soup.find_all(class_=re.compile(r"\b(modref|hd-modifiers)\b")):
        marker.decompose()
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()


def _child_tags(tag: Tag) -> list[Tag]:
    return [c for c in tag.children if isinstance(c, Tag)]


def _body_blocks(container: Tag, skip_headings: bool = True) -> list[str]:
    """Split a provision into paragraph-sized texts.

    Each direct child is one block. A very long child that is only a wrapper
    around several block children is opened up so chunks stay readable.
    """
    out: list[str] = []
    for child in _child_tags(container):
        if skip_headings and _is_heading(child):
            continue
        text = text_of(child)
        if not text:
            continue
        inner = [c for c in _child_tags(child) if text_of(c)]
        if child.name == "div" and len(text) > SPLIT_THRESHOLD and len(inner) > 1:
            out.extend(_body_blocks(child, skip_headings=False))
        else:
            out.append(text)
    return _join_bare_labels(out)


BARE_LABEL = re.compile(r"^(\d+[a-z]?\.|\([0-9a-z]{1,4}\))$")


def _join_bare_labels(texts: list[str]) -> list[str]:
    """Reattach a label that ended up alone ("2.") to the text it introduces."""
    out: list[str] = []
    pending = ""
    for text in texts:
        if BARE_LABEL.match(text):
            pending = f"{pending} {text}".strip()
            continue
        out.append(f"{pending} {text}".strip())
        pending = ""
    if pending:
        out.append(pending)
    return out


def _label_blocks(texts: list[str]) -> list[Block]:
    blocks: list[Block] = []
    para = ""
    for text in texts:
        point = ""
        m = PARAGRAPH_LABEL.match(text)
        if m:
            para = m.group(1)
        else:
            p = POINT_LABEL.match(text)
            if p:
                point = p.group(1)
        blocks.append(Block(text=text, para=para, point=point))
    return blocks


def _division_path(tag: Tag) -> list[str]:
    """Headings of the enclosing Part / Title / Chapter / Section, outermost first."""
    path: list[str] = []
    for parent in tag.parents:
        if not isinstance(parent, Tag) or not DIVISION_ID.search(parent.get("id") or ""):
            continue
        heading, title = "", ""
        for child in _child_tags(parent):
            if child.name == "p" and not heading:
                heading = text_of(child)
            elif "eli-title" in _classes(child) and not title:
                title = text_of(child)
            if heading and title:
                break
        entry = " — ".join(x for x in (heading, title) if x)
        if entry:
            path.append(entry)
    return list(reversed(path))


def _first_heading(tag: Tag, pattern: str) -> str:
    for child in _child_tags(tag):
        if re.search(pattern, _classes(child)):
            return text_of(child)
    return ""


# Greek and Cyrillic letters that some language versions type in place of
# the Latin ones in Roman numerals ("ΠΑΡΑΡΤΗΜΑ ΧΙ", "anx_ХI").
_LATIN_NUMERALS = str.maketrans({"Ι": "I", "Χ": "X", "Х": "X", "І": "I", "С": "C", "Ϲ": "C"})


def _roman(text: str) -> str:
    return text.translate(_LATIN_NUMERALS)


def _tidy_ids(soup: BeautifulSoup) -> None:
    """Some language versions end ids with a (non-breaking) space ("anx_I ")
    or write annex numerals with look-alike letters ("anx_ХI")."""
    for tag in soup.find_all(id=True):
        tag["id"] = tag["id"].strip()
        if tag["id"].startswith("anx_"):
            tag["id"] = _roman(tag["id"])


def _wrap_bare_annexes(soup: BeautifulSoup) -> None:
    """Some language versions (e.g. EL, RO) leave annexes unwrapped: each is a
    "title-annex-1" heading followed by its content as siblings. Wrap each in
    a div with the id the other versions use (anx_I, anx_II ...)."""
    if soup.find(id=ANNEX_ID):
        return
    for heading in soup.find_all(class_="title-annex-1"):
        numerals = [t.strip(".") for t in _roman(text_of(heading)).split()
                    if re.fullmatch(r"[IVXLC]+\.?", t)]
        if not numerals:
            continue
        wrapper = soup.new_tag("div", id=f"anx_{numerals[0]}")
        heading.insert_before(wrapper)
        node = heading
        while node is not None:
            following = node.next_sibling
            wrapper.append(node.extract())
            node = following
            if isinstance(node, Tag) and (
                    "title-annex-1" in (node.get("class") or []) or node.get("id")):
                break


def parse_html(html: str, source_celex: str = "") -> list[Provision]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # lxml not installed
        soup = BeautifulSoup(html, "html.parser")
    _tidy_ids(soup)
    _wrap_bare_annexes(soup)
    _clean(soup)
    provisions: list[Provision] = []

    for tag in soup.find_all(id=RECITAL_ID):
        number = RECITAL_ID.match(tag["id"]).group(1)
        text = re.sub(r"^\(\s*%s\s*\)\s*" % re.escape(number), "", text_of(tag))
        if text:
            provisions.append(Provision(
                id=tag["id"], kind="recital", number=number, label=f"({number})",
                blocks=[Block(text=text)], source_celex=source_celex,
            ))

    for tag in soup.find_all(id=ARTICLE_ID):
        number = ARTICLE_ID.match(tag["id"]).group(1)
        label = _first_heading(tag, r"(ti-art|title-article)") or f"Article {number}"
        title = ""
        title_tag = tag.find(class_="eli-title")
        if title_tag is not None:
            title = text_of(title_tag)
        provisions.append(Provision(
            id=tag["id"], kind="article", number=number, label=label, title=title,
            path=_division_path(tag), blocks=_label_blocks(_body_blocks(tag)),
            source_celex=source_celex,
        ))

    for tag in soup.find_all(id=ANNEX_ID):
        number = ANNEX_ID.match(tag["id"]).group(1)
        headings = [text_of(c) for c in _child_tags(tag)
                    if re.search(r"(doc-ti|title-annex)", _classes(c)) and text_of(c)]
        label = headings[0] if headings else f"Annex {number}"
        title = " ".join(headings[1:])
        provisions.append(Provision(
            id=tag["id"], kind="annex", number=number, label=label, title=title,
            blocks=_label_blocks(_body_blocks(tag)), source_celex=source_celex,
        ))

    return provisions
