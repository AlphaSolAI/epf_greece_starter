# -*- coding: utf-8 -*-
"""
pdf_maker_mcp — μικρός τοπικός MCP server (stdio) που φτιάχνει PDF από markdown/text αρχεία.

Δημιουργήθηκε 2026-07-05 με το /mcp-builder ως δοκιμή του workflow σε αυτό το repo.
Τρέχει με SYSTEM python (3.11 + user-site: mcp, fpdf2) — ΟΧΙ conda (δεν μπλοκάρει
την ουρά training). Υποστηρίζει πλήρως ελληνικά (Unicode TTF: Arial/Consolas των Windows).

Tools:
  - pdf_from_markdown : λίστα md/txt αρχείων → ένα PDF (κεφάλαιο ανά αρχείο)
  - pdf_info          : read-only πληροφορίες υπάρχοντος PDF

Εγγραφή στο Claude Code (παράδειγμα):
  claude mcp add pdf-maker -- "C:\\Program Files\\Python311\\python.exe" \
      "<repo>\\scripts\\mcp_pdf_maker\\server.py"
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from fpdf import FPDF

mcp = FastMCP("pdf_maker_mcp")

_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
_BODY, _BOLD, _MONO = _FONTS / "arial.ttf", _FONTS / "arialbd.ttf", _FONTS / "consola.ttf"

# fpdf2 + Arial δεν έχουν emoji glyphs — χαρτογράφηση σε ASCII πριν το render.
_EMOJI_MAP = {
    "✅": "[OK]", "⚠️": "[!]", "⚠": "[!]", "⏳": "[...]", "🟢": "[GO]",
    "🔵": "[i]", "❌": "[X]", "💡": "[idea]", "⛔": "[STOP]", "🚀": "[>>]",
    "📊": "[chart]", "💾": "[save]", "✍️": "[γράψε]", "🔍": "[scan]", "☀️": "[sun]",
    "▶️": "[run]", "✓": "[ok]", "→": "->",
}


def _sanitize(line: str) -> str:
    for k, v in _EMOJI_MAP.items():
        line = line.replace(k, v)
    # κράτα BMP έως 0x2FFF (λατινικά/ελληνικά/σύμβολα/€) — πέτα emoji & variation selectors
    return "".join(c for c in line if ord(c) < 0x3000 and c != "️")


class _Doc(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.doc_title = title
        self.add_font("Body", "", str(_BODY))
        self.add_font("Body", "B", str(_BOLD))
        self.add_font("Mono", "", str(_MONO))
        self.set_auto_page_break(True, margin=18)
        self.set_margins(16, 14, 16)

    def footer(self):
        self.set_y(-12)
        self.set_font("Body", "", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"{self.doc_title} — σελ. {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0)

    def _w(self):
        return self.w - self.l_margin - self.r_margin

    def render_markdown(self, text: str):
        in_code = False
        for raw in text.splitlines():
            line = _sanitize(raw.rstrip())
            if line.strip().startswith("```"):
                in_code = not in_code
                self.ln(1)
                continue
            if in_code or line.lstrip().startswith("|"):
                self.set_font("Mono", "", 7.3)
                self.multi_cell(self._w(), 3.4, line if line else " ")
                continue
            if not line.strip():
                self.ln(2)
                continue
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                level = len(m.group(1))
                size = {1: 15, 2: 12.5, 3: 11}.get(level, 10.5)
                self.ln(2 if level > 1 else 3)
                self.set_font("Body", "B", size)
                self.multi_cell(self._w(), size * 0.5, m.group(2))
                self.ln(1)
                continue
            if re.match(r"^\s*[-*]\s+", line):
                line = re.sub(r"^(\s*)[-*]\s+", r"\1• ", line)
            line = line.replace("**", "")  # χωρίς inline bold — απλό rendering
            self.set_font("Body", "", 9.5)
            self.multi_cell(self._w(), 4.6, line)


class PdfFromMarkdownInput(BaseModel):
    """Είσοδος για δημιουργία PDF από αρχεία markdown/text."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    files: List[str] = Field(
        ..., min_length=1, max_length=20,
        description="Λίστα από md/txt αρχεία (απόλυτα ή σχετικά ως προς το cwd του server), "
                    "π.χ. ['reports/overnight_20260705_report.md', 'last.md']. Κάθε αρχείο "
                    "γίνεται κεφάλαιο με heading το όνομά του.")
    output_path: str = Field(
        ..., min_length=1,
        description="Πού θα γραφτεί το PDF, π.χ. 'reports/bundle.pdf'. Ο φάκελος πρέπει να υπάρχει.")
    title: Optional[str] = Field(
        default=None, description="Τίτλος εξωφύλλου/footer. Default: το όνομα του output αρχείου.")
    overwrite: bool = Field(
        default=False, description="Αν false και το output υπάρχει ήδη → σφάλμα (προστασία).")


@mcp.tool(
    name="pdf_from_markdown",
    annotations={
        "title": "Δημιουργία PDF από markdown/text αρχεία",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def pdf_from_markdown(params: PdfFromMarkdownInput) -> str:
    """Ενώνει 1-20 markdown/text αρχεία σε ένα PDF (A4, Unicode/ελληνικά, κεφάλαιο ανά
    αρχείο, page numbers). Επιστρέφει JSON: output_path, pages, bytes, files.

    Σφάλματα επιστρέφονται ως JSON με πεδίο 'error' και συγκεκριμένη οδηγία διόρθωσης.
    """
    missing = [f for f in params.files if not Path(f).is_file()]
    if missing:
        return json.dumps({"error": f"Δεν βρέθηκαν αρχεία: {missing}. Δώσε υπαρκτά paths "
                                    f"(cwd του server: {os.getcwd()})."}, ensure_ascii=False)
    too_big = [f for f in params.files if Path(f).stat().st_size > 2_000_000]
    if too_big:
        return json.dumps({"error": f"Πολύ μεγάλα (>2MB): {too_big}. Σπάσε τα ή δώσε μικρότερα."},
                          ensure_ascii=False)
    out = Path(params.output_path)
    if out.suffix.lower() != ".pdf":
        return json.dumps({"error": f"Το output_path πρέπει να τελειώνει σε .pdf (δόθηκε: {out})."},
                          ensure_ascii=False)
    if out.exists() and not params.overwrite:
        return json.dumps({"error": f"Υπάρχει ήδη: {out}. Ξαναδοκίμασε με overwrite=true "
                                    f"ή άλλαξε output_path."}, ensure_ascii=False)
    if not out.parent.exists():
        return json.dumps({"error": f"Ο φάκελος {out.parent} δεν υπάρχει — φτιάξ' τον πρώτα."},
                          ensure_ascii=False)

    title = params.title or out.stem
    pdf = _Doc(title)
    pdf.add_page()
    pdf.set_font("Body", "B", 20)
    pdf.ln(40)
    pdf.multi_cell(pdf._w(), 10, _sanitize(title), align="C")
    pdf.set_font("Body", "", 10)
    pdf.multi_cell(pdf._w(), 6, f"{len(params.files)} αρχεία", align="C")
    for f in params.files:
        pdf.add_page()
        pdf.set_font("Mono", "", 9)
        pdf.set_text_color(90)
        pdf.multi_cell(pdf._w(), 5, f)
        pdf.set_text_color(0)
        pdf.ln(2)
        pdf.render_markdown(Path(f).read_text(encoding="utf-8", errors="replace"))
    pdf.output(str(out))
    return json.dumps({"output_path": str(out), "pages": pdf.page_no(),
                       "bytes": out.stat().st_size, "files": params.files}, ensure_ascii=False)


class PdfInfoInput(BaseModel):
    """Είσοδος για ανάγνωση πληροφοριών PDF."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    path: str = Field(..., min_length=1, description="Path υπάρχοντος PDF αρχείου.")


@mcp.tool(
    name="pdf_info",
    annotations={
        "title": "Πληροφορίες PDF αρχείου",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def pdf_info(params: PdfInfoInput) -> str:
    """Read-only: επιστρέφει JSON με ύπαρξη, μέγεθος (bytes) και εκτίμηση σελίδων
    (μέτρηση /Type /Page αντικειμένων) ενός PDF."""
    p = Path(params.path)
    if not p.is_file():
        return json.dumps({"error": f"Δεν υπάρχει αρχείο: {p} (cwd: {os.getcwd()})."},
                          ensure_ascii=False)
    data = p.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    return json.dumps({"path": str(p), "bytes": len(data), "pages_estimate": pages},
                      ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()  # stdio transport
