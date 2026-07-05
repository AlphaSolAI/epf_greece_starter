# pdf_maker_mcp — τοπικός MCP server για PDF από markdown/text

Δοκιμαστικό deliverable του `/mcp-builder` (2026-07-05). Stdio MCP server σε Python
(FastMCP), τρέχει με **system Python 3.11** (πακέτα `mcp`, `fpdf2` σε user-site) —
**όχι conda**, δεν μπλοκάρει την ουρά training. Πλήρης υποστήριξη ελληνικών
(Unicode TTF: Arial + Consolas των Windows, με αυτόματο font subsetting).

## Tools

| Tool | Τι κάνει | Annotations |
|---|---|---|
| `pdf_from_markdown` | 1-20 md/txt αρχεία → ένα A4 PDF (εξώφυλλο, κεφάλαιο ανά αρχείο, headings/code/tables/bullets, page numbers). Επιστρέφει JSON `{output_path, pages, bytes, files}`. Προστασία overwrite (`overwrite=true` ρητά). | write, μη-καταστροφικό, idempotent |
| `pdf_info` | Read-only: `{path, bytes, pages_estimate}` υπάρχοντος PDF | read-only |

Σφάλματα: πάντα JSON με πεδίο `error` + συγκεκριμένη οδηγία (ποιο path λείπει, τι να αλλάξεις).

## Εγγραφή στο Claude Code

```bash
claude mcp add pdf-maker -- "C:\Program Files\Python311\python.exe" "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\scripts\mcp_pdf_maker\server.py"
```

(ή σε `.mcp.json` του project για να ισχύει για όλους στο repo)

## Smoke test (χωρίς εγγραφή — πραγματικό stdio πρωτόκολλο)

```bash
"/c/Program Files/Python311/python.exe" -X utf8 scripts/mcp_pdf_maker/test_client.py
# → TOOLS: ['pdf_from_markdown', 'pdf_info'] ... SMOKE: PASS
```

Πρώτο παραχθέν PDF: `reports/overnight_20260705_bundle.pdf` (5 σελ., report + 2 SKILL.md).

## Σημειώσεις υλοποίησης

- Emoji δεν υπάρχουν στην Arial → χαρτογραφούνται σε ASCII (`✅→[OK]`, `⚠️→[!]` κ.λπ.),
  ό,τι άλλο >U+2FFF αφαιρείται. Ελληνικά/σύμβολα/€ περνάνε κανονικά.
- Όρια ασφαλείας: ≤20 αρχεία, ≤2MB το καθένα, output μόνο `.pdf`, ο φάκελος πρέπει να υπάρχει.
- Επόμενα (αν χρειαστεί ποτέ): Phase 4 evaluations του mcp-builder (10 QA pairs),
  HTML→PDF, TOC με links.
