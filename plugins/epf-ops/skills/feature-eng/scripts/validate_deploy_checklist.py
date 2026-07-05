"""Validate a filled feature deploy checklist (feature-eng stage 6).

Usage (system python, no conda needed):
    python .claude/skills/feature-eng/scripts/validate_deploy_checklist.py docs/features/<name>/deploy.md

Checks:
  1. All six required sections are present.
  2. No unfilled placeholders (<...>) or TBD/TODO markers remain.
  3. Suspended (pre-TZFIX / pre-AEL / leaked) headline numbers do not appear
     unless the same line explicitly marks them as suspended.
Exit code 0 = PASS, 1 = FAIL.
"""
import io
import re
import sys

REQUIRED_SECTIONS = [
    "## 1. LEAKAGE-FREE PROOF",
    "## 2. FEATURE IMPORTANCE",
    "## 3. EXPECTED vs ACTUAL",
    "## 4. KPI & VERDICT",
    "## 5. DANGERS & ROLLBACK",
    "## 6. TRACE",
]

SUSPENDED_NUMBERS = ["15.17", "16.10", "19.17", "14.43", "15.02"]
SUSPENSION_MARKERS = [
    "ΑΝΑΣΤΟΛΗ", "suspended", "SUSPENDED", "προ-AEL", "προ-TZFIX",
    "pre-AEL", "pre-TZFIX", "leaked", "ΑΚΥΡ",
]

PLACEHOLDER_RE = re.compile(r"<[^>]{0,60}>")
# lines that legitimately contain angle brackets (paths in usage examples etc.)
PLACEHOLDER_ALLOW = re.compile(r"^\s*(>|```|#)")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_deploy_checklist.py <deploy.md>")
        return 1
    path = sys.argv[1]
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print("FAIL: cannot read %s (%s)" % (path, e))
        return 1

    ok = True

    for section in REQUIRED_SECTIONS:
        if section not in text:
            print("FAIL: missing section: %s" % section)
            ok = False

    for i, line in enumerate(text.splitlines(), 1):
        if PLACEHOLDER_ALLOW.match(line):
            continue
        m = PLACEHOLDER_RE.search(line)
        if m:
            print("FAIL: unfilled placeholder line %d: %s" % (i, m.group(0)))
            ok = False
        if "TBD" in line or "TODO" in line:
            print("FAIL: TBD/TODO left on line %d" % i)
            ok = False
        for num in SUSPENDED_NUMBERS:
            if num in line and not any(mk in line for mk in SUSPENSION_MARKERS):
                print(
                    "FAIL: suspended number %s on line %d without suspension "
                    "marker (pre-TZFIX/pre-AEL results must never appear as "
                    "current)" % (num, i)
                )
                ok = False

    print("PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
