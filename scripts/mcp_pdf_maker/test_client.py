# -*- coding: utf-8 -*-
"""Smoke test του pdf_maker_mcp μέσα από το ΠΡΑΓΜΑΤΙΚΟ MCP πρωτόκολλο (stdio):
initialize → list_tools → call pdf_from_markdown → call pdf_info."""
import asyncio
import io
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SERVER = Path(__file__).with_name("server.py")
PY = sys.executable

FILES = [
    "reports/overnight_20260705_report.md",
    ".claude/skills/synthesize-ablation/SKILL.md",
    ".claude/skills/ingest-audit/SKILL.md",
]
OUT = "reports/overnight_20260705_bundle.pdf"


async def main() -> int:
    params = StdioServerParameters(command=PY, args=[str(SERVER)])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            names = [t.name for t in tools.tools]
            print("TOOLS:", names)

            res = await s.call_tool("pdf_from_markdown", {
                "params": {"files": FILES, "output_path": OUT,
                           "title": "EPF GR — Overnight 2026-07-05 bundle (δοκιμή MCP)",
                           "overwrite": True}})
            payload = json.loads(res.content[0].text)
            print("CREATE:", json.dumps(payload, ensure_ascii=False))
            if "error" in payload:
                return 1

            res2 = await s.call_tool("pdf_info", {"params": {"path": OUT}})
            info = json.loads(res2.content[0].text)
            print("INFO:  ", json.dumps(info, ensure_ascii=False))
            ok = ("error" not in info and info["bytes"] > 10_000
                  and info["pages_estimate"] == payload["pages"])
            print("SMOKE:", "PASS" if ok else "FAIL")
            return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
