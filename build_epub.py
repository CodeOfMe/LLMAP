#!/usr/bin/env python3
"""Build LLMAP_English.epub directly from markdown.

Fixes pandoc-incompatible markdown on the fly, then one pandoc call.

Usage:
    python build_epub.py
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CHAPTERS = [
    "chapter-00-preface.md",
    "chapter-01-prompt-engineering.md",
    "chapter-02-llm-api.md",
    "chapter-03-tokenization.md",
    "chapter-04-data.md",
    "chapter-05-transformer.md",
    "chapter-06-self-attention.md",
    "chapter-07-multihead-causal.md",
    "chapter-08-kv-cache.md",
    "chapter-09-positional-encoding.md",
    "chapter-10-norm-activation.md",
    "chapter-11-agent-loop.md",
    "chapter-12-reasoning-planning.md",
    "chapter-13-mcp.md",
    "chapter-14-skills.md",
    "chapter-15-memory.md",
    "chapter-16-subagent.md",
    "chapter-17-multi-agent.md",
    "chapter-18-context-engineering.md",
    "chapter-19-harness-engineering.md",
    "chapter-20-future.md",
    "references.md",
]


def fix_markdown(text):
    # Bare ``` → ~~~ (pandoc needs language tag or tildes)
    lines = text.split("\n")
    fixed = []
    fence = None
    for line in lines:
        s = line.strip()
        if fence is None:
            if re.match(r"^```\s*$", s):
                fixed.append("~~~")
                fence = "~~~"
            elif re.match(r"^```[a-zA-Z]", s):
                if "linenum" in s or "title=" in s:
                    lang = re.sub(r"^```(\w+).*", r"\1", s)
                    fixed.append(f"```{lang}")
                else:
                    fixed.append(line)
                fence = "```"
            else:
                fixed.append(line)
        else:
            if re.match(r"^```\s*$", s):
                fixed.append("~~~" if fence == "~~~" else line)
                fence = None
            elif re.match(r"^~~~\s*$", s):
                fixed.append(line)
                fence = None
            else:
                fixed.append(line)
    return "\n".join(fixed)


def build():
    merged = []
    for fname in CHAPTERS:
        p = ROOT / fname
        if p.exists():
            merged.append(fix_markdown(p.read_text("utf-8")))
    all_md = "\n\n".join(merged)

    tmp = ROOT / "_epub_tmp.md"
    tmp.write_text(all_md, "utf-8")

    out = ROOT / "LLMAP_English.epub"
    cmd = [
        "pandoc", str(tmp),
        "-f", "markdown+smart+yaml_metadata_block+footnotes",
        "-t", "epub3",
        "-o", str(out),
        "--wrap=none",
        f"--resource-path={ROOT}",
        "--toc",
        "--toc-depth=2",
        "--epub-chapter-level=1",
        "--epub-title-page",
        f"--epub-cover-image={ROOT / 'figures/png/fig05-transformer-architecture.png'}",
        f"--css={ROOT / 'epub.css'}",
        f"--epub-embed-font={Path.home()}/Library/Fonts/NotoSans-Regular.ttf",
        f"--epub-embed-font={Path.home()}/Library/Fonts/NotoSans-Bold.ttf",
        f"--epub-embed-font={Path.home()}/Library/Fonts/JetBrainsMono-Regular.ttf",
        f"--epub-embed-font={Path.home()}/Library/Fonts/JetBrainsMono-Bold.ttf",
        "--metadata", "title=LLMAP — Large Language Models: An Agent Perspective",
        "--metadata", "author=LLMAP",
        "--metadata", "lang=en",
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError("pandoc failed")

    tmp.unlink()
    mb = out.stat().st_size / 1024 / 1024
    print(f"{out.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    build()