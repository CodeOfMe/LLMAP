#!/usr/bin/env python3
"""Build KDP-ready EPUB for LLMAP.

Usage:
    python build_epub.py

Pipeline:
  1. Merge markdown chapters + fix code fences
  2. Clean references.md blockquote header
  3. Pandoc converts to EPUB3 with embedded fonts, CSS, native TOC
  4. Validate structure
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CHAPTER_ORDER = [
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
]

FONTS = [
    str(Path.home() / "Library/Fonts/NotoSans-Regular.ttf"),
    str(Path.home() / "Library/Fonts/NotoSans-Bold.ttf"),
    str(Path.home() / "Library/Fonts/JetBrainsMono-Regular.ttf"),
    str(Path.home() / "Library/Fonts/JetBrainsMono-Bold.ttf"),
]


def merge_markdown():
    parts = [(ROOT / "title-page.md").read_text("utf-8").strip()]
    for fname in CHAPTER_ORDER:
        fpath = ROOT / fname
        if fpath.exists():
            parts.append(fpath.read_text("utf-8").strip())
        else:
            print(f"  Warning: {fpath} not found")

    ref_path = ROOT / "references.md"
    if ref_path.exists():
        raw = ref_path.read_text("utf-8")
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            if line.startswith("> Core papers"):
                continue
            cleaned.append(line)
        parts.append("\n".join(cleaned).strip())

    merged = "\n\n".join(parts)
    lines = merged.split("\n")
    fixed = []
    fence_stack = []
    for line in lines:
        stripped = line.strip()
        if not fence_stack:
            if re.match(r"^```\s*$", stripped):
                fixed.append("~~~")
                fence_stack.append("~~~")
            elif re.match(r"^```[a-zA-Z]", stripped):
                if "linenum" in stripped or "title=" in stripped:
                    lang = re.sub(r"^```(\w+).*", r"\1", stripped)
                    fixed.append(f"```{lang}")
                else:
                    fixed.append(line)
                fence_stack.append("```")
            else:
                fixed.append(line)
        else:
            if re.match(r"^```\s*$", stripped):
                opening = fence_stack.pop()
                fixed.append("~~~" if opening == "~~~" else line)
            elif re.match(r"^~~~\s*$", stripped):
                fixed.append(line)
                if fence_stack:
                    fence_stack.pop()
            else:
                fixed.append(line)

    tmp = ROOT / "_merged_epub.md"
    tmp.write_text("\n".join(fixed), "utf-8")
    return tmp


def build_epub():
    out_path = ROOT / "LLMAP_English.epub"
    css_path = ROOT / "epub.css"

    print("Merging markdown...")
    tmp_md = merge_markdown()

    cmd = [
        "pandoc",
        str(tmp_md),
        "-f", "markdown+smart+yaml_metadata_block+footnotes",
        "-t", "epub3",
        "-o", str(out_path),
        "--wrap=none",
        "--columns=10000",
        f"--resource-path={ROOT}",
        f"--css={css_path}",
        "--toc",
        "--toc-depth=2",
        "--epub-chapter-level=1",
        "--epub-title-page=true",
    ]

    for font in FONTS:
        if Path(font).exists():
            cmd.append(f"--epub-embed-font={font}")

    print("Running pandoc...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Pandoc error: {r.stderr}")
        raise RuntimeError(f"Pandoc failed: {r.stderr}")

    mb = out_path.stat().st_size / 1024 / 1024
    print(f"Built: {out_path} ({mb:.1f} MB)")

    tmp_md.unlink(missing_ok=True)

    # Validate
    validate_epub(out_path)


def validate_epub(path):
    """Basic structural validation without epubcheck."""
    import zipfile

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()

    has_mimetype = "mimetype" in names
    has_container = any("META-INF/container.xml" in n for n in names)
    has_opf = any(n.endswith(".opf") for n in names)
    has_toc = any("toc" in n.lower() for n in names)
    html_count = sum(1 for n in names if n.endswith(".xhtml") or n.endswith(".html"))
    img_count = sum(1 for n in names if n.endswith(".png") or n.endswith(".svg"))
    css_count = sum(1 for n in names if n.endswith(".css"))
    font_count = sum(1 for n in names if n.endswith(".ttf") or n.endswith(".otf"))

    print(f"\nValidation:")
    print(f"  mimetype:    {has_mimetype}")
    print(f"  container:   {has_container}")
    print(f"  OPF:         {has_opf}")
    print(f"  TOC:         {has_toc}")
    print(f"  HTML pages:  {html_count}")
    print(f"  Images:      {img_count}")
    print(f"  CSS files:   {css_count}")
    print(f"  Fonts:       {font_count}")
    print(f"  Total files: {len(names)}")

    if not has_mimetype or not has_container or not has_opf:
        print("WARNING: EPUB structure incomplete!")
    else:
        print("EPUB structure OK for KDP upload.")


if __name__ == "__main__":
    build_epub()