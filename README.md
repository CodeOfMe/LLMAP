# LLMAP — Large Language Models: An Agent Perspective

> A complete textbook from prompts to agents — one concept per chapter, progressive in depth, all claims sourced, all code runnable.

## Why This Book

Most resources teach you either how Transformers work or how to build Agents. This textbook connects the full chain: from the first prompt you write, through the inference happening inside the model, to an Agent loop that takes action — and the engineering that keeps it reliable in production.

## Keywords

LLM · Agent · Prompt Engineering · Attention · Context Engineering · Tool Use · Multi-Agent

## Reading Path

### Part I: Talking to LLMs

You don't need to know how an engine works to drive a car. This part teaches you how to drive well.

| Ch | Title | You Will Learn |
|---|-------|---------------|
| [1](chapter-01-prompt-engineering.md) | Prompt Engineering | Zero-shot, few-shot, CoT; common writing pitfalls |
| [2](chapter-02-llm-api.md) | LLM API & Local Deployment | API mechanics; token billing; streaming; function calling; vLLM/SGLang |

### Part II: Inside the LLM

Pop the hood. These chapters cover every step from input to output.

| Ch | Title | You Will Learn |
|---|-------|---------------|
| [3](chapter-03-tokenization.md) | Tokens & Tokenization | BPE algorithm; why vocabulary size matters |
| [4](chapter-04-data.md) | Training Data | Cleaning pipelines; Chinchilla scaling; quality over quantity |
| [5](chapter-05-transformer.md) | Transformer Architecture | Why Decoder-Only won; forward pass data flow |
| [6](chapter-06-self-attention.md) | Self-Attention | QKV derivation; the full math |
| [7](chapter-07-multihead-causal.md) | Multi-Head & Causal Masks | Multiple heads; causal masking; MQA and GQA |
| [8](chapter-08-kv-cache.md) | KV Cache | PagedAttention; continuous batching |
| [9](chapter-09-positional-encoding.md) | Positional Encoding | Sinusoidal, learned, RoPE |
| [10](chapter-10-norm-activation.md) | Normalization & Activation | RidgeRMSNorm; ReLU→GELU→SwiGLU; residual connections |

### Part III: Making LLMs Act

Let the car run errands on its own.

| Ch | Title | You Will Learn |
|---|-------|---------------|
| [11](chapter-11-agent-loop.md) | Agent Loop & Tool Use | ReAct loop; function calling; security |
| [12](chapter-12-reasoning-planning.md) | Reasoning & Planning | CoT, ToT, slow thinking |
| [13](chapter-13-mcp.md) | MCP Protocol | Standardized tool protocol; server architecture |
| [14](chapter-14-skills.md) | Skill Systems | Discoverable skills; registration; solidification |
| [15](chapter-15-memory.md) | Memory Mechanisms | Short-term & long-term; MemGPT; vector memory |
| [16](chapter-16-subagent.md) | Sub-Agents | Delegation & specialization |
| [17](chapter-17-multi-agent.md) | Multi-Agent Collaboration | AutoGen, CrewAI, LangGraph; conflict resolution |

### Part IV: Making It Run Reliably

Install a steering wheel, brakes, and a dashboard.

| Ch | Title | You Will Learn |
|---|-------|---------------|
| [18](chapter-18-context-engineering.md) | Context Engineering | Window management; caching; compression; Lost in the Middle |
| [19](chapter-19-harness-engineering.md) | Harness Engineering | Guardrails; evaluation; observability; safety |
| [20](chapter-20-future.md) | Frontiers & Outlook | Inference-time compute; multimodal agents; MCP ecosystem |

## What Makes This Book Different

- **One concept per chapter.** No digressions. When covering attention, we cover attention.
- **Every claim sourced.** Not "studies show" — [Vaswani et al., 2017], [Hoffmann et al., 2022], with arXiv links.
- **All code runs.** Complete Python programs, not pseudocode. Minimal dependencies: PyTorch, tiktoken, transformers.

## Repository Structure

```
├── chapter-XX-*.md     # 21 chapters
├── codes/               # Runnable Python examples
├── figures/
│   ├── svg/             # Source figures (grayscale)
│   ├── pdf/             # For print
│   └── png/             # Preview (300 DPI)
├── references.md        # Complete reference list
├── references/
│   ├── citations-master.md
│   └── model-specs.md
├── LLMAP_English.docx   # KDP-ready Word document
├── LLMAP_English.epub   # KDP-ready EPUB
├── build_docx.py        # Build docx (pandoc + reference-doc)
├── build_epub.py        # Build epub (pandoc + CSS)
├── build_template.py    # Generate reference-doc templates
└── epub.css             # EPUB stylesheet
```

## Building

```bash
# Build docx (requires: pandoc, python-docx, Noto Sans + JetBrains Mono fonts)
python build_docx.py

# Build epub
python build_epub.py
```

## Citation Convention

All claims cite sources in `[Author, Year]` format. Full references in [references.md](references.md).

## License

All text and figures are newly written and created.