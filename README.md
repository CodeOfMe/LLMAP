# LLMAP — Large Language Models: Agent Perspective

> A complete textbook from prompts to agents — one concept per chapter, progressive in depth, all claims sourced, all visualizations traceable.

## Why This Book

The learning path goes like this: first learn to talk to LLMs (prompt engineering), then understand what happens behind the conversation (APIs and inference mechanisms), then dive inside the model to see how it computes (data, architecture, attention, caching), then come back out and learn how to make LLMs act (agents, tools, protocols), and finally learn to make it all run reliably (context engineering, harness engineering).

Each chapter covers exactly one concept. Conclusions from earlier chapters are referenced and deepened in later chapters.

## Reading Path

### Part I: Talking to LLMs

You don't need to know how an engine works to drive a car. This part teaches you how to drive well.

| Ch | Title | You Will Learn | Prerequisites |
|---|-------|---------------|--------------|
| [1](CN/chapter-01-prompt-engineering.md) | Prompt Engineering | How to write prompts that make LLMs behave; zero-shot, few-shot, CoT; common writing pitfalls | None |
| [2](CN/chapter-02-llm-api.md) | LLM API & Local Deployment | Ollama local deployment; how APIs work; token billing; streaming; function calling; vLLM/SGLang | Ch 1 |

### Part II: Inside the LLM

Now that you can drive, pop the hood and see what's inside. These chapters cover every step from input to output.

| Ch | Title | You Will Learn | Prerequisites |
|---|-------|---------------|--------------|
| [3](CN/chapter-03-tokenization.md) | Tokens & Tokenization | How text becomes tokens; BPE algorithm; why vocabulary size matters | Ch 2 |
| [4](CN/chapter-04-data.md) | Training Data | Where data comes from; cleaning pipelines; Chinchilla scaling; why quality beats quantity | Ch 3 |
| [5](CN/chapter-05-transformer.md) | Transformer Architecture | The overall architecture; why Decoder-Only won; data flow of forward pass | Ch 4 |
| [6](CN/chapter-06-self-attention.md) | Self-Attention | What attention actually computes; intuitive derivation of QKV; the full math | Ch 5 |
| [7](CN/chapter-07-multihead-causal.md) | Multi-Head Attention & Causal Masks | Why multiple heads; how causal masks prevent seeing the future; MQA and GQA | Ch 6 |
| [8](CN/chapter-08-kv-cache.md) | KV Cache | The core of inference acceleration; why only cache K and V; PagedAttention and continuous batching | Ch 7 |
| [9](CN/chapter-09-positional-encoding.md) | Positional Encoding | How models know word order; sinusoidal, learned, RoPE | Ch 5 |
| [10](CN/chapter-10-norm-activation.md) | Normalization & Activation Functions | LayerNorm vs RMSNorm; ReLU→GELU→SwiGLU evolution; residual connections | Ch 5 |

### Part III: Making LLMs Act

Now that you know how the car works, let it run errands on its own.

| Ch | Title | You Will Learn | Prerequisites |
|---|-------|---------------|--------------|
| [11](CN/chapter-11-agent-loop.md) | Agent Loop & Tool Use | From conversation to action; ReAct loop; function calling protocol; security & permissions | Ch 2 |
| [12](CN/chapter-12-reasoning-planning.md) | Reasoning & Planning | CoT, ToT, slow thinking; how to make LLMs plan multi-step tasks | Ch 1, 11 |
| [13](CN/chapter-13-mcp.md) | MCP Protocol | Standardized protocol for tool calling; server architecture; security considerations | Ch 11 |
| [14](CN/chapter-14-skills.md) | Skill Systems | From fixed tools to discoverable skills; registration & discovery; skill solidification | Ch 11, 13 |
| [15](CN/chapter-15-memory.md) | Memory Mechanisms | Short-term & long-term memory; MemGPT; vector memory & knowledge graphs | Ch 11 |
| [16](CN/chapter-16-subagent.md) | Sub-Agents | One agent calling another; delegation & specialization | Ch 11 |
| [17](CN/chapter-17-multi-agent.md) | Multi-Agent Collaboration | AutoGen, CrewAI, LangGraph; communication & coordination; conflict resolution | Ch 16 |

### Part IV: Making It Run Reliably

The car can drive itself now, but you still need a steering wheel, brakes, and a dashboard.

| Ch | Title | You Will Learn | Prerequisites |
|---|-------|---------------|--------------|
| [18](CN/chapter-18-context-engineering.md) | Context Engineering | Window management; caching strategies; compression & selection; Lost in the Middle | Ch 2, 8 |
| [19](CN/chapter-19-harness-engineering.md) | Harness Engineering | Guardrails & evaluation; observability; safety & alignment; production deployment | All prior |
| [20](CN/chapter-20-future.md) | Frontiers & Outlook | Inference-time compute; multimodal agents; MCP ecosystem; safety frontiers | All prior |

## Language Editions

| Directory | Language | Description |
|-----------|----------|-------------|
| `CN/` | 简体中文 | Original simplified Chinese edition |
| `CN_T/` | 繁體中文 | Traditional Chinese edition |
| `EN/` | English | English edition (translated) |

Each directory contains:
- `chapter-XX-*.md` — Chapter content
- `codes/` — Runnable Python code examples
- `references/` — Paper citations and model specs

## Figure Formats

All figures are available in three formats:

- **SVG**: Referenced directly in Markdown documents, scales without loss
- **PDF**: For TeX conversion later
- **PNG**: Quick preview, 300dpi resolution

Figure files are in the `figures/` directory:

```
figures/
├── svg/          # SVG format (referenced in Markdown)
├── pdf/          # PDF format (for TeX)
└── png/          # PNG format (preview)
```

## Citation Convention

All claims, data, and figures cite their sources in `[Author, Year]` or `[Organization, Year]` format. Full references are provided at the end of each chapter and in `references.md`.

## License

All text content is newly written. All figures are newly created.

## References

See [references.md](CN/references.md) for the complete reference list with innovation notes and importance ratings.

Individual chapter references are at the end of each chapter. PDF download links for core papers are in `references/papers/README.md`.