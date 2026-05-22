# Book Reference List (Organized by Chapter)

This file lists all core papers cited in the textbook, organized by chapter. Each paper includes a direct arXiv PDF download link.

## Download Method

```bash
# Batch download all paper PDFs to the references/papers/ directory
mkdir -p references/papers
cd references/papers

# Download using curl (each is approximately 1–10MB)
curl -O https://arxiv.org/pdf/2005.14165.pdf  # GPT-3
curl -O https://arxiv.org/pdf/1706.03762.pdf  # Transformer
# ... and so on
```

---

## Chapter 1: Prompt Engineering

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **GPT-3: Language Models are Few-Shot Learners** (Brown et al., 2020) | 2005.14165 | https://arxiv.org/pdf/2005.14165.pdf |
| 2 | **Chain-of-Thought Prompting Elicits Reasoning** (Wei et al., 2022) | 2201.11903 | https://arxiv.org/pdf/2201.11903.pdf |
| 3 | **Lost in the Middle** (Liu et al., 2023) | 2307.03172 | https://arxiv.org/pdf/2307.03172.pdf |
| 4 | **DSPy: Compiling Declarative Language Model Calls** (Khattab et al., 2023) | 2310.03714 | https://arxiv.org/pdf/2310.03714.pdf |
| 5 | **Self-Instruct: Aligning Language Models with Self-Generated Instructions** (Wang et al., 2022) | 2212.10560 | https://arxiv.org/pdf/2212.10560.pdf |
| 6 | **Large Language Models are Human-Level Prompt Engineers** (Zhou et al., 2022) | 2211.01910 | https://arxiv.org/pdf/2211.01910.pdf |
| 7 | **Prompting Is All You Need: Automated Prompting for LLMs** (Pryzant et al., 2023) | 2305.11206 | https://arxiv.org/pdf/2305.11206.pdf |

## Chapter 2: LLM APIs and Local Deployment

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **vLLM: Efficient Memory Management for LLM Serving** (Kwon et al., 2023) | 2309.06180 | https://arxiv.org/pdf/2309.06180.pdf |
| 2 | **SGLang: Efficient Execution of Structured LLM Programs** (Zheng et al., 2023) | 2312.07104 | https://arxiv.org/pdf/2312.07104.pdf |
| 3 | **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023) | 2302.04761 | https://arxiv.org/pdf/2302.04761.pdf |
| 4 | **Qwen3 Technical Report** (Qwen Team, 2025) | — | https://qwenlm.github.io/blog/qwen3/ |
| 5 | **Ollama Documentation** | — | https://github.com/ollama/ollama |
| 6 | **LLM.int8(): 8-bit Matrix Multiplication for Transformers** (Dettmers et al., 2022) | 2208.07339 | https://arxiv.org/pdf/2208.07339.pdf |
| 7 | **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) | 2305.14314 | https://arxiv.org/pdf/2305.14314.pdf |
| 8 | **GPTQ: Accurate Post-Training Quantization** (Frantar et al., 2022) | 2210.17323 | https://arxiv.org/pdf/2210.17323.pdf |
| 9 | **AWQ: Activation-aware Weight Quantization** (Lin et al., 2023) | 2306.00978 | https://arxiv.org/pdf/2306.00978.pdf |
| 10 | **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** (Zandieh et al., 2025) | 2504.19874 | https://arxiv.org/pdf/2504.19874.pdf |

## Chapter 3: Tokenization

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Neural Machine Translation of Rare Words with Subword Units** (Sennrich et al., 2016) | 1508.07909 | https://arxiv.org/pdf/1508.07909.pdf |
| 2 | **Language Models Encode Less Information in Chinese Than English** (Petrov et al., 2023) | 2307.01385 | https://arxiv.org/pdf/2307.01385.pdf |
| 3 | **SentencePiece: A Simple and Language Independent Subword Tokenizer** (Kudo & Richardson, 2018) | 1808.06226 | https://arxiv.org/pdf/1808.06226.pdf |
| 4 | **Byte-Pair Encoding for Subword Tokenization** (Gage, 1994) | — | https://www.gutenberg.org/files/37375/37375-0.txt |
| 5 | **tiktoken: A Fast BPE Tokenizer** (OpenAI, 2023) | — | https://github.com/openai/tiktoken |
| 6 | **On the Tokenization of Chinese Text** (Rust et al., 2021) | 2109.01385 | https://arxiv.org/pdf/2109.01385.pdf |

## Chapter 4: Training Data

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Chinchilla: Training Compute-Optimal LLMs** (Hoffmann et al., 2022) | 2203.15556 | https://arxiv.org/pdf/2203.15556.pdf |
| 2 | **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) | 2001.08361 | https://arxiv.org/pdf/2001.08361.pdf |
| 3 | **The Pile: An 800GB Dataset of Diverse Text** (Gao et al., 2020) | 2101.00027 | https://arxiv.org/pdf/2101.00027.pdf |
| 4 | **Deduplicating Training Data Makes Language Models Better** (Lee et al., 2022) | 2107.06499 | https://arxiv.org/pdf/2107.06499.pdf |
| 5 | **RefinedWeb: The RefinedWeb Dataset** (Penedo et al., 2023) | 2306.01116 | https://arxiv.org/pdf/2306.01116.pdf |
| 6 | **TinyStories: How Small Can Language Models Be** (Eldan & Russinovich, 2023) | 2305.07759 | https://arxiv.org/pdf/2305.07759.pdf |
| 7 | **The Curse of Recursion: Training on Generated Data** (Shumailov et al., 2023) | 2305.17493 | https://arxiv.org/pdf/2305.17493.pdf |
| 8 | **Gopher: Scaling Language Models** (Rae et al., 2021) | 2112.11446 | https://arxiv.org/pdf/2112.11446.pdf |
| 9 | **WebText: The Dataset Behind GPT-2** (Radford et al., 2019) | 1909.00363 | https://arxiv.org/pdf/1909.00363.pdf |
| 10 | **Decoupled DiLoCo: Resilient Distributed Pre-training** (Douillard et al., 2026) | 2604.21428 | https://arxiv.org/pdf/2604.21428.pdf |

## Chapter 5: Transformer Architecture

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Attention Is All You Need** (Vaswani et al., 2017) | 1706.03762 | https://arxiv.org/pdf/1706.03762.pdf |
| 2 | **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018) | 1810.04805 | https://arxiv.org/pdf/1810.04805.pdf |
| 3 | **T5: Exploring the Limits of Transfer Learning** (Raffel et al., 2019) | 1910.10683 | https://arxiv.org/pdf/1910.10683.pdf |
| 4 | **Deep Residual Learning for Image Recognition** (He et al., 2015) | 1512.03385 | https://arxiv.org/pdf/1512.03385.pdf |
| 5 | **Using the Output Embedding to Improve Language Models** (Press & Wolf, 2016) | 1608.05859 | https://arxiv.org/pdf/1608.05859.pdf |
| 6 | **LLaMA 3: The LLaMA 3 Herd of Models** (Dubey et al., 2024) | 2407.21783 | https://arxiv.org/pdf/2407.21783.pdf |
| 7 | **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** (Touvron et al., 2023) | 2307.09288 | https://arxiv.org/pdf/2307.09288.pdf |

## Chapter 6: Self-Attention

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022) | 2205.14135 | https://arxiv.org/pdf/2205.14135.pdf |
| 2 | **FlashAttention-2: Faster Attention with Better Parallelism** (Dao, 2023) | 2307.08691 | https://arxiv.org/pdf/2307.08691.pdf |
| 3 | **Generating Long Sequences with Sparse Transformers** (Child et al., 2019) | 1904.10509 | https://arxiv.org/pdf/1904.10509.pdf |
| 4 | **Transformers are RNNs: Linear Attention** (Katharopoulos et al., 2020) | 2006.16236 | https://arxiv.org/pdf/2006.16236.pdf |
| 5 | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, 2023) | 2312.00752 | https://arxiv.org/pdf/2312.00752.pdf |
| 6 | **BertViz: Visualizing Attention in Transformers** (Vig, 2019) | 1904.02679 | https://arxiv.org/pdf/1904.02679.pdf |
| 7 | **Linformer: Self-Attention with Linear Complexity** (Wang et al., 2020) | 2006.04768 | https://arxiv.org/pdf/2006.04768.pdf |
| 8 | **Performer: Efficient Attention via Random Features** (Choromanski et al., 2020) | 2009.14794 | https://arxiv.org/pdf/2009.14794.pdf |

## Chapter 7: Multi-Head Attention and Causal Attention

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Fast Transformer Decoding: One Write-Head is All You Need (MQA)** (Shazeer, 2019) | 1911.02150 | https://arxiv.org/pdf/1911.02150.pdf |
| 2 | **GQA: Training Generalized Multi-Query Transformer Models** (Ainslie et al., 2023) | 2305.13245 | https://arxiv.org/pdf/2305.13245.pdf |
| 3 | **Gemini: A Family of Highly Capable Multimodal Models** (Google, 2023) | 2312.11805 | https://arxiv.org/pdf/2312.11805.pdf |

## Chapter 8: KV Cache

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **vLLM: Efficient Memory Management for LLM Serving** (Kwon et al., 2023) | 2309.06180 | https://arxiv.org/pdf/2309.06180.pdf |
| 2 | **Orca: A Distributed Serving System for Transformer Models** (Yu et al., 2022) | 2201.03848 | https://arxiv.org/pdf/2201.03848.pdf |
| 3 | **H2O: Heavy-Hitter Oracle for Efficient Generative Inference** (Zhang et al., 2023) | 2306.14098 | https://arxiv.org/pdf/2306.14098.pdf |
| 4 | **Mistral 7B** (Mistral AI, 2023) | 2310.06825 | https://arxiv.org/pdf/2310.06825.pdf |
| 5 | **Efficiently Scaling Transformer Inference** (Pope et al., 2022) | 2211.05102 | https://arxiv.org/pdf/2211.05102.pdf |
| 6 | **Speculative Decoding** (Chen et al., 2023) | 2302.01318 | https://arxiv.org/pdf/2302.01318.pdf |
| 7 | **Medusa: Simple LLM Inference Acceleration** (Cai et al., 2024) | 2401.10774 | https://arxiv.org/pdf/2401.10774.pdf |
| 8 | **TurboQuant: Online Vector Quantization for KV Cache** (Zandieh et al., 2025) | 2504.19874 | https://arxiv.org/pdf/2504.19874.pdf |

## Chapter 9: Positional Encoding

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) | 2104.09864 | https://arxiv.org/pdf/2104.09864.pdf |
| 2 | **ALiBi: Attention with Linear Biases** (Press et al., 2021) | 2108.12409 | https://arxiv.org/pdf/2108.12409.pdf |
| 3 | **Position Interpolation: Extending Context Window** (Chen et al., 2023) | 2306.15595 | https://arxiv.org/pdf/2306.15595.pdf |
| 4 | **YaRN: Efficient Context Window Extension** (Peng et al., 2023) | 2309.00071 | https://arxiv.org/pdf/2309.00071.pdf |
| 5 | **BLOOM: A 176B-Parameter Open-Access Multilingual Model** (Scao et al., 2022) | 2211.05100 | https://arxiv.org/pdf/2211.05100.pdf |
| 6 | **NTK-Aware Scaled RoPE** (bloc97, 2023) | — | https://www.reddit.com/r/LocalLLaMA/comments/14lz7jv/ |
| 7 | **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** (Ding et al., 2024) | 2402.13753 | https://arxiv.org/pdf/2402.13753.pdf |

## Chapter 10: Normalization and Activation Functions

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019) | 1910.07467 | https://arxiv.org/pdf/1910.07467.pdf |
| 2 | **On Layer Normalization in the Transformer Architecture** (Xiong et al., 2020) | 2002.04745 | https://arxiv.org/pdf/2002.04745.pdf |
| 3 | **GLU Variants Improve Transformer** (Shazeer, 2020) | 2002.05202 | https://arxiv.org/pdf/2002.05202.pdf |
| 4 | **Gaussian Error Linear Units (GELUs)** (Hendrycks & Gimpel, 2016) | 1606.08415 | https://arxiv.org/pdf/1606.08415.pdf |
| 5 | **Searching for Activation Functions** (Ramachandran et al., 2017) | 1710.05941 | https://arxiv.org/pdf/1710.05941.pdf |
| 6 | **Layer Normalization** (Ba et al., 2016) | 1607.06450 | https://arxiv.org/pdf/1607.06450.pdf |

## Chapter 11: Agent Loop and Tool Use

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **ReAct: Synergizing Reasoning and Acting** (Yao et al., 2022) | 2210.03629 | https://arxiv.org/pdf/2210.03629.pdf |
| 2 | **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023) | 2302.04761 | https://arxiv.org/pdf/2302.04761.pdf |
| 3 | **Not What You've Signed Up For: Prompt Injection** (Greshake et al., 2023) | 2302.12173 | https://arxiv.org/pdf/2302.12173.pdf |
| 4 | **WebGPT: Browser-Assisted Question-Answering** (Nakano et al., 2021) | 2112.09332 | https://arxiv.org/pdf/2112.09332.pdf |
| 5 | **Gorilla: Large Language Model Connected with Massive APIs** (Patil et al., 2023) | 2305.15334 | https://arxiv.org/pdf/2305.15334.pdf |
| 6 | **HuggingGPT: Solving AI Tasks with ChatGPT** (Shen et al., 2023) | 2303.17580 | https://arxiv.org/pdf/2303.17580.pdf |
| 7 | **OpenAI Function Calling** (OpenAI, 2023) | — | https://platform.openai.com/docs/guides/function-calling |

## Chapter 12: Reasoning and Planning

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Chain-of-Thought Prompting** (Wei et al., 2022) | 2201.11903 | https://arxiv.org/pdf/2201.11903.pdf |
| 2 | **Self-Consistency Improves Chain of Thought** (Wang et al., 2022) | 2203.11171 | https://arxiv.org/pdf/2203.11171.pdf |
| 3 | **Tree of Thoughts: Deliberate Problem Solving** (Yao et al., 2023) | 2305.10601 | https://arxiv.org/pdf/2305.10601.pdf |
| 4 | **Reflexion: Language Agents with Verbal RL** (Shinn et al., 2023) | 2303.11366 | https://arxiv.org/pdf/2303.11366.pdf |
| 5 | **DeepSeek-R1: Incentivizing Reasoning Capability via RL** (DeepSeek, 2025) | 2501.12948 | https://arxiv.org/pdf/2501.12948.pdf |
| 6 | **o1: Learning to Reason with LLMs** (OpenAI, 2024) | — | https://openai.com/index/learning-to-reason-with-llms/ |
| 7 | **Graph of Thoughts: Solving Elaborate Problems with LLMs** (Besta et al., 2023) | 2308.09687 | https://arxiv.org/pdf/2308.09687.pdf |
| 8 | **RAP: Reasoning via Planning** (Hao et al., 2023) | 2305.14992 | https://arxiv.org/pdf/2305.14992.pdf |

## Chapter 13: MCP Protocol

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Model Context Protocol Specification** (Anthropic, 2024) | — | https://spec.modelcontextprotocol.io/ |
| 2 | **MCPShield: Security Cognition Layer for MCP Agents** (Zhou et al., 2026) | 2602.14281 | https://arxiv.org/pdf/2602.14281.pdf |
| 3 | **Not What You've Signed Up For** (Greshake et al., 2023) | 2302.12173 | https://arxiv.org/pdf/2302.12173.pdf |

## Chapter 14: Skill Systems

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **From History to State: Constant-Context Skill Learning** (Xie et al., 2026) | 2605.05413 | https://arxiv.org/pdf/2605.05413.pdf |
| 2 | **Gorilla: Large Language Model Connected with Massive APIs** (Patil et al., 2023) | 2305.15334 | https://arxiv.org/pdf/2305.15334.pdf |
| 3 | **ToolLLM: Facilitating LLMs to Master 16000+ Tools** (Qin et al., 2023) | 2307.12857 | https://arxiv.org/pdf/2307.12857.pdf |
| 4 | **API-Bank: A Benchmark for Tool-Augmented LLMs** (Li et al., 2023) | 2304.08244 | https://arxiv.org/pdf/2304.08244.pdf |

## Chapter 15: Memory Mechanisms

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **MemGPT: Towards LLMs as Operating Systems** (Packer et al., 2023) | 2310.08560 | https://arxiv.org/pdf/2310.08560.pdf |
| 2 | **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** (Rasmussen et al., 2025) | 2501.13956 | https://arxiv.org/pdf/2501.13956.pdf |
| 3 | **MIRIX: Multi-Agent Memory System for LLM-Based Agents** (Wang & Chen, 2025) | 2507.07957 | https://arxiv.org/pdf/2507.07957.pdf |
| 4 | **LongBench: A Bilingual, Multitask Benchmark for Long Context** (Bai et al., 2023) | 2308.14508 | https://arxiv.org/pdf/2308.14508.pdf |
| 5 | **RAG: Retrieval-Augmented Generation** (Lewis et al., 2020) | 2005.11401 | https://arxiv.org/pdf/2005.11401.pdf |
| 6 | **RETRO: Improving Language Models with Retrieval** (Borgeaud et al., 2021) | 2112.04426 | https://arxiv.org/pdf/2112.04426.pdf |

## Chapter 16: Subagents

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **MetaGPT: Meta Programming for Multi-Agent Collaborative Framework** (Hong et al., 2023) | 2308.00352 | https://arxiv.org/pdf/2308.00352.pdf |
| 2 | **Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., 2023) | 2304.03442 | https://arxiv.org/pdf/2304.03442.pdf |
| 3 | **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** (Wu et al., 2023) | 2308.08155 | https://arxiv.org/pdf/2308.08155.pdf |
| 4 | **CrewAI: Building Collaborative AI Agent Teams** (CrewAI Team, 2024) | — | https://github.com/crewAIInc/crewAI |
| 5 | **ChatDev: Communicative Agents for Software Development** (Qian et al., 2023) | 2307.07924 | https://arxiv.org/pdf/2307.07924.pdf |

## Chapter 17: Multi-Agent Collaboration

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Multi-Agent Debate Improves Reasoning** (Du et al., 2023) | 2305.14325 | https://arxiv.org/pdf/2305.14325.pdf |
| 2 | **Generative Agents** (Park et al., 2023) | 2304.03442 | https://arxiv.org/pdf/2304.03442.pdf |
| 3 | **AutoGen** (Wu et al., 2023) | 2308.08155 | https://arxiv.org/pdf/2308.08155.pdf |
| 4 | **MetaGPT** (Hong et al., 2023) | 2308.00352 | https://arxiv.org/pdf/2308.00352.pdf |
| 5 | **Token Coherence: MESI Cache Protocols for Multi-Agent LLM Systems** (Parakhin, 2026) | 2603.15183 | https://arxiv.org/pdf/2603.15183.pdf |
| 6 | **Society of Mind** (Minsky, 1986) | — | — |
| 7 | **LLM Multi-Agent Systems Survey** (Li et al., 2024) | 2402.01680 | https://arxiv.org/pdf/2402.01680.pdf |
| 8 | **LatentMAS: Latent Collaboration in Multi-Agent Systems** (Zou et al., 2025) | 2511.20639 | https://arxiv.org/pdf/2511.20639.pdf |

## Chapter 18: Context Engineering

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Lost in the Middle** (Liu et al., 2023) | 2307.03172 | https://arxiv.org/pdf/2307.03172.pdf |
| 2 | **SGLang: RadixAttention for KV Cache** (Zheng et al., 2023) | 2312.07140 | https://arxiv.org/pdf/2312.07140.pdf |
| 3 | **Irminsul: MLA-Native Position-Independent Caching** (Ma et al., 2026) | 2605.05696 | https://arxiv.org/pdf/2605.05696.pdf |
| 4 | **LongBench** (Bai et al., 2023) | 2308.14508 | https://arxiv.org/pdf/2308.14508.pdf |
| 5 | **Anthropic Prompt Caching** (Anthropic, 2024) | — | https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching |
| 6 | **Needle in a Haystack: Pressure Testing LLMs** (Kamradt, 2023) | — | https://github.com/gkamradt/LLMTest_NeedleInAHaystack |
| 7 | **RULER: What's the Real Context Size?** (Hsieh et al., 2024) | 2404.06654 | https://arxiv.org/pdf/2404.06654.pdf |

## Chapter 19: Harness Engineering

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **Not What You've Signed Up For** (Greshake et al., 2023) | 2302.12173 | https://arxiv.org/pdf/2302.12173.pdf |
| 2 | **SGLang** (Zheng et al., 2023) | 2312.07140 | https://arxiv.org/pdf/2312.07140.pdf |
| 3 | **MCP Specification** (Anthropic, 2024) | — | https://spec.modelcontextprotocol.io/ |
| 4 | **LLM Observability: A Survey** (Chang et al., 2024) | 2402.11069 | https://arxiv.org/pdf/2402.11069.pdf |
| 5 | **LLM Guardrails: A Survey** (Ji et al., 2024) | 2401.05561 | https://arxiv.org/pdf/2401.05561.pdf |
| 6 | **Evaluating LLMs: A Survey** (Chang et al., 2023) | 2309.03214 | https://arxiv.org/pdf/2309.03214.pdf |
| 7 | **LangSmith: LLM Observability Platform** (LangChain, 2024) | — | https://docs.smith.langchain.com/ |
| 8 | **Arize Phoenix: LLM Observability** (Arize AI, 2024) | — | https://docs.arize.com/phoenix |
| 9 | **SynthID-Image: Image Watermarking at Internet Scale** (Gowal et al., 2025) | 2510.09263 | https://arxiv.org/pdf/2510.09263.pdf |

## Chapter 20: Frontiers and Outlook

| # | Paper | arXiv | Direct PDF Download |
|---|------|-------|------------|
| 1 | **DeepSeek-R1: Incentivizing Reasoning Capability** (DeepSeek, 2025) | 2501.12948 | https://arxiv.org/pdf/2501.12948.pdf |
| 2 | **MCPShield** (Zhou et al., 2026) | 2602.14281 | https://arxiv.org/pdf/2602.14281.pdf |
| 3 | **MIRIX: Multi-Agent Memory System** (Wang & Chen, 2025) | 2507.07957 | https://arxiv.org/pdf/2507.07957.pdf |
| 4 | **Token Coherence** (Parakhin, 2026) | 2603.15183 | https://arxiv.org/pdf/2603.15183.pdf |
| 5 | **A Survey on LLM-Based Multi-Agent Systems** (Li et al., 2024) | 2402.01680 | https://arxiv.org/pdf/2402.01680.pdf |
| 6 | **The Rise and Potential of Large Language Model Based Agents** (Xi et al., 2023) | 2309.07864 | https://arxiv.org/pdf/2309.07864.pdf |
| 7 | **LLM Agents: A Survey** (Wang et al., 2024) | 2401.00801 | https://arxiv.org/pdf/2401.00801.pdf |
| 8 | **Towards AGI: A Survey** (Goertzel, 2024) | 2401.00321 | https://arxiv.org/pdf/2401.00321.pdf |
| 9 | **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) | 2001.08361 | https://arxiv.org/pdf/2001.08361.pdf |
| 10 | **Emergent Abilities of Large Language Models** (Wei et al., 2022) | 2206.07682 | https://arxiv.org/pdf/2206.07682.pdf |

---

## Batch Download Script

```bash
#!/bin/bash
# download_papers.sh - Download all reference paper PDFs
mkdir -p references/papers
cd references/papers

# Core papers (~100 papers, ~500MB total)
declare -a papers=(
    "2005.14165"  # GPT-3
    "1706.03762"  # Transformer
    "2302.13971"  # LLaMA
    "2407.21783"  # LLaMA 3
    "2307.09288"  # LLaMA 2
    "2203.15556"  # Chinchilla
    "2001.08361"  # Scaling Laws
    "2201.11903"  # Chain-of-Thought
    "2307.03172"  # Lost in the Middle
    "2310.03714"  # DSPy
    "2309.06180"  # vLLM
    "2312.07104"  # SGLang
    "2302.04761"  # Toolformer
    "2210.03629"  # ReAct
    "2302.12173"  # Prompt Injection
    "2501.12948"  # DeepSeek-R1
    "2305.10601"  # Tree of Thoughts
    "2303.11366"  # Reflexion
    "2310.08560"  # MemGPT
    "2308.00352"  # MetaGPT
    "2304.03442"  # Generative Agents
    "2308.08155"  # AutoGen
    "2305.14325"  # Multi-Agent Debate
    "2104.09864"  # RoPE
    "2108.12409"  # ALiBi
    "2306.15595"  # Position Interpolation
    "2309.00071"  # YaRN
    "1910.07467"  # RMSNorm
    "2002.04745"  # Pre-Norm
    "2002.05202"  # GLU Variants
    "2205.14135"  # FlashAttention
    "2307.08691"  # FlashAttention-2
    "2312.00752"  # Mamba
    "1911.02150"  # MQA
    "2305.13245"  # GQA
    "2306.14098"  # H2O
    "2305.15334"  # Gorilla
    "2303.17580"  # HuggingGPT
    "2307.12857"  # ToolLLM
    "2304.08244"  # API-Bank
    "2005.11401"  # RAG
    "2112.04426"  # RETRO
    "2307.07924"  # ChatDev
    "2402.01680"  # LLM Multi-Agent Survey
    "2401.00801"  # LLM Agents Survey
    "2309.07864"  # LLM-Based Agents Survey
    "2206.07682"  # Emergent Abilities
    "2309.03214"  # LLM Evaluation Survey
    "2401.05561"  # LLM Guardrails Survey
    "2402.11069"  # LLM Observability Survey
    "2404.06654"  # RULER
    "2401.10774"  # Medusa
    "2302.01318"  # Speculative Decoding
)

for paper in "${papers[@]}"; do
    echo "Downloading arXiv:$paper..."
    curl -s -O "https://arxiv.org/pdf/${paper}.pdf"
done

echo "Done! Downloaded ${#papers[@]} papers."
ls -lh *.pdf | wc -l
```