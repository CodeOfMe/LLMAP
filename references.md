# References

## Chapter 1: Prompt Engineering

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Language Models are Few-Shot Learners (Brown et al., 2020) | NeurIPS 2020, arXiv:2005.14165 | Proposed GPT-3, demonstrating that large models can perform diverse tasks through few-shot prompting without fine-tuning | The starting point of prompt engineering and the foundation for the entire book |
| 2 | Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022) | NeurIPS 2022, arXiv:2201.11903 | Discovered that prompting models to "think step by step" significantly improves reasoning ability | The core paper on CoT prompting, providing the theoretical foundation for Chapters 1 and 12 |
| 3 | Lost in the Middle (Liu et al., 2023) | arXiv:2307.03172 | Revealed that LLMs' ability to retrieve information from the middle of long contexts declines significantly | A core problem in context engineering, referenced in Chapters 1 and 18 |
| 4 | DSPy: Compiling Declarative Language Model Calls (Khattab et al., 2023) | NeurIPS 2023, arXiv:2310.03714 | Shifted prompts from hand-crafted to programmatic declarations with automatic optimization | The engineering direction for prompt engineering, introduced in Chapter 1 |
| 5 | Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang et al., 2022) | arXiv:2212.10560 | Enabled models to self-generate training instructions for alignment, reducing manual annotation | A foundational method for data synthesis and instruction alignment |
| 6 | Large Language Models are Human-Level Prompt Engineers (Zhou et al., 2022) | arXiv:2211.01910 | Automatically searched for optimal prompts, proving LLMs can optimize prompts themselves | Pioneering work on automatic prompt optimization |
| 7 | Prompting Is All You Need: Automated Prompting for LLMs (Pryzant et al., 2023) | arXiv:2305.11206 | Proposed gradient-based automatic prompt optimization | Another approach to automating prompt engineering |

## Chapter 2: LLM API and Local Deployment

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | vLLM: Efficient Memory Management for LLM Serving (Kwon et al., 2023) | SOSP 2023, arXiv:2309.06180 | Proposed PagedAttention, managing KV Cache with paging, doubling GPU memory utilization | The de facto standard for LLM inference deployment; core of Chapters 2 and 8 |
| 2 | SGLang: Efficient Execution of Structured LLM Programs (Zheng et al., 2024) | arXiv:2312.07104 | Proposed RadixAttention, using prefix trees to reuse KV Cache for faster structured generation | An inference framework alongside vLLM, referenced in Chapters 2 and 18 |
| 3 | Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023) | NeurIPS 2023, arXiv:2302.04761 | Models learned to autonomously call external tools (calculators, search engines, etc.) | A bridge from dialogue to tool usage, connecting Chapters 2 and 11 |
| 4 | Qwen3 Technical Report (Qwen Team, 2025) | https://qwenlm.github.io/blog/qwen3/ | Open-source MoE model with switchable thinking mode | The primary model for code examples in this textbook |
| 5 | LLM.int8(): 8-bit Matrix Multiplication for Transformers (Dettmers et al., 2022) | arXiv:2208.07339 | Proposed 8-bit quantized inference with nearly no accuracy loss, halving GPU memory | An introductory quantization method and key technology for local deployment |
| 6 | QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) | arXiv:2305.14314 | 4-bit quantization + LoRA fine-tuning, enabling 65B model fine-tuning on a single GPU | Made fine-tuning large models accessible to individual developers |
| 7 | GPTQ: Accurate Post-Training Quantization (Frantar et al., 2022) | arXiv:2210.17323 | Post-training quantization to 3-4 bits with per-layer optimal weight allocation | A mainstream method for compression deployment |
| 8 | AWQ: Activation-aware Weight Quantization (Lin et al., 2023) | arXiv:2306.00978 | Protected important weight channels based on activation distributions, maintaining accuracy at 4-bit quantization | An alternative to GPTQ with faster inference |
| 9 | TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (Zandieh et al., 2025) | arXiv:2504.19874 | Online vector quantization, compressing KV Cache to 1-2 bits | A frontier direction in KV Cache compression |

## Chapter 3: Tokens and Tokenization

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016) | ACL 2016, arXiv:1508.07909 | Proposed the BPE tokenization algorithm, solving the out-of-vocabulary word problem in translation | BPE is the foundation for nearly all LLM tokenizers |
| 2 | Language Models Encode Less Information in Chinese Than English (Petrov et al., 2023) | arXiv:2307.01385 | Quantitatively proved that Chinese encoding efficiency is lower than English, requiring 2-3x more tokens for the same content | A pain point in Chinese-language scenarios, directly impacting cost and performance |
| 3 | SentencePiece: A Simple and Language Independent Subword Tokenizer (Kudo & Richardson, 2018) | ACL 2018, arXiv:1808.06226 | Language-independent subword tokenizer supporting direct training from raw text | The standard tokenizer for multilingual models (e.g., LLaMA) |
| 4 | A New Algorithm for Data Compression (Gage, 1994) | The C Users Journal | Proposed the original BPE algorithm (for data compression) | The intellectual origin of BPE |
| 5 | tiktoken: A Fast BPE Tokenizer (OpenAI, 2023) | https://github.com/openai/tiktoken | Fast BPE tokenizer implementation used by the GPT series | Directly used in code examples |
| 6 | On the Tokenization of Chinese Text (Rust et al., 2021) | arXiv:2109.01385 | Systematically analyzed the deficiencies and inconsistencies of Chinese tokenizers | A reference for understanding Chinese tokenization issues |

## Chapter 4: Training Data

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Training Compute-Optimal Large Language Models — Chinchilla (Hoffmann et al., 2022) | arXiv:2203.15556 | Proved model parameters and data should scale proportionally; a 700B model should be trained on more data | A landmark paper that changed the industry's training strategy |
| 2 | Scaling Laws for Neural Language Models (Kaplan et al., 2020) | arXiv:2001.08361 | Discovered power-law relationships between model performance and parameter count, data volume, and compute | The foundational work on scaling laws |
| 3 | The Pile: An 800GB Dataset of Diverse Text (Gao et al., 2020) | arXiv:2101.00027 | Built a high-quality diverse training dataset covering 22 domains | The benchmark for open-source training data |
| 4 | Deduplic Training Data Makes Language Models Better (Lee et al., 2022) | arXiv:2107.06499 | Proved training on deduplicated data improves all metrics, with approximately 14% deduplication rate | A key step in data cleaning pipelines |
| 5 | RefinedWeb Dataset (Penedo et al., 2023) | arXiv:2306.01116 | Curated 6B-document dataset crawled from the web, the data source for the Falcon model | A practical reference for large-scale data cleaning |
| 6 | TinyStories: How Small Can Language Models Be (Eldan & Russinovich, 2023) | arXiv:2305.07759 | Trained a 1.3B model on children's stories, proving data quality can compensate for data quantity | An extreme case demonstrating "data quality matters more than quantity" |
| 7 | The Curse of Recursion: Training on Generated Data (Shumailov et al., 2023) | arXiv:2305.17493 | Proved training AI on AI-generated data leads to model collapse | Data synthesis must be approached with caution |
| 8 | Scaling Language Models — Gopher (Rae et al., 2021) | arXiv:2112.11446 | Detailed the data cleaning pipeline (20+ steps) for a 280B model | An authoritative reference for data engineering practices |
| 9 | Language Models are Unsupervised Multitask Learners (Radford et al., 2019) | OpenAI | GPT-2, demonstrating that large models can perform diverse tasks zero-shot after unsupervised pretraining | The starting point for data-driven learning |
| 10 | Decoupled DiLoCo: Resilient Distributed Pre-training (Douillard et al., 2026) | arXiv:2604.21428 | Decoupled distributed pre-training, solving communication bottlenecks across data centers | The frontier of large-scale training infrastructure |

## Chapter 5: Transformer Architecture

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Attention Is All You Need (Vaswani et al., 2017) | NeurIPS 2017, arXiv:1706.03762 | Proposed the Transformer architecture, entirely abandoning recurrence and convolution for pure attention | The architectural foundation for the entire book and the starting point of modern LLMs |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018) | arXiv:1810.04805 | Bidirectional pretraining with masked language modeling, a new paradigm for NLU tasks | A key reference for understanding pretraining paradigms |
| 3 | T5: Exploring the Limits of Transfer Learning (Raffel et al., 2019) | JMLR 2020, arXiv:1910.10683 | Unified text-to-text framework, a representative Encoder-Decoder architecture | A comparison reference against Decoder-Only architectures |
| 4 | Deep Residual Learning for Image Recognition (He et al., 2015) | CVPR 2016, arXiv:1512.03385 | Proposed residual connections, solving the vanishing gradient problem in deep networks | The theoretical origin of residual connections in Transformers |
| 5 | Using the Output Embedding to Improve Language Models (Press & Wolf, 2016) | arXiv:1608.05859 | Proposed weight tying, sharing parameters between input and output embeddings | A practical technique reducing model parameters by 50% |
| 6 | The LLaMA 3 Herd of Models (Dubey et al., 2024) | arXiv:2407.21783 | Open-source 8B-405B model family with fully disclosed training details | The latest representative of Decoder-Only architecture |
| 7 | LLaMA 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023) | arXiv:2307.09288 | Open-source 7B-70B models, demonstrating that open-source can approach GPT-3.5 | An open-source milestone for the LLaMA series |

## Chapter 6: Self-Attention Mechanism

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022) | NeurIPS 2022, arXiv:2205.14135 | Tiled attention computation avoiding materializing the full attention matrix in GPU memory, reducing IO complexity from O(N²) to O(N²d²/M) | The de facto standard for attention computation acceleration |
| 2 | FlashAttention-2: Faster Attention with Better Parallelism (Dao, 2023) | arXiv:2307.08691 | Improved GPU parallelism strategies, 2-4x faster than FlashAttention | The mainstream attention implementation in production |
| 3 | Generating Long Sequences with Sparse Transformers (Child et al., 2019) | arXiv:1904.10509 | Sparse attention patterns reducing attention complexity from O(N²) to O(N√N) | An early exploration of efficient attention |
| 4 | Transformers are RNNs: Linear Attention (Katharopoulos et al., 2020) | arXiv:2006.16236 | Used kernel functions to approximate softmax, reducing attention complexity to O(N) | A representative method for linear attention |
| 5 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023) | arXiv:2312.00752 | Selective state space model with linear-time inference, far surpassing Transformer on long-sequence reasoning | A representative alternative to the Transformer architecture |
| 6 | BertViz: Visualizing Attention in Transformers (Vig, 2019) | ACL 2019, arXiv:1904.02679 | Visualization tool for understanding what attention weights learn | A pedagogical tool for understanding attention |
| 7 | Linformer: Self-Attention with Linear Complexity (Wang et al., 2020) | arXiv:2006.04768 | Reduced self-attention complexity to linear via low-rank approximation | Another approach to linear attention |
| 8 | Performer: Efficient Attention via Random Features (Choromanski et al., 2020) | arXiv:2009.14794 | Used random features to approximate the softmax kernel, achieving linear attention | Bridging linear attention and exact attention |

## Chapter 7: Multi-Head Attention and Causal Masks

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Fast Transformer Decoding: One Write-Head is All You Need — MQA (Shazeer, 2019) | arXiv:1911.02150 | Multi-head attention with only one head for K and V (multi-query attention), reducing KV Cache by several times during inference | The beginning of KV Cache optimization and the precursor to GQA |
| 2 | GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023) | arXiv:2305.13245 | A middle ground between MHA and MQA, where multiple query heads share one group of KV heads | The attention scheme used in LLaMA 2, mainstream in industry |
| 3 | Gemini: A Family of Highly Capable Multimodal Models (Google, 2023) | arXiv:2312.11805 | Multimodal large model processing text, images, audio, etc. simultaneously | A reference for multimodal attention |

## Chapter 8: KV Cache

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | vLLM: Efficient Memory Management for LLM Serving (Kwon et al., 2023) | SOSP 2023, arXiv:2309.06180 | PagedAttention manages KV Cache with paging, similar to OS virtual memory | A milestone in LLM inference engines |
| 2 | Orca: A Distributed Serving System for Transformer Models (Yu et al., 2022) | arXiv:2201.03848 | Proposed continuous batching, releasing resources as soon as requests complete | Early work on improving GPU utilization |
| 3 | H2O: Heavy-Hitter Oracle for Efficient Generative Inference (Zhang et al., 2023) | arXiv:2306.14098 | Identified and retained "heavy-hitter" token KV, discarding unimportant ones to reduce KV Cache footprint | An important method for KV Cache compression |
| 4 | Mistral 7B (Mistral AI, 2023) | arXiv:2310.06825 | A 7B model using GQA and sliding window attention, achieving performance close to 13B models | A typical example validating GQA in practice |
| 5 | Efficiently Scaling Transformer Inference (Pope et al., 2022) | arXiv:2211.05102 | Analyzed GPU memory bottlenecks during Transformer inference, proposing KV Cache sharding strategies | System-level analysis for large model inference optimization |
| 6 | Speculative Decoding (Chen et al., 2023) | arXiv:2302.01318 | Used a small model to guess and a large model to verify, achieving lossless 2-3x inference speedup | An important direction for inference acceleration |
| 7 | Medusa: Simple LLM Inference Acceleration (Cai et al., 2024) | arXiv:2401.10774 | Multi-head parallel prediction of multiple tokens, a simpler acceleration scheme than speculative decoding | An alternative approach to inference acceleration |
| 8 | TurboQuant: Online Vector Quantization for KV Cache (Zandieh et al., 2025) | arXiv:2504.19874 | Quantized KV Cache to 1-2 bits, dramatically reducing inference memory | A frontier direction in KV Cache compression |

## Chapter 9: Positional Encoding

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | RoFormer: Enhanced Transformer with Rotary Position Embedding — RoPE (Su et al., 2021) | arXiv:2104.09864 | Rotary position encoding, encoding relative positions through complex number rotation, with better extrapolation than absolute position encoding | The standard positional encoding in current mainstream LLMs |
| 2 | ALiBi: Attention with Linear Biases (Press et al., 2021) | arXiv:2108.12409 | Added linear biases to attention scores without position embeddings; training on short sequences enables inference on longer sequences | An early approach for long-context extrapolation |
| 3 | Position Interpolation: Extending Context Window (Chen et al., 2023) | arXiv:2306.15595 | Extended the range of trained positional encodings to longer contexts through interpolation | A practical method for extending context windows |
| 4 | YaRN: Efficient Context Window Extension (Peng et al., 2023) | arXiv:2309.00071 | Combined temperature scaling with interpolation for more stable context extension than naive position interpolation | An improved approach for context extension |
| 5 | BLOOM: A 176B-Parameter Open-Access Multilingual Model (Scao et al., 2022) | arXiv:2211.05100 | A large-scale multilingual open-source model using ALiBi positional encoding | A real-world deployment case of ALiBi |
| 6 | NTK-Aware Scaled RoPE (bloc97, 2023) | https://www.reddit.com/r/LocalLLaMA/comments/14lz7jv/ | Extrapolated context windows by adjusting the base of RoPE, without retraining | An important community-driven discovery |
| 7 | LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (Ding et al., 2024) | arXiv:2402.13753 | Extended context window to 2 million tokens | The latest advancement in positional encoding extrapolation |

## Chapter 10: Normalization and Activation Functions

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Root Mean Square Layer Normalization — RMSNorm (Zhang & Sennrich, 2019) | arXiv:1910.07467 | Used root mean square normalization instead of mean-variance normalization, reducing computation by 10-50% with comparable performance | The standard normalization in mainstream models like LLaMA |
| 2 | On Layer Normalization in the Transformer Architecture — Pre-Norm (Xiong et al., 2020) | arXiv:2002.04745 | Proved Pre-Norm is more stable than Post-Norm for training, eliminating the need for warm-up | Changed how normalization is placed in Transformers |
| 3 | GLU Variants Improve Transformer (Shazeer, 2020) | arXiv:2002.05202 | Proposed SwiGLU activation function combining gating and GLU, outperforming ReLU and GELU | The standard feed-forward network activation in models like LLaMA |
| 4 | Gaussian Error Linear Units — GELUs (Hendrycks & Gimpel, 2016) | arXiv:1606.08415 | Used Gaussian error function as activation, a smooth alternative to ReLU | The activation function used in BERT/GPT series |
| 5 | Searching for Activation Functions — Swish (Ramachandran et al., 2017) | arXiv:1710.05941 | Used automated search to discover Swish activation, which is smooth and non-monotonic | A component of SwiGLU |
| 6 | Layer Normalization (Ba et al., 2016) | arXiv:1607.06450 | Proposed LayerNorm, normalizing along the feature dimension, solving BatchNorm's issues in sequence models | The original normalization scheme for Transformers |

## Chapter 11: Agent Loop and Tool Use

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023) | ICLR 2023, arXiv:2210.03629 | A loop framework combining reasoning (CoT) and action (tool calling) | The core paradigm for Agent Loops |
| 2 | Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023) | NeurIPS 2023, arXiv:2302.04761 | Models learned to autonomously decide when and which tools to call | Pioneering work on models self-learning tool use |
| 3 | Not What You've Signed Up For: Prompt Injection (Greshake et al., 2023) | arXiv:2302.12173 | Systematically analyzed threat models and defense strategies for prompt injection attacks | A must-read paper for Agent security |
| 4 | WebGPT: Browser-Assisted Question-Answering (Nakano et al., 2021) | arXiv:2112.09332 | Enabled models to browse the web and cite sources when answering questions | An early representative of models using external tools |
| 5 | Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023) | arXiv:2305.15334 | Enabled models to learn to call 16,000+ real APIs | Training methods for large-scale API calling |
| 6 | HuggingGPT: Solving AI Tasks with ChatGPT (Shen et al., 2023) | arXiv:2303.17580 | Used LLM as a controller to orchestrate multiple specialized models | An early attempt at multi-model collaboration |
| 7 | OpenAI Function Calling Documentation (OpenAI, 2023) | https://platform.openai.com/docs/guides/function-calling | Structured tool calling protocol definition | The industry standard for function calling |

## Chapter 12: Reasoning and Planning

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022) | NeurIPS 2022, arXiv:2201.11903 | Prompted models to show reasoning steps, significantly improving mathematical and logical reasoning | The foundational paper on CoT reasoning |
| 2 | Self-Consistency Improves Chain of Thought (Wang et al., 2022) | arXiv:2203.11171 | Sampled multiple reasoning paths and took majority vote, more robust than single CoT | A simple method for improving reasoning reliability |
| 3 | Tree of Thoughts: Deliberate Problem Solving (Yao et al., 2023) | arXiv:2305.10601 | Extended reasoning space from chains to trees, supporting backtracking and exploration | Beyond linear reasoning |
| 4 | Reflexion: Language Agents with Verbal RL (Shinn et al., 2023) | arXiv:2303.11366 | Agents reflected on failure causes in natural language and self-improved | Implementation of Agent self-improvement loops |
| 5 | DeepSeek-R1: Incentivizing Reasoning Capability via RL (DeepSeek, 2025) | arXiv:2501.12948 | Stimulated reasoning capabilities through reinforcement learning, a new paradigm for open-source reasoning models | An open-source milestone for reasoning models (o1 class) |
| 6 | o1: Learning to Reason with LLMs (OpenAI, 2024) | https://openai.com/index/learning-to-reason-with-llms/ | The first commercial reasoning model, pioneering test-time compute | The starting point of the test-time compute paradigm |
| 7 | Graph of Thoughts: Solving Elaborate Problems with LLMs (Besta et al., 2023) | arXiv:2308.09687 | Extended reasoning space from trees to graphs, supporting merging of reasoning paths | A more flexible reasoning structure than ToT |
| 8 | RAP: Reasoning via Planning (Hao et al., 2023) | arXiv:2305.14992 | Treated LLM reasoning as a planning problem using world models for simulation | Combining reasoning and planning |

## Chapter 13: MCP Protocol

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Model Context Protocol Specification (Anthropic, 2024) | https://spec.modelcontextprotocol.io/ | Open standard protocol for tool calling, unifying LLM and external service interfaces | The official specification for the MCP protocol |
| 2 | MCPShield: Security Cognition Layer for MCP Agents (Zhou et al., 2026) | arXiv:2602.14281 | Security cognition layer for the MCP protocol, defending against prompt injection and unauthorized operations | Early research on MCP security |
| 3 | Not What You've Signed Up For (Greshake et al., 2023) | arXiv:2302.12173 | Systematic analysis of prompt injection attacks | A reference for MCP security design |

## Chapter 14: Skill Systems

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | From History to State: Constant-Context Skill Learning (Xie et al., 2026) | arXiv:2605.05413 | Automatically extracted skills from conversation history, enabling skill discovery, registration, and consolidation | The theoretical foundation for skill systems |
| 2 | Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023) | arXiv:2305.15334 | Enabled models to learn to call a large number of APIs | Large-scale validation of skill discovery and invocation |
| 3 | ToolLLM: Facilitating LLMs to Master 16000+ Tools (Qin et al., 2023) | arXiv:2307.12857 | Built evaluation benchmarks and training data for 16,000+ tools | A benchmark for skill evaluation |
| 4 | API-Bank: A Benchmark for Tool-Augmented LLMs (Li et al., 2023) | arXiv:2304.08244 | Proposed a hierarchical tool calling evaluation benchmark | A methodology for skill assessment |

## Chapter 15: Memory Mechanisms

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023) | arXiv:2310.08560 | Analogized LLMs to operating systems, managing multi-level memory within limited context windows | The core framework for Agent memory management |
| 2 | Zep: A Temporal Knowledge Graph Architecture for Agent Memory (Rasmussen et al., 2025) | arXiv:2501.13956 | Temporal knowledge graphs as Agent long-term memory, supporting temporal reasoning | A structured storage solution for long-term memory |
| 3 | MIRIX: Multi-Agent Memory System for LLM-Based Agents (Wang & Chen, 2025) | arXiv:2507.07957 | Six-layer memory architecture (sensory, working, short-term, long-term, skill, emotional) | The latest design for multi-layer memory systems |
| 4 | LongBench: A Bilingual, Multitask Benchmark for Long Context (Bai et al., 2023) | arXiv:2308.14508 | Multi-task long-context evaluation benchmark, revealing significant performance degradation on long contexts | The standard benchmark for evaluating long-context capabilities |
| 5 | Retrieval-Augmented Generation — RAG (Lewis et al., 2020) | NeurIPS 2020, arXiv:2005.11401 | Combined retrieval and generation, enabling models to access external knowledge | RAG is the most practical method within memory mechanisms |
| 6 | RETRO: Improving Language Models with Retrieval (Borgeaud et al., 2021) | arXiv:2112.04426 | Integrated retrieval into every Transformer layer, with nearest neighbor chunks influencing each token's generation | A deeply integrated retrieval-augmented generation approach |

## Chapter 16: Sub-Agents

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | MetaGPT: Meta Programming for Multi-Agent Collaborative Framework (Hong et al., 2023) | ICLR 2024, arXiv:2308.00352 | Defined roles and SOPs for multiple Agents to collaborate following software engineering processes | The SOP paradigm for multi-Agent collaboration |
| 2 | Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023) | arXiv:2304.03442 | 25 AI agents autonomously living, interacting, and forming social relationships in a virtual town | A pioneering experiment on Agent social behavior |
| 3 | AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (Wu et al., 2023) | arXiv:2308.08155 | General multi-Agent conversation framework supporting custom Agents and human participation | Infrastructure for multi-Agent frameworks |
| 4 | CrewAI: Building Collaborative AI Agent Teams (CrewAI Team, 2024) | https://github.com/crewAIInc/crewAI | Role-driven multi-Agent collaboration framework emphasizing team division of labor and process management | Industrial practice of multi-Agent frameworks |
| 5 | ChatDev: Communicative Agents for Software Development (Qian et al., 2023) | arXiv:2307.07924 | Used conversation-driven software development with roles like CEO, CTO, and programmer collaborating | Application of multi-Agents in a specific domain |

## Chapter 17: Multi-Agent Collaboration

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Improving Factuality and Reasoning through Multiagent Debate (Du et al., 2023) | arXiv:2305.14325 | Multi-Agent debate can improve factual accuracy and reasoning quality | Empirical evidence that multi-Agent collaboration improves reliability |
| 2 | Generative Agents (Park et al., 2023) | arXiv:2304.03442 | (Same as Chapter 16) | A classic experiment in multi-Agent social simulation |
| 3 | AutoGen (Wu et al., 2023) | arXiv:2308.08155 | (Same as Chapter 16) | Multi-Agent framework |
| 4 | MetaGPT (Hong et al., 2023) | arXiv:2308.00352 | (Same as Chapter 16) | Multi-Agent collaboration SOP |
| 5 | Token Coherence: MESI Cache Protocols for Multi-Agent LLM Systems (Parakhin, 2026) | arXiv:2603.15183 | Introduced CPU cache coherence protocols (MESI) into context management for multi-Agent systems | A new paradigm for multi-Agent context sharing |
| 6 | The Society of Mind (Minsky, 1986) | Simon & Schuster | Intelligence emerges from the combination of many simple Agents | The theoretical origin of multi-Agent intelligence |
| 7 | A Survey on LLM-Based Multi-Agent Systems (Li et al., 2024) | arXiv:2402.01680 | Comprehensive survey of multi-Agent system architectures, collaboration patterns, and evaluation methods | A panoramic map of the multi-Agent field |
| 8 | LatentMAS: Latent Collaboration in Multi-Agent Systems (Zou et al., 2025) | arXiv:2511.20639 | Agents collaborate in latent space, reducing explicit communication overhead | A new direction for multi-Agent collaboration efficiency |

## Chapter 18: Context Engineering

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023) | arXiv:2307.03172 | Found that models' ability to retrieve information from the middle of long contexts declines significantly | The core problem context engineering aims to solve |
| 2 | SGLang: RadixAttention for KV Cache (Zheng et al., 2024) | arXiv:2312.07140 | Used prefix trees to automatically reuse KV Cache for faster structured generation | Core technology for context caching |
| 3 | Irminsul: MLA-Native Position-Independent Caching (Ma et al., 2026) | arXiv:2605.05696 | Position-independent KV caching, enabling cache reuse at different positions | The frontier of KV Cache reuse |
| 4 | LongBench (Bai et al., 2023) | arXiv:2308.14508 | (Same as Chapter 15) | Long-context evaluation |
| 5 | Anthropic Prompt Caching Documentation (Anthropic, 2024) | https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching | Industrial practice of prompt caching | Engineering reference for context caching |
| 6 | Needle in a Haystack: Pressure Testing LLMs (Kamradt, 2023) | https://github.com/gkamradt/LLMTest_NeedleInAHaystack | Needle-in-a-haystack testing: retrieving a single fact from long text | The standard method for long-context stress testing |
| 7 | RULER: What's the Real Context Size? (Hsieh et al., 2024) | arXiv:2404.06654 | A more comprehensive long-context evaluation benchmark than needle-in-a-haystack | A benchmark for evaluating real context capabilities |

## Chapter 19: Harness Engineering

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | Not What You've Signed Up For (Greshake et al., 2023) | arXiv:2302.12173 | (Same as Chapter 11) Systematic analysis of prompt injection attacks | A reference for security protection |
| 2 | SGLang (Zheng et al., 2024) | arXiv:2312.07140 | (Same as Chapters 2 and 18) | Inference engine |
| 3 | MCP Specification (Anthropic, 2024) | https://spec.modelcontextprotocol.io/ | (Same as Chapter 13) | Tool calling standard |
| 4 | LLM Observability: A Survey (Chang et al., 2024) | arXiv:2402.11069 | Comprehensive survey of LLM observability: tracing, metrics, and logging | Theoretical framework for Agent system monitoring |
| 5 | LLM Guardrails: A Survey (Ji et al., 2024) | arXiv:2401.05561 | Comprehensive survey of LLM safety guardrails: content filtering, jailbreak defense, and alignment | A comprehensive reference for security protection |
| 6 | Evaluating LLMs: A Survey (Chang et al., 2023) | arXiv:2309.03214 | Comprehensive survey of LLM evaluation methods: benchmarks, human evaluation, and LLM-as-Judge | A panoramic view of evaluation methodology |
| 7 | LangSmith: LLM Observability Platform (LangChain, 2024) | https://docs.smith.langchain.com/ | Tracing, evaluation, and monitoring platform for LLM applications | Industrial practice for observability |
| 8 | Arize Phoenix: LLM Observability (Arize AI, 2024) | https://docs.arize.com/phoenix | Open-source LLM observability tool | An open-source solution for observability |
| 9 | SynthID-Image: Image Watermarking at Internet Scale (Gowal et al., 2025) | arXiv:2510.09263 | AI-generated image watermarking technology, invisible but detectable | A technical direction for AI content identification |

## Chapter 20: Frontiers and Outlook

| # | Paper | Source | Key Innovation | Significance |
|---|------|------|-----------|--------|
| 1 | DeepSeek-R1: Incentivizing Reasoning Capability (DeepSeek, 2025) | arXiv:2501.12948 | (Same as Chapter 12) Stimulated reasoning capabilities through RL | An open-source milestone for reasoning models |
| 2 | MCPShield (Zhou et al., 2026) | arXiv:2602.14281 | (Same as Chapter 13) MCP security cognition layer | The frontier of Agent security |
| 3 | MIRIX: Multi-Agent Memory System (Wang & Chen, 2025) | arXiv:2507.07957 | (Same as Chapter 15) Six-layer memory architecture | The latest design for memory systems |
| 4 | Token Coherence (Parakhin, 2026) | arXiv:2603.15183 | (Same as Chapter 17) Multi-Agent context consistency | A new paradigm for multi-Agent collaboration |
| 5 | A Survey on LLM-Based Multi-Agent Systems (Li et al., 2024) | arXiv:2402.01680 | (Same as Chapter 17) | A multi-Agent panorama |
| 6 | The Rise and Potential of Large Language Model Based Agents (Xi et al., 2023) | arXiv:2309.07864 | Comprehensive survey of the LLM Agent technology stack, applications, and challenges | An introductory survey for the Agent field |
| 7 | LLM Agents: A Survey (Wang et al., 2024) | arXiv:2401.00801 | Surveyed Agents across four dimensions: architecture, memory, planning, and tool use | A systematic classification of Agents |
| 8 | Towards AGI: A Survey (Goertzel, 2024) | arXiv:2401.00321 | Explored paths and challenges from current LLMs to AGI | Theoretical reflections on long-term directions |
| 9 | Scaling Laws for Neural Language Models (Kaplan et al., 2020) | arXiv:2001.08361 | (Same as Chapter 4) Scaling laws | Theoretical foundation for future development |
| 10 | Emergent Abilities of Large Language Models (Wei et al., 2022) | arXiv:2206.07682 | Discovered that large models suddenly acquire new capabilities beyond certain scale thresholds | Key to understanding the evolution of model capabilities |