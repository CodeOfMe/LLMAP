# Model Specification Reference Data (Sources: Ollama Official + Qwen Official Blog)

Data retrieved: May 2026

## Qwen3 Series

Sources: https://ollama.com/library/qwen3 , https://qwenlm.github.io/blog/qwen3/

### Dense Models

| Model | Parameters | Ollama Download Size (Q4_K_M) | Context Length | Tie Embedding | Notes |
|------|--------|------------------------|-----------|--------------|---------|
| Qwen3-0.6B | 0.6B | 523MB | 32K | Yes | |
| Qwen3-1.7B | 1.7B | 1.4GB | 32K | Yes | |
| Qwen3-4B | 4B | 2.5GB | 32K | Yes | |
| Qwen3-8B | 8B | 5.2GB | 128K | No | Default Ollama tag points to this model |
| Qwen3-14B | 14B | 9.3GB | 128K | No | |
| Qwen3-32B | 32B | 20GB | 128K | No | |

### MoE Models

| Model | Total Parameters | Active Parameters | Ollama Download Size (Q4_K_M) | Context Length | # Experts (Total/Activated) |
|------|--------|---------|------------------------|-----------|---------------------------|
| Qwen3-30B-A3B | 30B | 3B | 19GB | 128K | 128/8 |
| Qwen3-235B-A22B | 235B | 22B | 142GB | 128K | 128/8 |

### Key Features

- Hybrid thinking mode: thinking mode (deep reasoning) + non-thinking mode (fast response)
- Supports /think and /no_think soft switches
- Supports 119 languages and dialects
- Apache 2.0 license
- Pre-training data: approximately 36 trillion tokens
- Vocabulary size: 151,936

## Qwen3.5 Series

Source: https://ollama.com/library/qwen3.5

### Model List

| Model | Parameters | Ollama Download Size (Q4_K_M) | Context Length | Modality | Notes |
|------|--------|------------------------|-----------|------|---------|
| Qwen3.5-0.8B | 0.8B | 1.0GB | 256K | Text+Image | |
| Qwen3.5-2B | 2B | 2.7GB | 256K | Text+Image | |
| Qwen3.5-4B | 4B | 3.4GB | 256K | Text+Image | |
| Qwen3.5-9B | 9B | 6.6GB | 256K | Text+Image | Default Ollama tag |
| Qwen3.5-27B | 27B | 17GB | 256K | Text+Image | |
| Qwen3.5-35B | 35B | 24GB | 256K | Text+Image | |
| Qwen3.5-122B | 122B | 81GB | 256K | Text+Image | |

### MoE Models

| Model | Total Parameters | Active Parameters | Modality |
|------|--------|---------|------|
| Qwen3.5-397B-A17B | 397B | 17B | Text+Image |

### Key Features

- Unified vision-language foundation model (multimodal, supports image input)
- Gated Delta Networks + sparse MoE hybrid architecture
- Supports thinking mode and tool calling
- Supports 201 languages and dialects
- 256K context window
- Vocabulary size: 248,320

## Hardware Requirements Estimate

The following estimates are based on actual Ollama download sizes (Q4_K_M quantization) + KV cache estimates:

| Model | Ollama Q4 Size | Minimum VRAM (incl. KV cache 8K) | Recommended GPU |
|------|-------------|----------------------|---------|
| qwen3.5:0.8b | 1.0GB | ~1.5GB | Integrated graphics sufficient |
| qwen3.5:2b | 2.7GB | ~3.5GB | Integrated graphics sufficient |
| qwen3.5:4b | 3.4GB | ~4.5GB | Integrated graphics sufficient |
| qwen3.5:9b | 6.6GB | ~8GB | RTX 4060 8GB |
| qwen3.5:27b | 17GB | ~20GB | RTX 4090 24GB |
| qwen3.5:35b | 24GB | ~28GB | 2×RTX 4090 or 1×A100 80GB |
| qwen3.5:122b | 81GB | ~90GB | 2×A100 80GB |

Note: Minimum VRAM = model size + KV cache (~0.5–1GB for 8K context) + system overhead

## Default Model Used in This Chapter

`ollama pull qwen3.5:2b`

- Download size: 2.7GB
- 256K context window
- Multimodal (supports image input)
- Supports thinking mode and tool calling