def calculate_kv_cache_size(num_layers, num_kv_heads, head_dim, seq_length, 
                            batch_size=1, fp16=True):
    bytes_per_element = 2 if fp16 else 4
    size = 2 * num_layers * num_kv_heads * head_dim * seq_length * bytes_per_element * batch_size
    return size

if __name__ == "__main__":
    models = {
        "LLaMA 3 8B": {"layers": 32, "kv_heads": 8, "head_dim": 128},
        "LLaMA 2 70B": {"layers": 80, "kv_heads": 8, "head_dim": 128},
        "GPT-3 175B": {"layers": 96, "kv_heads": 96, "head_dim": 128},
    }
    for name, config in models.items():
        for seq_len in [2048, 4096, 8192, 32768]:
            size = calculate_kv_cache_size(
                config["layers"], config["kv_heads"], config["head_dim"], seq_len
            )
            print(f"{name} @ {seq_len}K: {size / 1e9:.2f} GB")