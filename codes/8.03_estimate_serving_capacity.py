def estimate_serving_capacity(gpu_memory_gb, model_params_b, 
                              seq_length, num_kv_heads, head_dim, 
                              num_layers, fp16=True):
    model_memory = model_params_b * (2 if fp16 else 4)
    kv_per_request = (2 * num_layers * num_kv_heads * head_dim * 
                       seq_length * (2 if fp16 else 4))
    kv_per_request_gb = kv_per_request / 1e9
    available = gpu_memory_gb - model_memory
    if available <= 0:
        return 0
    max_requests = int(available / kv_per_request_gb)
    return max_requests

if __name__ == "__main__":
    requests = estimate_serving_capacity(80, 8, 4096, 8, 128, 32)
    print(f"Max concurrent requests: {requests}")