MODEL_CONFIG = {
    "simple_qa": "qwen3.5:2b",
    "code_review": "qwen3.5:2b",
    "complex_reasoning": "gpt-4o",
}

if __name__ == "__main__":
    print("Model routing configuration:")
    for task_type, model in MODEL_CONFIG.items():
        print(f"  {task_type}: {model}")