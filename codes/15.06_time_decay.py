def time_decay_score(memory, current_time, half_life_hours=24):
    age_hours = (current_time - memory["timestamp"]) / 3600
    return memory["importance"] * (0.5 ** (age_hours / half_life_hours))

if __name__ == "__main__":
    import time
    now = time.time()
    mem = {"importance": 1.0, "timestamp": now}
    
    for hours_ago in [1, 24, 48]:
        m = {"importance": 1.0, "timestamp": now - hours_ago * 3600}
        score = time_decay_score(m, now)
        print(f"{hours_ago} hours ago: {score:.4f}")