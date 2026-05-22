import time
import json

class KeyValueMemory:
    def __init__(self):
        self.store = {}
    
    def save(self, key, value):
        self.store[key] = {"value": value, "timestamp": time.time()}
    
    def load(self, key):
        return self.store.get(key, {}).get("value")

if __name__ == "__main__":
    mem = KeyValueMemory()
    mem.save("theme", "dark")
    mem.save("language", "python")
    print(mem.load("theme"))
    print(mem.load("language"))
    print(mem.load("nonexistent"))