class Blackboard:
    def __init__(self):
        self.data = {}
        self.subscriptions = {}
    
    def write(self, key, value, writer):
        if writer not in self.data.get(key, {}).get("writers", []):
            self.data[key] = {"value": value, "writer": writer}
            self._notify(key, value)
    
    def subscribe(self, key, agent):
        if key not in self.subscriptions:
            self.subscriptions[key] = []
        self.subscriptions[key].append(agent)
    
    def _notify(self, key, value):
        for agent in self.subscriptions.get(key, []):
            agent.notify(key, value)

class MockAgent:
    def __init__(self, name):
        self.name = name
        self.notifications = []
    def notify(self, key, value):
        self.notifications.append({"key": key, "value": value})

if __name__ == "__main__":
    observer = MockAgent("Observer")
    bb = Blackboard()
    bb.subscribe("design_updated", observer)
    bb.write("design_updated", "v2 plan submitted", "Designer")
    print(observer.notifications)