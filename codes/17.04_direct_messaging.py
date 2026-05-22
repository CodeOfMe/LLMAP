class Agent:
    def __init__(self, name):
        self.name = name
        self.inbox = []
    
    def send(self, recipient, message):
        recipient.inbox.append({
            "from": self.name,
            "content": message,
            "timestamp": __import__('time').time()
        })
    
    def receive(self):
        messages = self.inbox.copy()
        self.inbox.clear()
        return messages

if __name__ == "__main__":
    alice = Agent("Alice")
    bob = Agent("Bob")
    alice.send(bob, "Hello, please help me review this code")
    bob.send(alice, "Sure, I'll take a look right away")
    print(bob.receive())
    print(alice.receive())