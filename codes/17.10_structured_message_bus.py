class StructuredMessageBus:
    SCHEMAS = {
        "task_assignment": {
            "task_id": str,
            "description": str,
            "dependencies": list,
            "priority": int
        },
        "task_result": {
            "task_id": str,
            "status": str,
            "output": str,
            "confidence": float
        },
        "conflict_notification": {
            "conflicting_tasks": list,
            "reason": str
        }
    }
    
    def validate_message(self, msg_type, message):
        schema = self.SCHEMAS.get(msg_type)
        if not schema:
            return False
        for field, field_type in schema.items():
            if field not in message:
                return False
            if not isinstance(message[field], field_type):
                return False
        return True

if __name__ == "__main__":
    bus = StructuredMessageBus()
    msg_valid = {"task_id": "t1", "description": "Data cleaning", "dependencies": [], "priority": 1}
    msg_invalid = {"task_id": "t2", "description": 123, "dependencies": [], "priority": 1}
    print(bus.validate_message("task_assignment", msg_valid))
    print(bus.validate_message("task_assignment", msg_invalid))
    print(bus.validate_message("unknown_type", {}))