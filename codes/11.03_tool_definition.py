import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search for information in the user database. Supports searching by name, email, or ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword"
                    },
                    "field": {
                        "type": "string",
                        "enum": ["name", "email", "id"],
                        "description": "Which field to search"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

if __name__ == "__main__":
    print(json.dumps(tools, indent=2, ensure_ascii=False))