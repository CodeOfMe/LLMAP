import json

tool = {
    "name": "search_database",
    "description": "Search for information in the user database",
    "parameters": {"query": {"type": "string"}}
}

skill = {
    "name": "database_search",
    "trigger": "When the user needs to find user data, account information, or historical records",
    "context_requirements": ["Database connection information", "Query permission level"],
    "description": "Execute precise queries in MySQL user database",
    "parameters": {
        "query": {"type": "string", "description": "SQL query condition"},
        "table": {"type": "string", "enum": ["users", "orders", "logs"]}
    },
    "examples": [
        {"input": "Look up Zhang San's orders", "output": "SELECT * FROM orders WHERE user_id = (SELECT id FROM users WHERE name = 'Zhang San')"},
    ],
    "conflicts": ["cache_search"],
    "priority": 10
}

if __name__ == "__main__":
    print(json.dumps(tool, indent=2, ensure_ascii=False))
    print()
    print(json.dumps(skill, indent=2, ensure_ascii=False))