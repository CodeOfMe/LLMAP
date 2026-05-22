import json

DATABASE_QUERY_SKILL = {
    "name": "database_query",
    "trigger": "When the user needs to query structured data from a database",
    "description": "Execute SQL queries and return results. Supports SELECT statements, does not support DML.",
    "parameters": {
        "table": {
            "type": "string",
            "enum": ["users", "orders", "products", "logs"],
            "description": "Table to query"
        },
        "conditions": {
            "type": "string",
            "description": "WHERE condition in natural language, system will convert to SQL"
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to query, empty means all columns"
        }
    },
    "implementation": "execute_safe_sql"
}

if __name__ == "__main__":
    print(json.dumps(DATABASE_QUERY_SKILL, indent=2, ensure_ascii=False))