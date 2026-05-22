import builtins

DANGEROUS_OPERATIONS = {"drop_table", "delete_all", "send_email", "execute_shell", "modify_config"}

def execute_tool(name, args):
    return f"Executed {name}"

def execute_tool_with_guard(name, args, require_confirm=True):
    if name in DANGEROUS_OPERATIONS and require_confirm:
        confirm = input(f"Warning: About to execute dangerous operation {name}, confirm? (y/n): ")
        if confirm.lower() != 'y':
            return {"success": False, "error": "User cancelled the operation"}
    return execute_tool(name, args)

if __name__ == "__main__":
    print("=== Safe operation ===")
    result = execute_tool_with_guard("search_database", {})
    print(result)

    print("\n=== Dangerous operation (blocked) ===")
    responses = iter(['n'])
    builtins.input = lambda prompt: next(responses)
    print(execute_tool_with_guard("drop_table", {}))