import time
import json

time.sleep = lambda x: None

_current_behavior = "normal"

def execute_tool(name, args):
    if _current_behavior == "normal":
        return "ok"
    elif _current_behavior == "connection_error":
        raise ConnectionError("Connection timed out")
    elif _current_behavior == "value_error":
        raise ValueError("Missing required parameter: query")

def execute_tool_safe(name, args, timeout=10, retries=2):
    for attempt in range(retries + 1):
        try:
            result = execute_tool(name, args)
            return {"success": True, "result": result}
        except ConnectionError:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return {"success": False, "error": "Connection timed out, please retry later"}
        except ValueError as e:
            return {"success": False, "error": f"Parameter error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unknown error: {type(e).__name__}"}

if __name__ == "__main__":
    _current_behavior = "normal"
    print("=== Normal call ===")
    print(json.dumps(execute_tool_safe("search", {"query": "test"}), ensure_ascii=False))

    _current_behavior = "connection_error"
    print("\n=== Connection error (will retry then return) ===")
    print(json.dumps(execute_tool_safe("search", {"query": "test"}), ensure_ascii=False))

    _current_behavior = "value_error"
    print("\n=== Parameter error ===")
    print(json.dumps(execute_tool_safe("search", {}), ensure_ascii=False))