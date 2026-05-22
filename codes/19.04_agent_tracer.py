import time
import json

class AgentTracer:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.traces = []
    
    def trace_step(self, step_type, input_data, output_data, metadata=None):
        self.traces.append({
            "agent": self.agent_name,
            "step": len(self.traces),
            "type": step_type,
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
    
    def trace_llm_call(self, prompt, response, tokens_in, tokens_out, latency):
        self.trace_step("llm_call", prompt, response, {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency,
            "cost": tokens_in * 0.0000025 + tokens_out * 0.00001
        })
    
    def trace_tool_call(self, tool_name, args, result, latency):
        self.trace_step("tool_call", {"tool": tool_name, "args": args}, 
                       result, {"latency_ms": latency})
    
    def get_summary(self):
        llm_calls = [t for t in self.traces if t["type"] == "llm_call"]
        tool_calls = [t for t in self.traces if t["type"] == "tool_call"]
        return {
            "total_steps": len(self.traces),
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "total_tokens_in": sum(t["metadata"].get("tokens_in", 0) for t in llm_calls),
            "total_tokens_out": sum(t["metadata"].get("tokens_out", 0) for t in llm_calls),
            "total_cost": sum(t["metadata"].get("cost", 0) for t in llm_calls),
            "total_latency_ms": sum(t["metadata"].get("latency_ms", 0) for t in self.traces),
        }

if __name__ == "__main__":
    tracer = AgentTracer("code_agent")
    tracer.trace_llm_call("Write a sorting function", "def sort(arr):...", 150, 50, 800)
    tracer.trace_tool_call("python_exec", {"code": "sort([3,1,2])"}, "[1,2,3]", 120)
    tracer.trace_llm_call("Explain the sorting result", "Sorted array is [1,2,3]", 80, 30, 500)
    print(json.dumps(tracer.get_summary(), indent=2))