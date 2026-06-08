import math
import statistics
from langchain_core.tools import tool

SAFE_GLOBALS = {
    "__builtins__": {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "range": range, "int": int,
        "float": float, "str": str, "list": list, "print": print,
        "True": True, "False": False, "None": None,
    },
    "math": math,
    "statistics": statistics,
}

def run_code(code: str) -> dict:
    output = []

    def capture(*args):
        output.append(" ".join(str(a) for a in args))

    ns = {**SAFE_GLOBALS, "__builtins__": {**SAFE_GLOBALS["__builtins__"], "print": capture}}

    try:
        exec(compile(code, "<code>", "exec"), ns)
        if not output:
            last = code.strip().split("\n")[-1]
            try:
                val = eval(last, ns)
                if val is not None:
                    output.append(str(val))
            except Exception:
                pass
        return {"success": True, "output": "\n".join(output) or "Done"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def code_executor_tool(code: str) -> dict:
    """Run Python code for math and calculations. Has access to math and statistics libraries. Use print() to show results."""
    return run_code(code)