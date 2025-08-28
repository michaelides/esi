"""Custom Python REPL tool that captures Plotly figures."""

import sys
import io
from typing import Any, Dict, List, Optional
from langchain.tools.base import BaseTool


# Global list to store captured figures
_captured_figures: List[str] = []


def get_captured_figures() -> List[str]:
    """Get any captured Plotly figures as JSON strings."""
    global _captured_figures
    return _captured_figures.copy()


def clear_captured_figures():
    """Clear the captured figures."""
    global _captured_figures
    _captured_figures.clear()


def capture_figure(fig):
    """Capture a Plotly figure as JSON."""
    global _captured_figures
    try:
        # Convert the figure to JSON
        fig_json = fig.to_json()
        _captured_figures.append(fig_json)
    except Exception as e:
        print(f"Error capturing figure: {e}")


class CustomPythonREPLTool(BaseTool):
    """Custom Python REPL tool that captures Plotly figures."""

    name: str = "python_repl"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`."
    )
    
    globals: Optional[Dict[str, Any]] = None
    locals: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize with common data analysis libraries
        if self.globals is None:
            self.globals = {}
        
        # Import and add common libraries
        try:
            import pandas as pd
            import numpy as np
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Add libraries to globals
            self.globals.update({
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
            })
            
        except ImportError as e:
            print(f"Warning: Could not import some libraries: {e}")

    def _run(self, query: str) -> str:
        """Execute the Python query and capture any figures."""
        # Clear any previously captured figures
        clear_captured_figures()
        
        # Redirect stdout to capture print output
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        
        try:
            # Execute the code
            exec(query, self.globals, self.locals)
            # Get the output
            output = mystdout.getvalue()
            return output
        except Exception as e:
            # Return the error message
            return f"Error: {str(e)}"
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        return self._run(query)